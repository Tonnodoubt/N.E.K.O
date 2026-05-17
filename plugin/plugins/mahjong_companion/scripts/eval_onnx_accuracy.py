"""Evaluate ONNX tile classification against multi-theme ground truth.

This script intentionally reuses the current runtime crop helpers instead of
old brightness-scan snippets. It measures classifier accuracy on the crops the
plugin actually feeds to the ONNX backend:

* hand crops from ``build_hand_layout`` with the detected hand baseline
* discard crops from ``crop_discard_slot`` / ``build_discard_layout``

Use ``MAHJONG_COMPANION_VIT_ONNX_DIR`` to point at a candidate export.
"""

from __future__ import annotations

from collections import Counter
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURES_ROOT = _REPO_ROOT / "plugin" / "plugins" / "mahjong_companion" / "tests" / "fixtures" / "multi_theme"

sys.path.insert(0, str(_REPO_ROOT))
from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.discard_parser import crop_discard_slot
from plugin.plugins.mahjong_companion.perception.hand_baseline import detect_hand_baseline
from plugin.plugins.mahjong_companion.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch import detect_red_five
from plugin.plugins.mahjong_companion.perception.vit_tile_classifier_onnx import classify_tile_crops_onnx
from plugin.plugins.mahjong_companion.tile_labels import normalize_tile


PLAYERS = ("self", "left_opponent", "top_opponent", "right_opponent")
RED_FIVE_TO_BASE = {"0m": "5m", "0p": "5p", "0s": "5s"}


@dataclass
class Metrics:
    correct: int = 0
    expected: int = 0
    predicted: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.expected if self.expected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> int:
    cases = _collect_cases()
    print(f"Evaluating {len(cases)} screenshots\n")

    hand_total = Metrics()
    discard_total = Metrics()
    per_theme: dict[str, dict[str, Metrics]] = {}

    for theme_id, png_path, sidecar in cases:
        gt = json.loads(sidecar.read_text(encoding="utf-8"))
        image = Image.open(png_path).convert("RGB")

        hand = _score_hand(image, gt)
        discard = _score_discards(image, gt)
        _accumulate(hand_total, hand)
        _accumulate(discard_total, discard)
        bucket = per_theme.setdefault(theme_id, {"hand": Metrics(), "discard": Metrics()})
        _accumulate(bucket["hand"], hand)
        _accumulate(bucket["discard"], discard)

        print(
            f"  {theme_id}/{png_path.stem}: "
            f"hand {hand.correct}/{hand.expected} "
            f"discard {discard.correct}/{discard.expected}"
        )

    print("\n" + "=" * 72)
    print("ONNX CROP SUMMARY")
    print("=" * 72)
    print(f"{'theme':<24} {'hand':>16} {'discard':>16}")
    for theme_id, bucket in sorted(per_theme.items()):
        print(
            f"{theme_id:<24} "
            f"{_fmt(bucket['hand']):>16} "
            f"{_fmt(bucket['discard']):>16}"
        )

    overall = Metrics()
    _accumulate(overall, hand_total)
    _accumulate(overall, discard_total)
    print("-" * 72)
    print(f"{'OVERALL':<24} {_fmt(hand_total):>16} {_fmt(discard_total):>16}")
    print(f"{'ALL CROPS':<24} {_fmt(overall):>16}")
    return 0


def _collect_cases() -> list[tuple[str, Path, Path]]:
    cases: list[tuple[str, Path, Path]] = []
    for theme_dir in sorted(_FIXTURES_ROOT.iterdir()):
        if not theme_dir.is_dir() or theme_dir.name.startswith("_"):
            continue
        for png in sorted(theme_dir.glob("*.png")):
            sidecar = png.with_suffix(".tiles.json")
            if sidecar.exists():
                cases.append((theme_dir.name, png, sidecar))
    return cases


def _score_hand(image: Image.Image, gt: dict[str, Any]) -> Metrics:
    expected = [_base_tile(t) for t in gt.get("hand_tiles", []) if _base_tile(t)]
    if not expected:
        return Metrics()
    baseline = detect_hand_baseline(image)
    layout = build_hand_layout(*image.size, baseline=baseline)
    crops = []
    for slot in layout.get("hand", [])[: len(expected)]:
        box = slot.box
        crops.append(image.crop((box.left, box.top, box.right, box.bottom)))
    predicted = _classify_batch(crops, hand=True)
    return _score_positionally(expected, predicted)


def _score_discards(image: Image.Image, gt: dict[str, Any]) -> Metrics:
    baseline = detect_hand_baseline(image)
    layout = build_discard_layout(*image.size, baseline=baseline)
    crops: list[Image.Image] = []
    expected: list[str] = []
    gt_discards = gt.get("discard_piles", {})
    for player in PLAYERS:
        gt_map = _discard_turn_map(gt_discards.get(player, []))
        for slot in layout.get(player, []):
            tile = gt_map.get(slot.turn_index)
            if not tile:
                continue
            expected.append(tile)
            crops.append(crop_discard_slot(image, slot))
    predicted = _classify_batch(crops, hand=False)
    return _score_multiset(expected, predicted)


def _classify_batch(crops: list[Image.Image], *, hand: bool) -> list[str]:
    if not crops:
        return []
    preds = classify_tile_crops_onnx(crops, top_k=1)
    tiles: list[str] = []
    for crop, pred in zip(crops, preds, strict=True):
        tile = normalize_tile(pred.tile if pred else "") if pred else ""
        if hand and tile:
            red = detect_red_five(crop, tile)
            if red:
                tile = red
        tiles.append(_base_tile(tile) if tile else "")
    return tiles


def _discard_turn_map(raw: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    if not isinstance(raw, list):
        return result
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            turn = int(item.get("turn_index") or item.get("turn") or index)
            tile = _base_tile(item.get("tile"))
        else:
            turn = index
            tile = _base_tile(item)
        if tile:
            result[turn] = tile
    return result


def _score_positionally(expected: list[str], predicted: list[str]) -> Metrics:
    n = min(len(expected), len(predicted))
    return Metrics(
        correct=sum(1 for idx in range(n) if expected[idx] == predicted[idx]),
        expected=len(expected),
        predicted=sum(1 for tile in predicted if tile),
    )


def _score_multiset(expected: list[str], predicted: list[str]) -> Metrics:
    predicted_clean = [tile for tile in predicted if tile]
    overlap = Counter(expected) & Counter(predicted_clean)
    return Metrics(
        correct=sum(overlap.values()),
        expected=len(expected),
        predicted=len(predicted_clean),
    )


def _base_tile(value: Any) -> str:
    tile = normalize_tile(value)
    return RED_FIVE_TO_BASE.get(tile, tile)


def _accumulate(target: Metrics, item: Metrics) -> None:
    target.correct += item.correct
    target.expected += item.expected
    target.predicted += item.predicted


def _fmt(metrics: Metrics) -> str:
    return f"{metrics.correct}/{metrics.expected} F1={metrics.f1:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
