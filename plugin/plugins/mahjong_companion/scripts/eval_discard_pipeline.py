"""Evaluate ONNX discard accuracy with the real pipeline.

Uses parse_discards_from_image which includes both the geometric occupancy
detector and the ONNX softmax confidence gate. The default thresholds are
intended as a lightweight release gate for the discard recognizer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.discard_parser import parse_discards_from_image


FIXTURES_ROOT = _REPO_ROOT / "plugin" / "plugins" / "mahjong_companion" / "tests" / "fixtures" / "multi_theme"
DEFAULT_MIN_PRECISION = 0.90
DEFAULT_MIN_RECALL = 0.95
DEFAULT_MIN_F1 = 0.94


@dataclass
class Score:
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

    def add(self, *, correct: int, expected: int, predicted: int) -> None:
        self.correct += correct
        self.expected += expected
        self.predicted += predicted

    def merge(self, other: "Score") -> None:
        self.add(correct=other.correct, expected=other.expected, predicted=other.predicted)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = []
    fixtures_root = Path(args.fixtures_root)
    for theme_dir in sorted(fixtures_root.iterdir()):
        if not theme_dir.is_dir() or theme_dir.name.startswith("_"):
            continue
        for png in sorted(theme_dir.glob("*.png")):
            sidecar = png.with_suffix(".tiles.json")
            if not sidecar.exists():
                continue
            cases.append((theme_dir.name, png, sidecar))

    print(f"Evaluating {len(cases)} screenshots\n")

    total = Score()
    per_theme: dict[str, Score] = {}

    for theme_id, png_path, sidecar in cases:
        gt = json.loads(sidecar.read_text())
        img = Image.open(png_path).convert("RGB")

        result = parse_discards_from_image(img, {"source": "eval"})
        predicted_piles = result.discard_piles

        gt_discards = gt.get("discard_piles", {})
        case_correct, case_expected, case_predicted = 0, 0, 0

        for player in ("self", "left_opponent", "top_opponent", "right_opponent"):
            gt_pile = gt_discards.get(player, [])
            gt_tiles = [
                str(t.get("tile", "") if isinstance(t, dict) else t).strip()
                for t in gt_pile
            ]
            pred_pile = predicted_piles.get(player, [])
            pred_tiles = [str(t.get("tile", "")).strip() for t in pred_pile]

            case_expected += len(gt_tiles)
            case_predicted += len(pred_tiles)

            from collections import Counter
            gt_c = Counter(gt_tiles)
            pred_c = Counter(pred_tiles)
            overlap = sum((gt_c & pred_c).values())
            case_correct += overlap

        case_score = Score(case_correct, case_expected, case_predicted)
        total.merge(case_score)
        per_theme.setdefault(theme_id, Score()).merge(case_score)

        print(
            f"  {theme_id}/{png_path.stem}: {case_correct}/{case_expected}  "
            f"P={case_score.precision:.2f} R={case_score.recall:.2f} F1={case_score.f1:.2f}"
        )

    print("\n" + "=" * 60)
    print("DISCARD SUMMARY")
    print("=" * 60)
    for theme_id, b in sorted(per_theme.items()):
        print(f"  {theme_id:<24} {b.correct:>3}/{b.expected:<3}  P={b.precision:.2f} R={b.recall:.2f} F1={b.f1:.2f}")

    print(f"  {'OVERALL':<24} {total.correct:>3}/{total.expected:<3}  P={total.precision:.2f} R={total.recall:.2f} F1={total.f1:.2f}")

    failures = _threshold_failures(
        total,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
    )
    if failures and not args.no_gate:
        print("\nGATE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    if not args.no_gate:
        print("\nGATE PASSED")
    return 0


def _threshold_failures(
    score: Score,
    *,
    min_precision: float,
    min_recall: float,
    min_f1: float,
) -> list[str]:
    failures: list[str] = []
    if score.precision < min_precision:
        failures.append(f"precision {score.precision:.3f} < {min_precision:.3f}")
    if score.recall < min_recall:
        failures.append(f"recall {score.recall:.3f} < {min_recall:.3f}")
    if score.f1 < min_f1:
        failures.append(f"F1 {score.f1:.3f} < {min_f1:.3f}")
    return failures


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ONNX discard pipeline accuracy")
    parser.add_argument("--fixtures-root", default=str(FIXTURES_ROOT))
    parser.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION)
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1)
    parser.add_argument("--no-gate", action="store_true", help="Print metrics but do not fail on thresholds")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
