"""Export tile crops from labelled fixtures for classifier training.

Produces a directory of per-class crops plus an ``empty/`` class of
non-tile samples, ready for training a lightweight tile classifier.

Usage::

    uv run python -m plugin.plugins.mahjong_companion.scripts.export_tile_classifier_dataset \\
        --label-root plugin/plugins/mahjong_companion/tests/fixtures/multi_theme \\
        --output-dir /tmp/tile_dataset

Output structure::

    output-dir/
      train/
        0m/  0p/  0s/          # red-five (auto-detected from 5m/5p/5s)
        1m/  1p/  1s/  ... 9s/
        1z/  2z/  ... 7z/
        empty/                  # negative samples (non-tile regions)
      val/
        (same structure)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..perception.discard_layout import build_discard_layout
from ..perception.discard_parser import normalize_discard_crop
from ..perception.hand_baseline import detect_hand_baseline
from ..perception.hand_layout import build_hand_layout
from ..perception.roi import collect_region_metrics
from ..perception.tile_classifier_dispatch import detect_red_five

DEFAULT_OUTPUT_DIR = Path("/tmp/tile_dataset")
RED_FIVE_TILES = frozenset({"0m", "0p", "0s"})
ALL_TILES = [
    *(f"{i}m" for i in range(1, 10)),
    *(f"{i}p" for i in range(1, 10)),
    *(f"{i}s" for i in range(1, 10)),
    *(f"{i}z" for i in range(1, 8)),
    "0m", "0p", "0s",
    "empty",
]
VALIDATION_SPLIT = 0.15
EMPTY_SAMPLES_PER_FRAME = 20


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    random.seed(42)
    np.random.seed(42)

    output = Path(args.output_dir)
    train_dir = output / "train"
    val_dir = output / "val"
    for cls in ALL_TILES:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

    label_root = Path(args.label_root)
    png_files = list(label_root.rglob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"no .png files under {label_root}")

    stats = {cls: 0 for cls in ALL_TILES}
    total = 0

    for png in png_files:
        gt_file = png.with_suffix(".tiles.json")
        gt = _load_gt(gt_file)
        try:
            img = Image.open(png).convert("RGB")
        except Exception:
            continue

        w, h = img.size
        baseline = detect_hand_baseline(img)
        hand_layout = build_hand_layout(w, h, baseline=baseline)
        discard_layout = build_discard_layout(w, h, baseline=baseline)

        # --- Hand tile crops ---
        gt_hand = gt.get("hand_tiles", [])
        for slot_idx, slot in enumerate(hand_layout.get("hand", [])[:14]):
            box = slot.box
            if box.width <= 0 or box.height <= 0:
                continue
            crop = img.crop((box.left, box.top, box.right, box.bottom))
            if slot_idx < len(gt_hand):
                tile = gt_hand[slot_idx]
                # Check for red five via colour detection
                red5 = detect_red_five(crop, tile)
                if red5:
                    tile = red5
                _save_crop(crop, tile, train_dir, val_dir, stats)
                total += 1

        # --- Discard tile crops ---
        for player, slots in discard_layout.items():
            gt_pile = gt.get("discard_piles", {}).get(player, [])
            gt_map = {d["turn_index"]: d["tile"] for d in gt_pile}
            for slot in slots:
                if slot.turn_index not in gt_map:
                    continue
                box = slot.box
                if box.width <= 0 or box.height <= 0:
                    continue
                crop = img.crop((box.left, box.top, box.right, box.bottom))
                normalized = normalize_discard_crop(crop, slot.orientation)
                tile = gt_map[slot.turn_index]
                _save_crop(normalized, tile, train_dir, val_dir, stats)
                total += 1

        # --- Empty (negative) samples ---
        hand_slots = hand_layout.get("hand", [])[:14]
        if hand_slots:
            # Sample empty regions between hand slots
            for _ in range(min(EMPTY_SAMPLES_PER_FRAME, 10)):
                # Random position in hand area but NOT on a tile
                first = hand_slots[0].box
                last = hand_slots[-1].box
                x0 = first.left
                x1 = last.right
                y0 = first.top - first.height // 2
                y1 = first.bottom + first.height // 2
                # Random small crop
                cw = random.randint(20, 50)
                ch = random.randint(30, 60)
                cx = random.randint(x0, max(x0 + 1, x1 - cw))
                cy = random.randint(y0, max(y0 + 1, y1 - ch))
                cx = max(0, min(w - cw, cx))
                cy = max(0, min(h - ch, cy))
                empty_crop = img.crop((cx, cy, cx + cw, cy + ch))
                # Verify it doesn't overlap a hand slot
                overlaps = False
                for slot in hand_slots:
                    sb = slot.box
                    if (cx < sb.right and cx + cw > sb.left and
                            cy < sb.bottom and cy + ch > sb.top):
                        overlaps = True
                        break
                if not overlaps:
                    _save_crop(empty_crop, "empty", train_dir, val_dir, stats)
                    total += 1

        # Empty samples from table felt
        for _ in range(min(EMPTY_SAMPLES_PER_FRAME, 10)):
            felt_x = random.randint(100, w - 200)
            felt_y = random.randint(int(h * 0.4), int(h * 0.65))
            cw = random.randint(24, 56)
            ch = random.randint(32, 72)
            felt_x = max(0, min(w - cw, felt_x))
            felt_y = max(0, min(h - ch, felt_y))
            felt_crop = img.crop((felt_x, felt_y, felt_x + cw, felt_y + ch))
            _save_crop(felt_crop, "empty", train_dir, val_dir, stats)
            total += 1

    # Report
    print(f"Exported {total} crops to {output}")
    for cls in ALL_TILES:
        n = stats.get(cls, 0)
        if n > 0:
            print(f"  {cls:8s}: {n:5d}")
    print(f"  {'TOTAL':8s}: {total:5d}")

    # Write class labels
    (output / "labels.txt").write_text("\n".join(ALL_TILES))
    return 0


def _save_crop(
    crop: Image.Image,
    tile: str,
    train_dir: Path,
    val_dir: Path,
    stats: dict[str, int],
) -> None:
    is_val = random.random() < VALIDATION_SPLIT
    dest = (val_dir if is_val else train_dir) / tile
    idx = stats.get(tile, 0)
    stats[tile] = idx + 1
    dest_file = dest / f"{idx:05d}.png"
    crop.save(dest_file)


def _load_gt(gt_file: Path) -> dict[str, Any]:
    if gt_file.exists():
        return json.loads(gt_file.read_text())
    return {}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tile classifier training dataset")
    parser.add_argument("--label-root", default=str(Path("plugin/plugins/mahjong_companion/tests/fixtures/multi_theme")))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
