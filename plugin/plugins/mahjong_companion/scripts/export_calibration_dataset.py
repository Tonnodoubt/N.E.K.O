"""Export tile crops from calibration-label screenshots for classifier training.

Reads ``.label.json`` sidecars (calibration v1 format) and extracts per-tile
crops using the annotated bounding boxes.  Produces the directory layout
expected by ``train_tile_classifier.py``.

Usage::

    uv run python -m plugin.plugins.mahjong_companion.scripts.export_calibration_dataset \\
        --label-root data/calibration/raw/manual/屏幕截图(33) \\
        --output-dir /tmp/tile_dataset

Output structure::

    output-dir/
      train/
        1m/  1p/  ...  empty/
      val/
        (same)
      labels.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

RED_FIVE_TILES = frozenset({"0m", "0p", "0s", "R5m", "R5p", "R5s"})
ALL_TILES = [
    *(f"{i}m" for i in range(1, 10)),
    *(f"{i}p" for i in range(1, 10)),
    *(f"{i}s" for i in range(1, 10)),
    *(f"{i}z" for i in range(1, 8)),
    "0m", "0p", "0s",
    "empty",
]
VALIDATION_SPLIT = 0.15
INNER_MARGIN = 0.05  # shrink bbox by 5% on each side to drop border pixels


def _normalize_tile(raw: str) -> str:
    """Map calibration-format tile strings to the classifier label set."""
    t = raw.strip()
    if t in ("R5m", "R5p", "R5s"):
        return {"R5m": "0m", "R5p": "0p", "R5s": "0s"}[t]
    return t


def _inner_bbox(box: dict) -> tuple[int, int, int, int]:
    """Apply inner margin to a bbox, returning (left, top, right, bottom)."""
    left = box["left"]
    top = box["top"]
    w = box["width"]
    h = box["height"]
    dx = int(w * INNER_MARGIN)
    dy_top = int(h * 0.06)  # top margin smaller — less border there
    dy_bot = int(h * 0.12)  # bottom margin larger — often has shadow
    return (left + dx, top + dy_top, left + w - dx, top + h - dy_bot)


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
    dest.mkdir(parents=True, exist_ok=True)
    dest_file = dest / f"{idx:05d}.png"
    crop.save(dest_file)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    random.seed(42)

    output = Path(args.output_dir)
    train_dir = output / "train"
    val_dir = output / "val"
    for cls in ALL_TILES:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

    label_root = Path(args.label_root)
    label_files = sorted(
        f for f in label_root.rglob("*.label.json") if not f.name.startswith(".")
    )

    stats: dict[str, int] = {}
    total = 0

    for label_file in label_files:
        data = json.loads(label_file.read_text())
        if data.get("scene") != "in_match":
            continue

        png_file = label_file.parent / f"{label_file.stem.replace('.label', '')}.png"
        if not png_file.exists():
            # stem might not have .label suffix
            png_file = label_file.with_suffix(".png")
        if not png_file.exists():
            continue

        try:
            img = Image.open(png_file).convert("RGB")
        except (OSError, Image.DecompressionBombError):
            continue

        w, h = img.size

        # --- Hand tile crops ---
        hand_slots = data.get("layout", {}).get("hand_slots", [])
        meld_slots = data.get("layout", {}).get("meld_slots", [])
        dora_slots = data.get("layout", {}).get("dora_slots", [])

        for slot in hand_slots + meld_slots + dora_slots:
            tile_raw = slot.get("tile", "")
            if not tile_raw:
                continue
            box = slot.get("box")
            if not box:
                continue
            try:
                l, t, r, b = _inner_bbox(box)
                if r <= l or b <= t:
                    continue
                crop = img.crop((l, t, r, b))
            except (OSError, ValueError):
                continue

            tile = _normalize_tile(tile_raw)
            _save_crop(crop, tile, train_dir, val_dir, stats)
            total += 1

        # --- Discard tile crops ---
        discard_piles = data.get("discard_piles", {})
        for _player, tiles in discard_piles.items():
            for entry in tiles:
                bbox = entry.get("bbox")
                tile_raw = entry.get("tile", "")
                if not bbox or not tile_raw:
                    continue
                try:
                    l, t, r, b = bbox
                    # apply smaller inner margin for discard (tiles are smaller)
                    dx = int((r - l) * INNER_MARGIN)
                    dy = int((b - t) * INNER_MARGIN)
                    l2, t2, r2, b2 = l + dx, t + dy, r - dx, b - dy
                    if r2 <= l2 or b2 <= t2:
                        continue
                    crop = img.crop((l2, t2, r2, b2))
                except (OSError, ValueError):
                    continue

                tile = _normalize_tile(tile_raw)
                _save_crop(crop, tile, train_dir, val_dir, stats)
                total += 1

        # --- Negative (empty) samples from table felt ---
        n_empty = min(args.empty_per_frame, 15)
        for _ in range(n_empty):
            felt_x = random.randint(int(w * 0.1), int(w * 0.9) - 32)
            felt_y = random.randint(int(h * 0.4), int(h * 0.65) - 32)
            cw = random.randint(24, 56)
            ch = random.randint(32, 72)
            felt_x = max(0, min(w - cw, felt_x))
            felt_y = max(0, min(h - ch, felt_y))
            empty_crop = img.crop((felt_x, felt_y, felt_x + cw, felt_y + ch))
            _save_crop(empty_crop, "empty", train_dir, val_dir, stats)
            total += 1

    # Report
    print(f"Exported {total} crops from {len(label_files)} label files")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")
    print(f"Classes:")
    for cls in ALL_TILES:
        n = stats.get(cls, 0)
        if n > 0:
            bar = "#" * max(1, n // 3)
            print(f"  {cls:>7s}: {n:4d} {bar}")
    print(f"  {'TOTAL':>7s}: {total:4d}")

    # Write class labels
    (output / "labels.txt").write_text("\n".join(ALL_TILES))
    print(f"\nLabels: {output / 'labels.txt'}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tile crops from calibration-label screenshots",
    )
    parser.add_argument(
        "--label-root",
        default="data/calibration/raw/manual/屏幕截图(33)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/tile_dataset",
    )
    parser.add_argument(
        "--empty-per-frame",
        type=int,
        default=15,
        help="Negative (empty) samples generated per frame (default: 15)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
