"""Extract hand tile crops from Mahjong Soul screenshots for training data.

Usage:
    # 从截图目录提取手牌 crop，用 template matcher 自动标注
    python scripts/extract_hand_crops.py --input-dir path/to/screenshots --output-dir data/hand_crops

    # 指定分辨率（默认自动检测）
    python scripts/extract_hand_crops.py --input-dir path/to/screenshots --output-dir data/hand_crops --width 1920 --height 1080

    # 只提取，不分类（生成 raw crops + 棋盘格预览，方便手动标注）
    python scripts/extract_hand_crops.py --input-dir path/to/screenshots --output-dir data/hand_crops --no-classify

Output structure:
    data/hand_crops/
        1m/  2m/  ...  7z/  unclassified/
            0001.png  0002.png  ...
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from PIL import Image

def _load_module(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

# Bootstrap perception modules without importing the full plugin package
_PERCEPTION = Path(__file__).resolve().parent.parent / "plugin" / "plugins" / "mahjong_coach" / "perception"
_MAHJONG = Path(__file__).resolve().parent.parent / "plugin" / "plugins" / "mahjong_coach"

_load_module("plugin.plugins.mahjong_coach.tile_labels", _MAHJONG / "tile_labels.py")
_load_module("plugin.plugins.mahjong_coach.perception.roi", _PERCEPTION / "roi.py")
_load_module("plugin.plugins.mahjong_coach.perception.calibration", _PERCEPTION / "calibration.py")
_load_module("plugin.plugins.mahjong_coach.perception.hand_layout", _PERCEPTION / "hand_layout.py")
_load_module("plugin.plugins.mahjong_coach.perception.tile_templates", _PERCEPTION / "tile_templates.py")
_load_module("plugin.plugins.mahjong_coach.perception.vit_tile_classifier_onnx", _PERCEPTION / "vit_tile_classifier_onnx.py")
_load_module("plugin.plugins.mahjong_coach.perception.tile_classifier_dispatch", _PERCEPTION / "tile_classifier_dispatch.py")

from plugin.plugins.mahjong_coach.perception.calibration import resolve_calibration_profile
from plugin.plugins.mahjong_coach.perception.hand_layout import build_hand_layout
from plugin.plugins.mahjong_coach.perception.roi import collect_region_metrics
from plugin.plugins.mahjong_coach.perception.tile_classifier_dispatch import classify_hand_tile
from plugin.plugins.mahjong_coach.perception.tile_templates import is_probably_occupied_hand_slot

_CALIBRATION_DIR = (
    Path(__file__).resolve().parent.parent
    / "plugin" / "plugins" / "mahjong_coach" / "data" / "calibration" / "profiles"
)

ALL_TILE_LABELS = [
    f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)
] + [f"{rank}z" for rank in range(1, 8)]

MIN_HAND_TILES = 10  # lower threshold for extraction (vs 12 for live recognition)


def extract_crops_from_image(
    image_path: Path,
    *,
    output_dir: Path,
    calibration_dir: Path,
    classify: bool = True,
    min_confidence: float = 0.10,
) -> dict[str, int]:
    """Extract hand tile crops from a single screenshot. Returns per-label counts."""
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
    except Exception:
        return {}

    w, h = image.size
    calibration = resolve_calibration_profile(w, h, calibration_dir=calibration_dir)
    template_payload = calibration.hand_tile_templates

    layout = build_hand_layout(w, h, calibration=calibration)
    counts: dict[str, int] = {}

    for slot in layout["hand"][:14]:
        metrics = collect_region_metrics(image, slot.box, sample_step=6)
        occupied = is_probably_occupied_hand_slot({
            "slot_mean_luma": metrics.get("mean_luma"),
            "slot_bright_ratio": metrics.get("bright_ratio"),
            "slot_dark_ratio": metrics.get("dark_ratio"),
            "slot_stddev": metrics.get("stddev"),
        })
        if not occupied:
            continue

        crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))

        if classify and template_payload:
            match = classify_hand_tile(crop, template_payload)
            if match and match.confidence >= min_confidence:
                label = match.tile
            else:
                label = "unclassified"
        else:
            label = "unclassified"

        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(label_dir.glob("*.png")))
        stem = f"{image_path.stem}_{slot.slot_id}"
        crop.save(label_dir / f"{stem}.png")
        counts[label] = counts.get(label, 0) + 1

    return counts


def build_contact_sheet(output_dir: Path, cols: int = 20) -> None:
    """Build a contact sheet per label for quick visual review."""
    for label_dir in sorted(output_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue
        images = sorted(label_dir.glob("*.png"))
        if not images:
            continue
        first = Image.open(images[0])
        tw, th = first.size
        rows = (len(images) + cols - 1) // cols
        sheet = Image.new("RGB", (tw * cols, th * rows), (40, 40, 40))
        for idx, img_path in enumerate(images):
            with Image.open(img_path) as tile_img:
                r, c = divmod(idx, cols)
                sheet.paste(tile_img, (c * tw, r * th))
        sheet.save(label_dir / "_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract hand tile crops from Mahjong Soul screenshots")
    parser.add_argument("--input-dir", required=True, help="Directory of Mahjong Soul screenshots")
    parser.add_argument("--output-dir", default="data/hand_crops", help="Output directory for crops")
    parser.add_argument("--calibration-dir", default=str(_CALIBRATION_DIR), help="Calibration profiles directory")
    parser.add_argument("--no-classify", action="store_true", help="Skip auto-classification, save as unclassified")
    parser.add_argument("--contact-sheet", action="store_true", help="Generate contact sheets for visual review")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    calibration_dir = Path(args.calibration_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    classify = not args.no_classify

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in extensions)
    if not images:
        print("No images found in input directory.")
        sys.exit(1)

    total_counts: dict[str, int] = {}
    for i, image_path in enumerate(images, 1):
        counts = extract_crops_from_image(
            image_path,
            output_dir=output_dir,
            calibration_dir=calibration_dir,
            classify=classify,
        )
        for label, count in counts.items():
            total_counts[label] = total_counts.get(label, 0) + count
        extracted = sum(counts.values())
        print(f"[{i}/{len(images)}] {image_path.name}: {extracted} crops extracted")

    if args.contact_sheet:
        print("Building contact sheets...")
        build_contact_sheet(output_dir)

    print(f"\nTotal crops extracted:")
    for label in sorted(total_counts):
        print(f"  {label}: {total_counts[label]}")
    print(f"  TOTAL: {sum(total_counts.values())}")
    if total_counts.get("unclassified"):
        print(f"\n  {total_counts['unclassified']} crops need manual review in output_dir/unclassified/")


if __name__ == "__main__":
    main()
