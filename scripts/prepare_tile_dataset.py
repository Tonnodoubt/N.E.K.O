"""Prepare combined training dataset for tile classifier fine-tuning.

Merges HuggingFace discard crops with local hand crops, applies augmentations,
and splits into train/val/test.

Usage:
    # 从 HuggingFace 下载 + 本地手牌 crop 合并
    python scripts/prepare_tile_dataset.py --hand-crops-dir data/hand_crops --output-dir data/tile_dataset

    # 只用本地手牌 crop（不下载 HuggingFace）
    python scripts/prepare_tile_dataset.py --hand-crops-dir data/hand_crops --output-dir data/tile_dataset --no-hf

    # 指定手牌数据占比
    python scripts/prepare_tile_dataset.py --hand-crops-dir data/hand_crops --output-dir data/tile_dataset --hand-ratio 0.4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TILE_CLASSES = [
    f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)
] + [f"{rank}z" for rank in range(1, 8)]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TILE_CLASSES)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(TILE_CLASSES)  # 34

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _iter_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def load_hf_dataset(cache_dir: Path | None = None) -> dict[str, list[Path]]:
    """Load pjura/mahjong_souls_tiles from HuggingFace Hub.

    Returns dict mapping tile label -> list of image paths.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("请安装 datasets: pip install datasets")
        return {}

    log.info("Loading pjura/mahjong_souls_tiles from HuggingFace...")
    ds = load_dataset("pjura/mahjong_souls_tiles", split="train", cache_dir=str(cache_dir) if cache_dir else None)

    output_map: dict[str, list[Path]] = {}
    temp_dir = cache_dir / "hf_tile_crops" if cache_dir else Path("tmp/hf_tile_crops")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(ds):
        img = row.get("image")
        label = str(row.get("label", "")).strip()
        if not img or not label or label not in LABEL_TO_INDEX:
            continue
        label_dir = temp_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        path = label_dir / f"hf_{idx:06d}.png"
        if not path.exists():
            img.save(path)
        output_map.setdefault(label, []).append(path)

    total = sum(len(v) for v in output_map.values())
    log.info(f"HuggingFace dataset: {total} images across {len(output_map)} classes")
    return output_map


def load_local_crops(crops_dir: Path) -> dict[str, list[Path]]:
    """Load local hand crops from directory structure: crops_dir/{label}/*.png"""
    result: dict[str, list[Path]] = {}
    if not crops_dir.exists():
        return result
    for label_dir in sorted(crops_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("_"):
            continue
        label = label_dir.name
        if label not in LABEL_TO_INDEX:
            log.warning(f"Skipping unknown label directory: {label}")
            continue
        images = _iter_images(label_dir)
        if images:
            result[label] = images
    total = sum(len(v) for v in result.values())
    log.info(f"Local crops: {total} images across {len(result)} classes")
    return result


def augment_hand_crop(img: Image.Image, seed: int) -> Image.Image:
    """Apply hand-tile-specific augmentations."""
    rng = random.Random(seed)
    result = img.copy()

    # Random brightness (simulates different screen brightness)
    if rng.random() < 0.5:
        factor = rng.uniform(0.75, 1.25)
        result = ImageEnhance.Brightness(result).enhance(factor)

    # Random contrast
    if rng.random() < 0.4:
        factor = rng.uniform(0.8, 1.2)
        result = ImageEnhance.Contrast(result).enhance(factor)

    # Random color jitter
    if rng.random() < 0.3:
        factor = rng.uniform(0.85, 1.15)
        result = ImageEnhance.Color(result).enhance(factor)

    # Slight Gaussian blur (simulates screenshot quality variation)
    if rng.random() < 0.2:
        result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    return result


def augment_discard_crop(img: Image.Image, seed: int) -> Image.Image:
    """Apply discard-tile-specific augmentations."""
    rng = random.Random(seed)
    result = img.copy()

    if rng.random() < 0.3:
        factor = rng.uniform(0.85, 1.15)
        result = ImageEnhance.Brightness(result).enhance(factor)

    if rng.random() < 0.2:
        factor = rng.uniform(0.9, 1.1)
        result = ImageEnhance.Contrast(result).enhance(factor)

    return result


def letterbox_pad(img: Image.Image, size: int = 224, fill: int = 0) -> Image.Image:
    """Resize with letterbox padding to preserve aspect ratio."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def build_dataset(
    hf_crops: dict[str, list[Path]],
    local_crops: dict[str, list[Path]],
    output_dir: Path,
    *,
    hand_augment_copies: int = 5,
    discard_augment_copies: int = 1,
    hand_ratio: float = 0.35,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the combined dataset with augmentations and train/val/test split."""
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    stats: dict[str, dict[str, int]] = {
        "hf_discard": {},
        "local_hand": {},
        "augmented_hand": {},
    }

    for label in TILE_CLASSES:
        hf_paths = hf_crops.get(label, [])
        local_paths = local_crops.get(label, [])

        # Collect all (path, source) pairs
        all_items: list[tuple[Path, str]] = []
        all_items.extend((p, "hf_discard") for p in hf_paths)
        all_items.extend((p, "local_hand") for p in local_paths)

        rng.shuffle(all_items)

        # Apply hand-crop augmentation to boost hand data
        augmented: list[tuple[Path, str]] = []
        for path, source in local_paths:
            for copy_idx in range(hand_augment_copies):
                aug_seed = rng.randint(0, 2**31)
                aug_path = output_dir / "_aug_cache" / f"{label}" / f"aug_{path.stem}_{copy_idx}.png"
                aug_path.parent.mkdir(parents=True, exist_ok=True)
                if not aug_path.exists():
                    with Image.open(path) as img:
                        aug = augment_hand_crop(img, aug_seed)
                        aug.save(aug_path)
                augmented.append((aug_path, "augmented_hand"))
                stats["augmented_hand"][label] = stats["augmented_hand"].get(label, 0) + 1

        # Apply discard augmentation
        for path in hf_paths:
            for copy_idx in range(discard_augment_copies):
                aug_seed = rng.randint(0, 2**31)
                aug_path = output_dir / "_aug_cache" / f"{label}" / f"hf_aug_{path.stem}_{copy_idx}.png"
                aug_path.parent.mkdir(parents=True, exist_ok=True)
                if not aug_path.exists():
                    with Image.open(path) as img:
                        aug = augment_discard_crop(img, aug_seed)
                        aug.save(aug_path)
                augmented.append((aug_path, "hf_discard"))

        all_items.extend(augmented)
        rng.shuffle(all_items)

        # Split
        n = len(all_items)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        for i, (path, source) in enumerate(all_items):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            splits[split].append((path, label))
            stats.setdefault(source, {}).setdefault(label, 0)
            stats[source][label] = stats[source].get(label, 0) + 1

    # Shuffle each split
    for split_name in splits:
        rng.shuffle(splits[split_name])

    # Write to disk
    for split_name, items in splits.items():
        split_dir = output_dir / split_name
        for label in TILE_CLASSES:
            (split_dir / label).mkdir(parents=True, exist_ok=True)
        for path, label in items:
            dest = split_dir / label / path.name
            if not dest.exists():
                shutil.copy2(path, dest)

    # Write dataset info
    info = {
        "num_classes": NUM_CLASSES,
        "classes": TILE_CLASSES,
        "splits": {name: len(items) for name, items in splits.items()},
        "stats": stats,
        "hand_ratio": hand_ratio,
        "seed": seed,
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Dataset written to {output_dir}")
    for name, items in splits.items():
        log.info(f"  {name}: {len(items)} images")

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tile classifier training dataset")
    parser.add_argument("--hand-crops-dir", default="data/hand_crops", help="Local hand crop directory")
    parser.add_argument("--output-dir", default="data/tile_dataset", help="Output dataset directory")
    parser.add_argument("--hf-cache-dir", default="tmp/hf_cache", help="HuggingFace cache directory")
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace dataset download")
    parser.add_argument("--hand-augment", type=int, default=5, help="Number of augmented copies per hand crop")
    parser.add_argument("--discard-augment", type=int, default=1, help="Augmented copies per discard crop")
    parser.add_argument("--hand-ratio", type=float, default=0.35, help="Target hand data ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_crops: dict[str, list[Path]] = {}
    if not args.no_hf:
        hf_crops = load_hf_dataset(cache_dir=Path(args.hf_cache_dir))

    local_crops = load_local_crops(Path(args.hand_crops_dir))

    if not hf_crops and not local_crops:
        log.error("No data sources available. Provide --hand-crops-dir or remove --no-hf.")
        sys.exit(1)

    info = build_dataset(
        hf_crops,
        local_crops,
        output_dir,
        hand_augment_copies=args.hand_augment,
        discard_augment_copies=args.discard_augment,
        hand_ratio=args.hand_ratio,
        seed=args.seed,
    )
    print(f"\nDataset prepared: {info['splits']}")


if __name__ == "__main__":
    main()
