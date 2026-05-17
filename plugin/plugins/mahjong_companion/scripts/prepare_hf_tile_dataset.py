"""Prepare a lightweight classifier dataset from Mahjong Soul HF crops.

The public ``pjura/mahjong_souls_tiles`` dataset already contains cropped
Mahjong Soul tiles, but its labels use a HuggingFace-friendly naming scheme
(``1n``, ``1b``, ``ew``...) while the plugin runtime expects canonical tile
codes (``1m``, ``1s``, ``1z``...). This script copies the dataset into the
layout consumed by ``train_tile_classifier.py`` and writes ``labels.txt``.

Usage::

    uv run python -m plugin.plugins.mahjong_companion.scripts.prepare_hf_tile_dataset \
        --output-dir /tmp/tile_dataset \
        --empty-source-dir plugin/plugins/mahjong_companion/data/debug_samples/live

If ``--source-dir`` is not provided the script tries to download the dataset
with ``huggingface_hub``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote
from urllib.request import urlopen

from PIL import Image


DEFAULT_DATASET = "pjura/mahjong_souls_tiles"
DEFAULT_OUTPUT_DIR = Path("/tmp/tile_dataset")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

HF_LABEL_MAP = {
    **{f"{index}n": f"{index}m" for index in range(1, 10)},
    **{f"{index}m": f"{index}m" for index in range(1, 10)},
    **{f"{index}p": f"{index}p" for index in range(1, 10)},
    **{f"{index}b": f"{index}s" for index in range(1, 10)},
    **{f"{index}s": f"{index}s" for index in range(1, 10)},
    "ew": "1z",
    "east": "1z",
    "sw": "2z",
    "south": "2z",
    "ww": "3z",
    "west": "3z",
    "nw": "4z",
    "north": "4z",
    "wd": "5z",
    "white": "5z",
    "gd": "6z",
    "green": "6z",
    "rd": "7z",
    "red": "7z",
}

CANONICAL_TILES = [
    *(f"{i}m" for i in range(1, 10)),
    *(f"{i}p" for i in range(1, 10)),
    *(f"{i}s" for i in range(1, 10)),
    *(f"{i}z" for i in range(1, 8)),
]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    random.seed(args.seed)

    source_dir = _resolve_source_dir(args)
    dataset_root = _find_dataset_root(source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        args.train_split: "train",
        args.val_split: "val",
    }
    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    copied = 0

    for source_split, dest_split in split_map.items():
        split_dir = dataset_root / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split not found: {split_dir}")
        copied += _copy_split(split_dir, output_dir / dest_split, stats[dest_split])

    empty_count = 0
    if args.empty_source_dir:
        empty_count = _add_empty_samples(
            Path(args.empty_source_dir),
            output_dir=output_dir,
            per_frame=args.empty_per_frame,
            stats=stats,
        )

    labels = _labels_for_output(stats)
    (output_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")

    print(f"Source: {dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Copied tile crops: {copied}")
    if empty_count:
        print(f"Added empty crops: {empty_count}")
    print(f"Classes: {len(labels)}")
    for split in ("train", "val"):
        total = sum(stats[split].values())
        print(f"  {split}: {total}")
        for label in labels:
            count = stats[split].get(label, 0)
            if count:
                print(f"    {label:>5s}: {count}")
    return 0


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    train_split: str = "train",
    val_split: str = "test",
    overwrite: bool = False,
    seed: int = 42,
) -> list[str]:
    """Programmatic wrapper used by tests and small local tools."""
    argv = [
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--train-split",
        train_split,
        "--val-split",
        val_split,
        "--seed",
        str(seed),
    ]
    if overwrite:
        argv.append("--overwrite")
    main(argv)
    return [
        line.strip()
        for line in (output_dir / "labels.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_source_dir(args: argparse.Namespace) -> Path:
    if args.source_dir:
        return Path(args.source_dir).expanduser().resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(f"huggingface_hub unavailable ({exc}); falling back to concurrent HTTP download.")
        return _download_dataset_without_hub(
            args.dataset,
            revision=args.revision,
            workers=args.download_workers,
        )
    downloaded = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        revision=args.revision,
    )
    return Path(downloaded).resolve()


def _download_dataset_without_hub(dataset: str, *, revision: str | None, workers: int = 16) -> Path:
    clean_dataset = str(dataset or DEFAULT_DATASET).strip() or DEFAULT_DATASET
    clean_revision = str(revision or "main").strip() or "main"
    cache_dir = (
        Path(tempfile.gettempdir())
        / "mahjong_companion_hf"
        / clean_dataset.replace("/", "__")
        / clean_revision.replace("/", "__")
    )
    entries = [
        *_fetch_hf_tree(clean_dataset, clean_revision, "dataset/train"),
        *_fetch_hf_tree(clean_dataset, clean_revision, "dataset/test"),
    ]

    image_entries = [
        item
        for item in entries
        if isinstance(item, dict)
        and item.get("type") == "file"
        and _is_dataset_image_path(str(item.get("path", "")))
    ]
    if not image_entries:
        raise RuntimeError(f"no dataset images found for {clean_dataset}@{clean_revision}")

    jobs = [
        (str(item["path"]), int(item.get("size") or 0))
        for item in image_entries
    ]
    completed = 0
    worker_count = max(1, int(workers or 16))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _download_one_hf_file,
                clean_dataset,
                clean_revision,
                rel_path,
                expected_size,
                cache_dir,
            )
            for rel_path, expected_size in jobs
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(futures):
                print(f"Downloaded/verified {completed}/{len(futures)} files")
    return cache_dir.resolve()


def _fetch_hf_tree(dataset: str, revision: str, subpath: str) -> list[dict]:
    encoded_revision = quote(revision, safe="")
    encoded_subpath = quote(subpath.strip("/"), safe="")
    url = (
        f"https://huggingface.co/api/datasets/{dataset}/tree/"
        f"{encoded_revision}/{encoded_subpath}?expand=false&recursive=true&limit=1000"
    )
    entries: list[dict] = []
    while url:
        with urlopen(url, timeout=60) as response:
            page = json.loads(response.read().decode("utf-8"))
            next_url = _next_link(response.headers.get("Link", ""))
        if not isinstance(page, list):
            raise RuntimeError(f"unexpected HuggingFace tree response for {dataset}/{subpath}")
        entries.extend(item for item in page if isinstance(item, dict))
        url = next_url
    return entries


def _next_link(header: str) -> str | None:
    for part in str(header or "").split(","):
        if 'rel="next"' not in part:
            continue
        match = re.search(r"<([^>]+)>", part)
        if match:
            return match.group(1)
    return None


def _download_one_hf_file(
    dataset: str,
    revision: str,
    rel_path: str,
    expected_size: int,
    cache_dir: Path,
) -> None:
    dest = cache_dir / rel_path
    if dest.exists() and (expected_size <= 0 or dest.stat().st_size == expected_size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = quote(rel_path, safe="/")
    url = f"https://huggingface.co/datasets/{dataset}/resolve/{quote(revision, safe='')}/{encoded_path}"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        tmp_dest = dest.with_suffix(dest.suffix + f".part{attempt}")
        if tmp_dest.exists():
            tmp_dest.unlink()
        try:
            with urlopen(url, timeout=90) as response, tmp_dest.open("wb") as out:
                shutil.copyfileobj(response, out)
            actual_size = tmp_dest.stat().st_size
            if expected_size > 0 and actual_size != expected_size:
                raise RuntimeError(
                    f"downloaded size mismatch for {rel_path}: "
                    f"{actual_size} != {expected_size}"
                )
            tmp_dest.replace(dest)
            return
        except Exception as exc:
            last_error = exc
            tmp_dest.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(0.5 * attempt)
    assert last_error is not None
    raise last_error


def _find_dataset_root(source_dir: Path) -> Path:
    candidates = [source_dir, source_dir / "dataset"]
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not find an imagefolder dataset root under {source_dir}; "
        "expected train/ or dataset/train/"
    )


def _is_dataset_image_path(path: str) -> bool:
    clean = path.replace("\\", "/")
    return (
        (clean.startswith("dataset/train/") or clean.startswith("dataset/test/"))
        and Path(clean).suffix.lower() in IMAGE_EXTENSIONS
    )


def _copy_split(split_dir: Path, dest_split_dir: Path, stats: dict[str, int]) -> int:
    copied = 0
    for label_dir in sorted(item for item in split_dir.iterdir() if item.is_dir()):
        canonical = normalize_hf_label(label_dir.name)
        if not canonical:
            print(f"Skipping unknown label: {label_dir.name}")
            continue
        dest_class_dir = dest_split_dir / canonical
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        start_index = stats.get(canonical, 0)
        for index, image_path in enumerate(_iter_images(label_dir), start=start_index):
            dest = dest_class_dir / f"{index:05d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, dest)
            stats[canonical] = stats.get(canonical, 0) + 1
            copied += 1
    return copied


def _add_empty_samples(
    source_dir: Path,
    *,
    output_dir: Path,
    per_frame: int,
    stats: dict[str, dict[str, int]],
) -> int:
    if per_frame <= 0:
        return 0
    frames = list(_iter_images(source_dir))
    total = 0
    for frame in frames:
        try:
            image = Image.open(frame).convert("RGB")
        except (OSError, ValueError):
            continue
        split = "val" if random.random() < 0.15 else "train"
        for _ in range(per_frame):
            crop = _random_table_crop(image)
            dest_dir = output_dir / split / "empty"
            dest_dir.mkdir(parents=True, exist_ok=True)
            index = stats[split].get("empty", 0)
            stats[split]["empty"] = index + 1
            crop.save(dest_dir / f"{index:05d}.png")
            total += 1
    return total


def _random_table_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_width = random.randint(24, 56)
    crop_height = random.randint(32, 72)
    left_min = int(width * 0.12)
    left_max = max(left_min, int(width * 0.88) - crop_width)
    top_min = int(height * 0.38)
    top_max = max(top_min, int(height * 0.66) - crop_height)
    left = random.randint(left_min, left_max)
    top = random.randint(top_min, top_max)
    left = max(0, min(width - crop_width, left))
    top = max(0, min(height - crop_height, top))
    return image.crop((left, top, left + crop_width, top + crop_height))


def _labels_for_output(stats: dict[str, dict[str, int]]) -> list[str]:
    labels = [
        label
        for label in CANONICAL_TILES
        if stats["train"].get(label, 0) or stats["val"].get(label, 0)
    ]
    if stats["train"].get("empty", 0) or stats["val"].get("empty", 0):
        labels.append("empty")
    return labels


def normalize_hf_label(label: str) -> str:
    return HF_LABEL_MAP.get(str(label or "").strip().lower(), "")


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Mahjong Soul HF tile dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--empty-source-dir", default=None)
    parser.add_argument("--empty-per-frame", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
