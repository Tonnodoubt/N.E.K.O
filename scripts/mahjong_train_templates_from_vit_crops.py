from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from plugin.plugins.mahjong_companion.perception.vit_template_training import (
    DEFAULT_MAX_SAMPLES_PER_TILE,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    discover_crop_roots,
    train_profile_discard_templates_from_vit_frames,
    train_profile_templates_from_vit_crops,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Mahjong Companion templates from ViT-labeled crop images.")
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Crop directories or parent directories containing *_crops folders.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path("plugin/plugins/mahjong_companion/data/calibration/profiles/majsoul-pc-manual-2026.05-1920x1080.json"),
    )
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=Path(".tmp/mahjong_template_training/profiles/majsoul-pc-manual-2026.05-1920x1080-vit-discard.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(".tmp/mahjong_template_training/vit-template-training-report.json"),
    )
    parser.add_argument("--target", choices=["discard", "hand"], default="discard")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument("--max-samples-per-tile", type=int, default=DEFAULT_MAX_SAMPLES_PER_TILE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="krmin/mahjong_soul_vision")
    parser.add_argument("--device", default="-1")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--runtime-frames",
        action="store_true",
        help="Treat roots as full-frame screenshots and train from the runtime discard cropper.",
    )
    parser.add_argument(
        "--frame-resolution",
        default="",
        help="Only use runtime frames with this resolution, for example 2560x1440.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limit runtime-frame training to the first N frames.")
    args = parser.parse_args()

    roots = args.roots or [Path(".tmp/mahjong_vision_probe")]
    classifier_config = {
        "model": args.model,
        "device": args.device,
        "top_k": args.top_k,
    }
    if args.runtime_frames:
        frame_paths = _collect_frame_paths(roots)
        if args.frame_resolution:
            frame_paths = _filter_frame_paths_by_resolution(frame_paths, args.frame_resolution)
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]
        if args.target != "discard":
            raise SystemExit("--runtime-frames currently supports --target discard only")
        report = train_profile_discard_templates_from_vit_frames(
            frame_paths,
            base_profile_path=args.base_profile,
            output_profile_path=args.output_profile,
            output_report_path=args.output_report,
            classifier_config=classifier_config,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            max_samples_per_tile=args.max_samples_per_tile,
            batch_size=args.batch_size,
        )
        print("frame_paths", len(frame_paths))
    else:
        crop_roots: list[Path] = []
        for root in roots:
            crop_roots.extend(discover_crop_roots(root))
        crop_roots = sorted(set(crop_roots))
        report = train_profile_templates_from_vit_crops(
            crop_roots,
            base_profile_path=args.base_profile,
            output_profile_path=args.output_profile,
            output_report_path=args.output_report,
            target=args.target,
            classifier_config=classifier_config,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            max_samples_per_tile=args.max_samples_per_tile,
            batch_size=args.batch_size,
        )
        print("crop_roots", len(crop_roots))
    print("total_crops", report.total_crops)
    print("accepted_crops", report.accepted_crops)
    print("rejected_crops", report.rejected_crops)
    print("accepted_by_tile", report.accepted_by_tile)
    print("output_profile_path", report.output_profile_path)
    print("output_report_path", report.output_report_path)
    return 0


def _collect_frame_paths(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.is_file() and _is_source_frame(root):
            paths.append(root)
        elif root.is_dir():
            for suffix in ("*.png", "*.jpg", "*.jpeg"):
                paths.extend(sorted(path for path in root.rglob(suffix) if _is_source_frame(path)))
    return sorted(set(paths))


def _is_source_frame(path: Path) -> bool:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return False
    stem = path.stem.lower()
    return not (
        stem.endswith("-overlay")
        or "-overlay-" in stem
        or stem.endswith("-vit-overlay")
        or "-vit-overlay-" in stem
    )


def _filter_frame_paths_by_resolution(paths: list[Path], resolution: str) -> list[Path]:
    width, height = _parse_frame_resolution(resolution)
    matched: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                if image.size == (width, height):
                    matched.append(path)
        except OSError:
            continue
    return matched


def _parse_frame_resolution(value: str) -> tuple[int, int]:
    normalized = str(value or "").strip().lower().replace("*", "x")
    parts = normalized.split("x", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("--frame-resolution must look like 2560x1440")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise ValueError("--frame-resolution must look like 2560x1440") from exc
    if width <= 0 or height <= 0:
        raise ValueError("--frame-resolution must use positive width and height")
    return width, height


if __name__ == "__main__":
    raise SystemExit(main())
