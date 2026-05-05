from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.discard_parser import (
    _collect_discard_slot_metrics,
    crop_discard_slot,
    is_probably_occupied_discard_slot,
)
from plugin.plugins.mahjong_companion.perception.discard_quad_finder import refine_discard_slot_quad
from plugin.plugins.mahjong_companion.perception.vit_tile_classifier import (
    DEFAULT_VIT_MODEL,
    classify_tile_crops,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ViT-labeled Mahjong Soul discard crops grouped by tile.")
    parser.add_argument("roots", nargs="+", type=Path, help="Full-frame screenshot files or directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-resolution", default="", help="Only use frames matching WIDTHxHEIGHT.")
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--min-margin", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_VIT_MODEL)
    parser.add_argument("--device", default="-1")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    frame_paths = _collect_frame_paths(args.roots)
    if args.frame_resolution:
        frame_paths = _filter_frame_paths_by_resolution(frame_paths, args.frame_resolution)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending: list[dict[str, Any]] = []
    crops: list[Image.Image] = []
    frame_sizes: set[tuple[int, int]] = set()
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as opened:
                image = opened.convert("RGB")
        except OSError:
            continue
        frame_sizes.add(image.size)
        layout = build_discard_layout(*image.size)
        for slots in layout.values():
            for slot in slots:
                metrics = _collect_discard_slot_metrics(image, slot)
                refinement = refine_discard_slot_quad(image, slot)
                occupied = is_probably_occupied_discard_slot(metrics) or refinement is not None
                if not occupied:
                    continue
                crop = crop_discard_slot(image, slot, refine=True)
                pending.append(
                    {
                        "frame": str(frame_path),
                        "frame_name": frame_path.name,
                        "frame_size": f"{image.width}x{image.height}",
                        "slot_id": slot.slot_id,
                        "player": slot.player,
                        "turn_index": slot.turn_index,
                        "orientation": slot.orientation,
                        "bbox": slot.bbox,
                        "quad": [[x, y] for x, y in slot.corners],
                        "used_refinement": refinement is not None,
                    }
                )
                crops.append(crop)

    accepted: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    accepted_by_tile: Counter[str] = Counter()
    for offset in range(0, len(crops), max(1, int(args.batch_size))):
        batch_crops = crops[offset : offset + max(1, int(args.batch_size))]
        batch_items = pending[offset : offset + max(1, int(args.batch_size))]
        predictions = classify_tile_crops(
            batch_crops,
            model=args.model,
            device=_coerce_device(args.device),
            top_k=max(1, int(args.top_k)),
        )
        for item, crop, prediction in zip(batch_items, batch_crops, predictions, strict=True):
            reason = _rejection_reason(
                prediction,
                min_confidence=args.min_confidence,
                min_margin=args.min_margin,
            )
            if reason:
                rejected_reasons[reason] += 1
                continue
            assert prediction is not None
            tile = prediction.tile
            accepted_by_tile[tile] += 1
            tile_index = accepted_by_tile[tile]
            filename = (
                f"{tile}_{tile_index:04d}_{Path(item['frame_name']).stem}_"
                f"{item['slot_id']}_c{prediction.confidence:.3f}.png"
            )
            relative_path = Path(tile) / _safe_filename(filename)
            output_path = args.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path)
            top_k = list(prediction.top_k)
            accepted.append(
                {
                    **item,
                    "tile": tile,
                    "vit_label": prediction.label,
                    "confidence": round(float(prediction.confidence), 4),
                    "top_k": top_k,
                    "path": str(relative_path).replace("\\", "/"),
                }
            )

    manifest = {
        "schema": "mahjong-vit-labeled-discard-crops-v1",
        "model": args.model,
        "device": str(args.device),
        "top_k": max(1, int(args.top_k)),
        "min_confidence": float(args.min_confidence),
        "min_margin": float(args.min_margin),
        "frame_count": len(frame_paths),
        "frame_sizes": [f"{width}x{height}" for width, height in sorted(frame_sizes)],
        "candidate_crop_count": len(crops),
        "accepted_crop_count": len(accepted),
        "rejected_crop_count": len(crops) - len(accepted),
        "accepted_by_tile": dict(sorted(accepted_by_tile.items())),
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "items": accepted,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("frame_paths", len(frame_paths))
    print("candidate_crops", len(crops))
    print("accepted_crops", len(accepted))
    print("rejected_crops", len(crops) - len(accepted))
    print("accepted_by_tile", dict(sorted(accepted_by_tile.items())))
    print("output_dir", args.output_dir)
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
    width, height = _parse_resolution(resolution)
    matched: list[Path] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                if image.size == (width, height):
                    matched.append(path)
        except OSError:
            continue
    return matched


def _parse_resolution(value: str) -> tuple[int, int]:
    normalized = str(value or "").strip().lower().replace("*", "x")
    parts = normalized.split("x", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("--frame-resolution must look like 1920x1080")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("--frame-resolution must use positive width and height")
    return width, height


def _rejection_reason(prediction: Any, *, min_confidence: float, min_margin: float) -> str:
    if prediction is None:
        return "no_prediction"
    if float(prediction.confidence or 0.0) < min_confidence:
        return "low_confidence"
    top_k = prediction.top_k if isinstance(prediction.top_k, list) else []
    if len(top_k) >= 2:
        first = _coerce_float(top_k[0].get("score"), default=0.0)
        second = _coerce_float(top_k[1].get("score"), default=0.0)
        if first - second < min_margin:
            return "low_margin"
    return ""


def _coerce_device(value: str) -> Any:
    text = str(value).strip()
    if not text:
        return -1
    try:
        return int(text)
    except ValueError:
        return text


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_filename(value: str) -> str:
    return "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
