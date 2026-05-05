from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from plugin.plugins.mahjong_companion.perception.calibration import resolve_calibration_profile
from plugin.plugins.mahjong_companion.perception.hand_layout import TileSlot, build_hand_layout
from plugin.plugins.mahjong_companion.perception.roi import collect_region_metrics
from plugin.plugins.mahjong_companion.perception.tile_parser import (
    DISCARD_TURN_HAND_COUNTS,
    WAITING_HAND_COUNTS,
)
from plugin.plugins.mahjong_companion.perception.tile_templates import is_probably_occupied_hand_slot
from plugin.plugins.mahjong_companion.perception.vit_tile_classifier import (
    DEFAULT_VIT_MODEL,
    classify_tile_crops,
)


DRAW_SLOT_INDICES = (14, 11, 8, 5, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ViT-labeled Mahjong Soul hand crops grouped by tile.")
    parser.add_argument("roots", nargs="+", type=Path, help="Full-frame screenshot files or directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("plugin/plugins/mahjong_companion/data/calibration"),
    )
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

    candidates_by_frame: list[list[dict[str, Any]]] = []
    crops: list[Image.Image] = []
    frame_sizes: set[tuple[int, int]] = set()
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as opened:
                image = opened.convert("RGB")
        except OSError:
            candidates_by_frame.append([])
            continue
        frame_sizes.add(image.size)
        calibration = resolve_calibration_profile(*image.size, calibration_dir=args.calibration_dir)
        candidates_by_frame.append(
            _prepare_frame_candidates(
                image,
                frame_path=frame_path,
                calibration=calibration,
                crops=crops,
            )
        )

    predictions: list[Any] = []
    for offset in range(0, len(crops), max(1, int(args.batch_size))):
        predictions.extend(
            classify_tile_crops(
                crops[offset : offset + max(1, int(args.batch_size))],
                model=args.model,
                device=_coerce_device(args.device),
                top_k=max(1, int(args.top_k)),
            )
        )

    accepted_items: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    accepted_by_tile: Counter[str] = Counter()
    selected_hand_counts: Counter[str] = Counter()
    for frame_candidates in candidates_by_frame:
        finalized = [
            _finalize_candidate(
                candidate,
                predictions=predictions,
                min_confidence=args.min_confidence,
                min_margin=args.min_margin,
                rejected_reasons=rejected_reasons,
            )
            for candidate in frame_candidates
        ]
        if not finalized:
            continue
        selected = max(finalized, key=_hand_layout_score)
        selected_hand_counts[str(len(selected["accepted"]))] += 1
        for accepted in selected["accepted"]:
            tile = accepted["tile"]
            accepted_by_tile[tile] += 1
            tile_index = accepted_by_tile[tile]
            crop = crops[int(accepted["crop_index"])]
            filename = (
                f"{tile}_{tile_index:04d}_{Path(accepted['frame_name']).stem}_"
                f"{accepted['slot_id']}_c{accepted['confidence']:.3f}.png"
            )
            relative_path = Path(tile) / _safe_filename(filename)
            output_path = args.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path)
            accepted_items.append(
                {
                    key: value
                    for key, value in accepted.items()
                    if key not in {"crop_index"}
                }
                | {
                    "path": str(relative_path).replace("\\", "/"),
                    "selected_draw_slot_index": selected["draw_slot_index"],
                    "selected_hand_count": len(selected["accepted"]),
                }
            )

    manifest = {
        "schema": "mahjong-vit-labeled-hand-crops-v1",
        "model": args.model,
        "device": str(args.device),
        "top_k": max(1, int(args.top_k)),
        "min_confidence": float(args.min_confidence),
        "min_margin": float(args.min_margin),
        "frame_count": len(frame_paths),
        "frame_sizes": [f"{width}x{height}" for width, height in sorted(frame_sizes)],
        "candidate_crop_count": len(crops),
        "accepted_crop_count": len(accepted_items),
        "accepted_by_tile": dict(sorted(accepted_by_tile.items())),
        "selected_hand_counts": dict(sorted(selected_hand_counts.items())),
        "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "items": accepted_items,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("frame_paths", len(frame_paths))
    print("candidate_crops", len(crops))
    print("accepted_crops", len(accepted_items))
    print("accepted_by_tile", dict(sorted(accepted_by_tile.items())))
    print("selected_hand_counts", dict(sorted(selected_hand_counts.items())))
    print("output_dir", args.output_dir)
    return 0


def _prepare_frame_candidates(
    image: Image.Image,
    *,
    frame_path: Path,
    calibration: Any,
    crops: list[Image.Image],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for draw_slot_index in DRAW_SLOT_INDICES:
        layout = build_hand_layout(
            image.width,
            image.height,
            calibration=calibration,
            draw_slot_index=draw_slot_index,
        )
        crop_refs: list[dict[str, Any]] = []
        seen_occupied = False
        for slot in layout["hand"][:14]:
            metrics = _collect_hand_slot_metrics(image, slot)
            occupied = is_probably_occupied_hand_slot(metrics)
            if not occupied:
                if seen_occupied:
                    break
                continue
            seen_occupied = True
            crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))
            crop_refs.append(
                {
                    "crop_index": len(crops),
                    "frame": str(frame_path),
                    "frame_name": frame_path.name,
                    "frame_size": f"{image.width}x{image.height}",
                    "slot_id": slot.slot_id,
                    "box": slot.box.to_dict(),
                    "slot_mean_luma": metrics.get("slot_mean_luma"),
                    "slot_colorful_ratio": metrics.get("slot_colorful_ratio"),
                    "draw_slot_index": draw_slot_index,
                }
            )
            crops.append(crop)
        candidates.append(
            {
                "draw_slot_index": draw_slot_index,
                "crop_refs": crop_refs,
                "accepted": [],
                "confidences": [],
            }
        )
    return candidates


def _finalize_candidate(
    candidate: dict[str, Any],
    *,
    predictions: list[Any],
    min_confidence: float,
    min_margin: float,
    rejected_reasons: Counter[str],
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    confidences: list[float] = []
    for crop_ref in candidate["crop_refs"]:
        crop_index = int(crop_ref["crop_index"])
        if crop_index >= len(predictions):
            rejected_reasons["missing_prediction"] += 1
            continue
        prediction = predictions[crop_index]
        reason = _rejection_reason(
            prediction,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        if reason:
            rejected_reasons[reason] += 1
            continue
        assert prediction is not None
        confidence = round(float(prediction.confidence), 4)
        accepted.append(
            {
                **crop_ref,
                "tile": prediction.tile,
                "vit_label": prediction.label,
                "confidence": confidence,
                "top_k": list(prediction.top_k),
            }
        )
        confidences.append(confidence)
    candidate = dict(candidate)
    candidate["accepted"] = accepted
    candidate["confidences"] = confidences
    return candidate


def _collect_hand_slot_metrics(image: Image.Image, slot: TileSlot) -> dict[str, Any]:
    metrics = collect_region_metrics(image, slot.box, sample_step=4)
    return {
        "slot_mean_luma": metrics["mean_luma"],
        "slot_bright_ratio": metrics["bright_ratio"],
        "slot_dark_ratio": metrics["dark_ratio"],
        "slot_colorful_ratio": metrics["colorful_ratio"],
        "slot_stddev": metrics["stddev"],
    }


def _hand_layout_score(result: dict[str, Any]) -> tuple[int, int, float, int]:
    accepted = result.get("accepted") if isinstance(result.get("accepted"), list) else []
    confidences = result.get("confidences") if isinstance(result.get("confidences"), list) else []
    count = len(accepted)
    mean_confidence = sum(float(value or 0.0) for value in confidences) / max(1, len(confidences))
    shape_score = 2 if count in (DISCARD_TURN_HAND_COUNTS | WAITING_HAND_COUNTS) else 0
    if count >= 12:
        shape_score += 1
    return (shape_score, count, mean_confidence, -abs(int(result.get("draw_slot_index", 14) or 14) - 14))


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
