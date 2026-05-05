from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..contracts import PerceivedGameState
from ..storage import write_json_atomic


def write_debug_artifacts(
    frame_path: Path,
    perceived: PerceivedGameState,
    debug_payload: dict[str, Any],
) -> dict[str, str]:
    base_path = frame_path.with_suffix("")
    perception_path = base_path.with_name(base_path.name + "-perception.json")
    overlay_path = base_path.with_name(base_path.name + "-overlay.json")

    write_json_atomic(
        perception_path,
        {
            "frame_path": str(frame_path),
            "perceived_state": perceived.to_dict(),
            "debug": debug_payload,
        },
    )
    write_json_atomic(
        overlay_path,
        {
            "frame_path": str(frame_path),
            "roi_boxes": debug_payload.get("roi_boxes", {}),
            "roi_hits": perceived.roi_hits,
            "button_regions": perceived.button_regions,
            "discard_piles": perceived.discard_piles,
            "notes": perceived.notes,
        },
    )
    artifacts = {
        "perception_path": str(perception_path),
        "overlay_path": str(overlay_path),
    }
    tile_overlay_path = _write_vit_tile_overlay(frame_path, perceived)
    if tile_overlay_path is not None:
        artifacts["vit_overlay_path"] = str(tile_overlay_path)
    return artifacts


def _write_vit_tile_overlay(frame_path: Path, perceived: PerceivedGameState) -> Path | None:
    vit_detections = [
        item
        for item in perceived.raw_detections
        if isinstance(item, dict)
        and str(item.get("source", "")).strip() in {"vit_classifier", "discard_vit_classifier"}
        and str(item.get("candidate_tile", "")).strip()
    ]
    if not vit_detections:
        return None

    try:
        with Image.open(frame_path) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        for detection in vit_detections:
            bbox = _detection_bbox(detection)
            if bbox is None:
                continue
            accepted = detection.get("accepted", True) is not False
            color = (35, 190, 95) if accepted else (230, 150, 35)
            draw.rectangle(bbox, outline=color, width=3)
            label = _detection_label(detection)
            if label:
                _draw_label(draw, bbox, label, color=color)
        stem = frame_path.stem
        if stem.endswith("-vit-overlay"):
            stem = stem[: -len("-vit-overlay")]
        output_path = frame_path.with_suffix("").with_name(stem + "-vit-overlay.png")
        image.save(output_path)
        return output_path
    except Exception:
        return None


def _detection_bbox(detection: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = detection.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(int(value) for value in bbox)  # type: ignore[return-value]
    box = detection.get("box")
    if not isinstance(box, dict):
        return None
    try:
        left = int(box.get("left", 0) or 0)
        top = int(box.get("top", 0) or 0)
        width = int(box.get("width", 0) or 0)
        height = int(box.get("height", 0) or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (left, top, left + width, top + height)


def _detection_label(detection: dict[str, Any]) -> str:
    tile = str(detection.get("candidate_tile", "")).strip()
    if not tile:
        return ""
    try:
        confidence = float(detection.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return f"{tile} {confidence:.2f}"


def _draw_label(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    label: str,
    *,
    color: tuple[int, int, int],
) -> None:
    left, top, right, _bottom = bbox
    text_bbox = draw.textbbox((left, top), label)
    label_height = text_bbox[3] - text_bbox[1] + 4
    label_width = text_bbox[2] - text_bbox[0] + 6
    label_top = max(0, top - label_height)
    label_right = min(max(right, left + label_width), left + label_width)
    draw.rectangle((left, label_top, label_right, label_top + label_height), fill=(12, 18, 24))
    draw.text((left + 3, label_top + 2), label, fill=color)
