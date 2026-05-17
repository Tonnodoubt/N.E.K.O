from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .calibration import resolve_calibration_profile
from .hand_layout import TileSlot, build_hand_layout
from .roi import collect_region_metrics
from .tile_classifier_dispatch import classify_hand_tile
from .tile_templates import is_probably_occupied_hand_slot

MIN_DRAW_TILE_CONFIDENCE = 0.12


@dataclass(frozen=True)
class DrawnTileFastPathResult:
    ok: bool = False
    tile: str = ""
    confidence: float = 0.0
    reason: str = ""
    slot_id: str = "hand_14"
    raw_detection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_drawn_tile_fast_path(
    image_path: Path,
    *,
    calibration_dir: Path | None = None,
    draw_slot_index: int = 14,
) -> DrawnTileFastPathResult:
    draw_slot_index = _bounded_hand_slot_index(draw_slot_index)
    requested_slot_id = f"hand_{draw_slot_index}"
    if not image_path.exists():
        return DrawnTileFastPathResult(reason="image_missing", slot_id=requested_slot_id)

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        calibration = resolve_calibration_profile(*image.size, calibration_dir=calibration_dir)
        if not calibration.enabled or not calibration.hand_tile_templates:
            return DrawnTileFastPathResult(reason="missing_hand_tile_templates", slot_id=requested_slot_id)

        layout = build_hand_layout(*image.size, calibration=calibration)
        draw_slot = _resolve_draw_slot(layout["hand"], draw_slot_index)
        slot_metrics = collect_region_metrics(image, draw_slot.box, sample_step=4)
        detection = {
            "slot_id": draw_slot.slot_id,
            "group": draw_slot.group,
            "candidate_tile": "",
            "confidence": 0.0,
            "box": draw_slot.box.to_dict(),
            "slot_mean_luma": slot_metrics.get("mean_luma"),
            "slot_colorful_ratio": slot_metrics.get("colorful_ratio"),
            "slot_bright_ratio": slot_metrics.get("bright_ratio"),
            "slot_dark_ratio": slot_metrics.get("dark_ratio"),
            "slot_stddev": slot_metrics.get("stddev"),
            "source": "drawn_tile_fast_path",
        }
        if not is_probably_occupied_hand_slot(
            {
                "slot_mean_luma": slot_metrics.get("mean_luma"),
                "slot_bright_ratio": slot_metrics.get("bright_ratio"),
                "slot_dark_ratio": slot_metrics.get("dark_ratio"),
                "slot_stddev": slot_metrics.get("stddev"),
            }
        ):
            detection["occupied"] = False
            return DrawnTileFastPathResult(
                reason="draw_slot_empty",
                slot_id=draw_slot.slot_id,
                raw_detection=detection,
            )

        crop = image.crop((draw_slot.box.left, draw_slot.box.top, draw_slot.box.right, draw_slot.box.bottom))
        match = classify_hand_tile(crop, calibration.hand_tile_templates)
        if match is None:
            detection["occupied"] = True
            return DrawnTileFastPathResult(
                reason="template_unmatched",
                slot_id=draw_slot.slot_id,
                raw_detection=detection,
            )

        detection.update(
            {
                "candidate_tile": match.tile,
                "confidence": match.confidence,
                "template_distance": match.distance,
                "runner_up_tile": match.runner_up_tile,
                "runner_up_distance": match.runner_up_distance,
                "occupied": True,
            }
        )
        if match.confidence < MIN_DRAW_TILE_CONFIDENCE:
            detection["accepted"] = False
            detection["rejection_reason"] = "low_confidence"
            return DrawnTileFastPathResult(
                tile=match.tile,
                confidence=match.confidence,
                reason="low_confidence",
                slot_id=draw_slot.slot_id,
                raw_detection=detection,
            )

        detection["accepted"] = True
        return DrawnTileFastPathResult(
            ok=True,
            tile=match.tile,
            confidence=match.confidence,
            reason="matched",
            slot_id=draw_slot.slot_id,
            raw_detection=detection,
        )


def _bounded_hand_slot_index(value: int) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        index = 14
    return min(14, max(1, index))


def _resolve_draw_slot(hand_slots: list[TileSlot], draw_slot_index: int) -> TileSlot:
    if not hand_slots:
        raise ValueError("hand_slots must not be empty for draw slot resolution")
    # Clamp before indexing: callers may pass draw_slot_index up to 14 even when
    # the recognised hand only has 13 slots. Indexing first risked IndexError that
    # took down the whole fast path (CODE_REVIEW_v1.2 M1).
    safe_index = max(1, min(draw_slot_index, len(hand_slots)))
    draw_slot = hand_slots[safe_index - 1]
    if safe_index == 14 or len(hand_slots) < 14:
        return draw_slot

    regular_gap = max(0, hand_slots[1].box.left - hand_slots[0].box.right)
    drawn_tile_gap = max(0, hand_slots[13].box.left - hand_slots[12].box.right - regular_gap)
    if drawn_tile_gap <= 0:
        return draw_slot

    box = draw_slot.box
    shifted_box = type(box)(
        name=box.name,
        left=box.left + drawn_tile_gap,
        top=box.top,
        width=box.width,
        height=box.height,
    )
    return TileSlot(slot_id=draw_slot.slot_id, group=draw_slot.group, box=shifted_box)
