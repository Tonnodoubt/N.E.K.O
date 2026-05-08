from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import CalibrationProfile
from .hand_baseline import HandBaselineAnchor
from .roi import RoiBox


@dataclass(frozen=True)
class TileSlot:
    slot_id: str
    box: RoiBox
    group: str = "hand"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "group": self.group,
            "box": self.box.to_dict(),
        }


def build_hand_layout(
    width: int,
    height: int,
    *,
    calibration: CalibrationProfile | None = None,
    draw_slot_index: int = 14,
    baseline: HandBaselineAnchor | None = None,
) -> dict[str, list[TileSlot]]:
    draw_slot_index = _bounded_hand_slot_index(draw_slot_index)
    if baseline is not None and _baseline_plausible(baseline, width, height):
        return _build_anchor_layout(
            baseline, width, height, draw_slot_index=draw_slot_index,
            calibration=calibration,
        )
    calibration = calibration or CalibrationProfile(screen_width=width, screen_height=height)
    return _build_hardcoded_layout(
        width, height, calibration=calibration, draw_slot_index=draw_slot_index,
    )


def _baseline_plausible(
    baseline: HandBaselineAnchor,
    width: int,
    height: int,
) -> bool:
    if baseline.image_size != (width, height):
        return False
    if baseline.left_x > width * 0.3:
        return False
    if baseline.top_y < height * 0.75 or baseline.top_y > height * 0.95:
        return False
    return True


def _build_anchor_layout(
    baseline: HandBaselineAnchor,
    width: int,
    height: int,
    *,
    draw_slot_index: int,
    calibration: CalibrationProfile | None = None,
) -> dict[str, list[TileSlot]]:
    from .anchor_geometry import anchor_derived_rois

    layout = anchor_derived_rois(baseline, width, height)
    origin = layout.hand
    cal = calibration

    tile_width = origin.width + (cal.hand_offsets.width_px if cal else 0)
    tile_height = origin.height + (cal.hand_offsets.height_px if cal else 0)
    gap = max(0, int(tile_width * 0.12) + (cal.hand_offsets.gap_px if cal else 0))
    draw_gap = max(0, int(tile_width * 0.35) + (cal.hand_offsets.draw_gap_px if cal else 0))
    hand_left = origin.left + (cal.hand_offsets.x_px if cal else 0)
    hand_top = origin.top + (cal.hand_offsets.y_px if cal else 0)

    hand_slots = [
        TileSlot(
            slot_id=f"hand_{index + 1}",
            group="hand",
            box=RoiBox(
                name=f"hand_{index + 1}",
                left=hand_left + index * (tile_width + gap) + (draw_gap if index == draw_slot_index - 1 else 0),
                top=hand_top,
                width=tile_width,
                height=tile_height,
            ),
        )
        for index in range(14)
    ]

    dora_origin = layout.dora_origin
    dora_gap = max(2, int(dora_origin.width * 0.1))
    dora_slots = [
        TileSlot(
            slot_id=f"dora_{index + 1}",
            group="dora",
            box=RoiBox(
                name=f"dora_{index + 1}",
                left=dora_origin.left + index * (dora_origin.width + dora_gap),
                top=dora_origin.top,
                width=dora_origin.width,
                height=dora_origin.height,
            ),
        )
        for index in range(5)
    ]

    meld_origin = layout.meld_origin
    meld_gap = max(2, int(meld_origin.width * 0.08))
    meld_slots = [
        TileSlot(
            slot_id=f"meld_{index + 1}",
            group="meld",
            box=RoiBox(
                name=f"meld_{index + 1}",
                left=meld_origin.left + index * (meld_origin.width + meld_gap),
                top=meld_origin.top,
                width=meld_origin.width,
                height=meld_origin.height,
            ),
        )
        for index in range(4)
    ]

    return {"hand": hand_slots, "dora": dora_slots, "meld": meld_slots}


def _build_hardcoded_layout(
    width: int,
    height: int,
    *,
    calibration: CalibrationProfile,
    draw_slot_index: int,
) -> dict[str, list[TileSlot]]:
    hand_left = int(width * 0.14) + calibration.hand_offsets.x_px
    hand_top = int(height * 0.72) + calibration.hand_offsets.y_px
    tile_width = max(18, int(width * 0.036) + calibration.hand_offsets.width_px)
    tile_height = max(26, int(height * 0.112) + calibration.hand_offsets.height_px)
    gap = max(0, int(tile_width * 0.12) + calibration.hand_offsets.gap_px)
    draw_gap = max(0, calibration.hand_offsets.draw_gap_px)

    hand_slots = [
        TileSlot(
            slot_id=f"hand_{index + 1}",
            group="hand",
            box=RoiBox(
                name=f"hand_{index + 1}",
                left=hand_left + index * (tile_width + gap) + (draw_gap if index == draw_slot_index - 1 else 0),
                top=hand_top,
                width=tile_width,
                height=tile_height,
            ),
        )
        for index in range(14)
    ]

    dora_width = max(18, int(width * 0.034) + calibration.dora_offsets.width_px)
    dora_height = max(24, int(height * 0.09) + calibration.dora_offsets.height_px)
    dora_left = int(width * 0.43) + calibration.dora_offsets.x_px
    dora_top = int(height * 0.10) + calibration.dora_offsets.y_px
    dora_slots = [
        TileSlot(
            slot_id=f"dora_{index + 1}",
            group="dora",
            box=RoiBox(
                name=f"dora_{index + 1}",
                left=dora_left + index * (dora_width + max(2, int(dora_width * 0.1))),
                top=dora_top,
                width=dora_width,
                height=dora_height,
            ),
        )
        for index in range(5)
    ]

    meld_width = max(20, int(width * 0.042) + calibration.meld_offsets.width_px)
    meld_height = max(24, int(height * 0.10) + calibration.meld_offsets.height_px)
    meld_left = int(width * 0.72) + calibration.meld_offsets.x_px
    meld_top = int(height * 0.54) + calibration.meld_offsets.y_px
    meld_slots = [
        TileSlot(
            slot_id=f"meld_{index + 1}",
            group="meld",
            box=RoiBox(
                name=f"meld_{index + 1}",
                left=meld_left + index * (meld_width + max(2, int(meld_width * 0.08))),
                top=meld_top,
                width=meld_width,
                height=meld_height,
            ),
        )
        for index in range(4)
    ]

    return {"hand": hand_slots, "dora": dora_slots, "meld": meld_slots}


def _bounded_hand_slot_index(value: int) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        index = 14
    return min(14, max(1, index))
