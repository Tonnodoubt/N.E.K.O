from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import CalibrationProfile
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
) -> dict[str, list[TileSlot]]:
    calibration = calibration or CalibrationProfile(screen_width=width, screen_height=height)
    draw_slot_index = max(1, min(14, int(draw_slot_index or 14)))
    hand_left = int(width * 0.14) + calibration.hand_offsets.x_px
    hand_top = int(height * 0.72) + calibration.hand_offsets.y_px
    tile_width = max(18, int(width * 0.036) + calibration.hand_offsets.width_px)
    tile_height = max(26, int(height * 0.112) + calibration.hand_offsets.height_px)
    gap = max(0, int(tile_width * 0.12) + calibration.hand_offsets.gap_px)
    draw_gap = max(0, calibration.hand_offsets.draw_gap_px)
    return {
        "hand": [
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
    }

