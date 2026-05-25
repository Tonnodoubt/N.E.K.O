from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .roi import RoiBox


BASE_WIDTH = 1920
BASE_HEIGHT = 1080
DISCARD_PLAYERS = ("self", "left_opponent", "top_opponent", "right_opponent")


@dataclass(frozen=True)
class DiscardSlot:
    slot_id: str
    player: str
    turn_index: int
    orientation: str
    box: RoiBox
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]

    @property
    def bbox(self) -> list[int]:
        xs = [point[0] for point in self.quad]
        ys = [point[1] for point in self.quad]
        return [min(xs), min(ys), max(xs), max(ys)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "player": self.player,
            "turn_index": self.turn_index,
            "orientation": self.orientation,
            "bbox": self.bbox,
            "quad": [[x, y] for x, y in self.quad],
            "box": self.box.to_dict(),
        }


@dataclass(frozen=True)
class _LayoutSpec:
    origin_left: int
    origin_top: int
    tile_width: int
    tile_height: int
    step_x: int
    step_y: int
    columns: int
    rows: int
    orientation: str
    order: str = "row_major"


_BASE_LAYOUTS = {
    "self": _LayoutSpec(762, 542, 58, 70, 64, 70, 6, 3, "bottom"),
    "left_opponent": _LayoutSpec(624, 290, 84, 58, 82, 62, 3, 6, "left", "column_major"),
    "top_opponent": _LayoutSpec(802, 242, 58, 70, 64, -70, 6, 3, "top"),
    "right_opponent": _LayoutSpec(1148, 290, 84, 58, 82, 62, 3, 6, "right", "column_major"),
}


def build_discard_layout(width: int, height: int) -> dict[str, list[DiscardSlot]]:
    screen_width = max(1, int(width))
    screen_height = max(1, int(height))
    return {
        player: _build_player_slots(player, spec, screen_width, screen_height)
        for player, spec in _BASE_LAYOUTS.items()
    }


def _build_player_slots(
    player: str,
    spec: _LayoutSpec,
    screen_width: int,
    screen_height: int,
) -> list[DiscardSlot]:
    coordinates = (
        ((row, column) for column in range(spec.columns) for row in range(spec.rows))
        if spec.order == "column_major"
        else ((row, column) for row in range(spec.rows) for column in range(spec.columns))
    )
    slots: list[DiscardSlot] = []
    for turn_index, (row, column) in enumerate(coordinates, start=1):
        name = f"discard_{player}_{turn_index:02d}"
        quad = _scaled_quad(
            left=spec.origin_left + column * spec.step_x,
            top=spec.origin_top + row * spec.step_y,
            box_width=spec.tile_width,
            box_height=spec.tile_height,
            screen_width=screen_width,
            screen_height=screen_height,
            orientation=spec.orientation,
        )
        slots.append(
            DiscardSlot(
                slot_id=name,
                player=player,
                turn_index=turn_index,
                orientation=spec.orientation,
                box=_box_from_quad(name, quad, screen_width, screen_height),
                quad=quad,
            )
        )
    return slots


def _scaled_quad(
    *,
    left: int,
    top: int,
    box_width: int,
    box_height: int,
    screen_width: int,
    screen_height: int,
    orientation: str,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    scale_x = screen_width / BASE_WIDTH
    scale_y = screen_height / BASE_HEIGHT
    scaled_left = _clamp_int(round(left * scale_x), 0, screen_width - 1)
    scaled_top = _clamp_int(round(top * scale_y), 0, screen_height - 1)
    scaled_width = max(1, int(round(box_width * scale_x)))
    scaled_height = max(1, int(round(box_height * scale_y)))
    right = _clamp_int(scaled_left + scaled_width, 1, screen_width)
    bottom = _clamp_int(scaled_top + scaled_height, 1, screen_height)

    if orientation == "left":
        skew_x = max(1, scaled_width // 8)
        return (
            (scaled_left, scaled_top),
            (_clamp_int(scaled_left - skew_x, 0, screen_width - 1), bottom),
            (_clamp_int(right - skew_x, 1, screen_width), bottom),
            (right, scaled_top),
        )
    if orientation == "right":
        skew_x = max(1, scaled_width // 8)
        return (
            (_clamp_int(scaled_left + skew_x, 0, screen_width - 1), scaled_top),
            (scaled_left, bottom),
            (right, bottom),
            (_clamp_int(right + skew_x, 1, screen_width), scaled_top),
        )
    if orientation == "top":
        skew_x = max(1, scaled_width // 12)
        return (
            (_clamp_int(scaled_left + skew_x, 0, screen_width - 1), scaled_top),
            (scaled_left, bottom),
            (right, bottom),
            (_clamp_int(right - skew_x, 1, screen_width), scaled_top),
        )
    return ((scaled_left, scaled_top), (scaled_left, bottom), (right, bottom), (right, scaled_top))


def _box_from_quad(
    name: str,
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    screen_width: int,
    screen_height: int,
) -> RoiBox:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    left = _clamp_int(min(xs), 0, screen_width - 1)
    top = _clamp_int(min(ys), 0, screen_height - 1)
    right = _clamp_int(max(xs), left + 1, screen_width)
    bottom = _clamp_int(max(ys), top + 1, screen_height)
    return RoiBox(name=name, left=left, top=top, width=right - left, height=bottom - top)


def _clamp_int(value: float | int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
