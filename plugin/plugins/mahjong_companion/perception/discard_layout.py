from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import CalibrationProfile
from .hand_baseline import HandBaselineAnchor
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
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None = None

    @property
    def corners(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        if self.quad is not None:
            return self.quad
        return (
            (self.box.left, self.box.top),
            (self.box.left, self.box.bottom),
            (self.box.right, self.box.bottom),
            (self.box.right, self.box.top),
        )

    @property
    def bbox(self) -> list[int]:
        xs = [point[0] for point in self.corners]
        ys = [point[1] for point in self.corners]
        return [min(xs), min(ys), max(xs), max(ys)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "player": self.player,
            "turn_index": self.turn_index,
            "orientation": self.orientation,
            "bbox": self.bbox,
            "quad": [[x, y] for x, y in self.corners],
            "quad_order": "upper_left,lower_left,lower_right,upper_right",
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
    "self": _LayoutSpec(
        origin_left=762,
        origin_top=542,
        tile_width=58,
        tile_height=70,
        step_x=64,
        step_y=70,
        columns=6,
        rows=3,
        orientation="bottom",
    ),
    "left_opponent": _LayoutSpec(
        origin_left=624,
        origin_top=290,
        tile_width=84,
        tile_height=58,
        step_x=82,
        step_y=62,
        columns=3,
        rows=6,
        orientation="left",
        order="column_major",
    ),
    "top_opponent": _LayoutSpec(
        origin_left=802,
        origin_top=242,
        tile_width=58,
        tile_height=70,
        step_x=64,
        step_y=-70,
        columns=6,
        rows=3,
        orientation="top",
    ),
    "right_opponent": _LayoutSpec(
        origin_left=1148,
        origin_top=290,
        tile_width=84,
        tile_height=58,
        step_x=82,
        step_y=62,
        columns=3,
        rows=6,
        orientation="right",
        order="column_major",
    ),
}


def build_discard_layout(
    width: int,
    height: int,
    *,
    calibration: CalibrationProfile | None = None,
    baseline: HandBaselineAnchor | None = None,
) -> dict[str, list[DiscardSlot]]:
    screen_width = _positive_int(width)
    screen_height = _positive_int(height)
    _ = calibration

    if baseline is not None and _baseline_plausible(baseline, width, height):
        return _build_anchor_layout(baseline, width, height)

    return {
        player: _build_player_slots(player, spec, screen_width, screen_height)
        for player, spec in _BASE_LAYOUTS.items()
    }


def _baseline_plausible(
    baseline: HandBaselineAnchor,
    width: int,
    height: int,
) -> bool:
    if baseline.left_x > width * 0.3:
        return False
    if baseline.top_y < height * 0.75 or baseline.top_y > height * 0.95:
        return False
    return True


def _build_anchor_layout(
    baseline: HandBaselineAnchor,
    width: int,
    height: int,
) -> dict[str, list[DiscardSlot]]:
    from .anchor_geometry import anchor_derived_rois

    layout = anchor_derived_rois(baseline, width, height)
    result: dict[str, list[DiscardSlot]] = {}
    for player in DISCARD_PLAYERS:
        spec = layout.discard[player]
        result[player] = _build_player_slots(
            player,
            _LayoutSpec(
                origin_left=spec.origin_left,
                origin_top=spec.origin_top,
                tile_width=spec.tile_width,
                tile_height=spec.tile_height,
                step_x=spec.step_x,
                step_y=spec.step_y,
                columns=spec.columns,
                rows=spec.rows,
                orientation=spec.orientation,
                order=spec.order,
            ),
            width,
            height,
        )
    return result


def _build_player_slots(
    player: str,
    spec: _LayoutSpec,
    screen_width: int,
    screen_height: int,
) -> list[DiscardSlot]:
    slots: list[DiscardSlot] = []
    coordinates = (
        ((row, column) for column in range(spec.columns) for row in range(spec.rows))
        if spec.order == "column_major"
        else ((row, column) for row in range(spec.rows) for column in range(spec.columns))
    )
    for turn_index, (row, column) in enumerate(coordinates, start=1):
        quad = _scaled_quad(
            name=f"discard_{player}_{turn_index:02d}",
            left=spec.origin_left + column * spec.step_x,
            top=spec.origin_top + row * spec.step_y,
            box_width=spec.tile_width,
            box_height=spec.tile_height,
            screen_width=screen_width,
            screen_height=screen_height,
            orientation=spec.orientation,
        )
        box = _box_from_quad(name=f"discard_{player}_{turn_index:02d}", quad=quad, screen_width=screen_width, screen_height=screen_height)
        slots.append(
            DiscardSlot(
                slot_id=box.name,
                player=player,
                turn_index=turn_index,
                orientation=spec.orientation,
                box=box,
                quad=quad,
            )
        )
    return slots


def _scaled_quad(
    *,
    name: str,
    left: int,
    top: int,
    box_width: int,
    box_height: int,
    screen_width: int,
    screen_height: int,
    orientation: str = "bottom",
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    _ = name
    scale_x = screen_width / BASE_WIDTH
    scale_y = screen_height / BASE_HEIGHT
    scaled_left = _clamp_int(int(round(left * scale_x)), 0, screen_width - 1)
    scaled_top = _clamp_int(int(round(top * scale_y)), 0, screen_height - 1)
    scaled_width = max(1, int(round(box_width * scale_x)))
    scaled_height = max(1, int(round(box_height * scale_y)))

    # Perspective skew for opponent tiles — the 3-D table surface means
    # side-player tiles are parallelograms, not axis-aligned rectangles.
    if orientation == "left":
        # Tile on left side of screen: top edge leans right (closer edge is taller).
        skew_x = max(1, scaled_width // 8)
        return (
            (scaled_left, scaled_top),                            # upper-left
            (scaled_left - skew_x, scaled_top + scaled_height),    # lower-left
            (scaled_left + scaled_width - skew_x, scaled_top + scaled_height),  # lower-right
            (scaled_left + scaled_width, scaled_top),             # upper-right
        )
    elif orientation == "right":
        # Tile on right side: top edge leans left.
        skew_x = max(1, scaled_width // 8)
        return (
            (scaled_left + skew_x, scaled_top),                    # upper-left
            (scaled_left, scaled_top + scaled_height),             # lower-left
            (scaled_left + scaled_width, scaled_top + scaled_height),  # lower-right
            (scaled_left + scaled_width + skew_x, scaled_top),     # upper-right
        )
    elif orientation == "top":
        # Far side: trapezoidal — far edge (top of screen) is narrower.
        skew_x = max(1, scaled_width // 12)
        return (
            (scaled_left + skew_x, scaled_top),                    # upper-left (narrower)
            (scaled_left, scaled_top + scaled_height),             # lower-left
            (scaled_left + scaled_width, scaled_top + scaled_height),  # lower-right
            (scaled_left + scaled_width - skew_x, scaled_top),     # upper-right (narrower)
        )
    else:
        # bottom (self) — axis-aligned rectangle.
        right = _clamp_int(scaled_left + scaled_width, 1, screen_width)
        bottom = _clamp_int(scaled_top + scaled_height, 1, screen_height)
        if right <= scaled_left:
            right = min(screen_width, scaled_left + 1)
        if bottom <= scaled_top:
            bottom = min(screen_height, scaled_top + 1)
        return (
            (scaled_left, scaled_top),
            (scaled_left, bottom),
            (right, bottom),
            (right, scaled_top),
        )


def _box_from_quad(
    *,
    name: str,
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    screen_width: int,
    screen_height: int,
) -> RoiBox:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    clamped_left = _clamp_int(min(xs), 0, screen_width - 1)
    clamped_top = _clamp_int(min(ys), 0, screen_height - 1)
    clamped_right = _clamp_int(max(xs), clamped_left + 1, screen_width)
    clamped_bottom = _clamp_int(max(ys), clamped_top + 1, screen_height)
    return RoiBox(
        name=name,
        left=clamped_left,
        top=clamped_top,
        width=clamped_right - clamped_left,
        height=clamped_bottom - clamped_top,
    )


def _positive_int(value: int) -> int:
    return max(1, int(value))


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
