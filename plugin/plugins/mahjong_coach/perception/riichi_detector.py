from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .discard_layout import BASE_WIDTH, BASE_HEIGHT, build_discard_layout
from .roi import RoiBox, collect_region_metrics

_BASE_WIDTH = BASE_WIDTH
_BASE_HEIGHT = BASE_HEIGHT

# For each opponent, a sideways (riichi) tile extends beyond the normal slot
# boundary. We check the "overflow region" — the area just outside the slot
# in the direction perpendicular to the normal tile orientation.
# If that overflow region has tile-like pixels (bright, occupied), the tile
# is sideways = riichi.
_OVERFLOW_CHECKS: dict[str, dict[str, Any]] = {
    # top_opponent: tiles are portrait (h > w). Sideways tile extends horizontally.
    # Check regions to left and right of the slot.
    "top_opponent": {
        "axis": "x",
        "overflow_px": 18,
        "tile_threshold_width_ratio": 1.2,
    },
    # left_opponent: tiles are landscape-ish. Sideways tile extends vertically.
    "left_opponent": {
        "axis": "y",
        "overflow_px": 18,
        "tile_threshold_width_ratio": 1.2,
    },
    # right_opponent: same as left.
    "right_opponent": {
        "axis": "y",
        "overflow_px": 18,
        "tile_threshold_width_ratio": 1.2,
    },
}


@dataclass
class RiichiDetectResult:
    riichi_players: list[str] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"riichi_players": list(self.riichi_players), "detections": list(self.detections)}


def detect_riichi_sticks(image_path: Path) -> RiichiDetectResult:
    if not image_path.exists():
        return RiichiDetectResult()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    layout = build_discard_layout(image.width, image.height)
    detected: list[str] = []
    detections: list[dict[str, Any]] = []

    for player, spec in _OVERFLOW_CHECKS.items():
        slots = layout.get(player, [])
        if not slots:
            continue
        # Check last 4 discard slots — riichi tile is typically one of the recent discards
        is_riichi = False
        best_score = 0.0
        for slot in slots[-4:]:
            score = _sideways_score(image, slot, spec)
            if score > best_score:
                best_score = score
            if score >= 0.5:
                is_riichi = True
        detections.append({
            "player": player,
            "best_score": round(best_score, 4),
            "is_riichi": is_riichi,
        })
        if is_riichi:
            detected.append(player)

    return RiichiDetectResult(riichi_players=detected, detections=detections)


def _sideways_score(image: Image.Image, slot: Any, spec: dict[str, Any]) -> float:
    """Score how likely this discard slot contains a sideways (riichi) tile.

    A normal tile fits within the slot. A sideways tile overflows into
    adjacent regions. We compare the pixel content in the overflow region
    vs the slot itself.
    """
    box: RoiBox = slot.box
    overflow_px = int(spec.get("overflow_px", 18) * image.width / _BASE_WIDTH)
    overflow_px = max(4, overflow_px)

    # Get metrics for the slot itself
    slot_metrics = collect_region_metrics(image, box, sample_step=4)
    slot_bright = float(slot_metrics.get("bright_ratio") or 0)
    slot_luma = float(slot_metrics.get("mean_luma") or 0)

    # If the slot doesn't look occupied, skip
    if slot_luma < 70 or slot_bright < 0.05:
        return 0.0

    # Check overflow regions based on axis
    axis = spec.get("axis", "x")
    overflow_boxes: list[RoiBox] = []

    if axis == "x":
        # Check left and right of slot
        left_box = RoiBox(
            f"{box.name}_left",
            max(0, box.left - overflow_px),
            box.top,
            overflow_px,
            box.height,
        ).clipped(image.width, image.height)
        right_box = RoiBox(
            f"{box.name}_right",
            box.right,
            box.top,
            overflow_px,
            box.height,
        ).clipped(image.width, image.height)
        overflow_boxes = [left_box, right_box]
    else:
        # Check above and below slot
        top_box = RoiBox(
            f"{box.name}_top",
            box.left,
            max(0, box.top - overflow_px),
            box.width,
            overflow_px,
        ).clipped(image.width, image.height)
        bottom_box = RoiBox(
            f"{box.name}_bottom",
            box.left,
            box.bottom,
            box.width,
            overflow_px,
        ).clipped(image.width, image.height)
        overflow_boxes = [top_box, bottom_box]

    # If the tile is sideways, the overflow regions will have tile-like content
    overflow_scores: list[float] = []
    for ob in overflow_boxes:
        if ob.width <= 2 or ob.height <= 2:
            continue
        m = collect_region_metrics(image, ob, sample_step=4)
        bright = float(m.get("bright_ratio") or 0)
        luma = float(m.get("mean_luma") or 0)
        # Tile-like: bright and high luma
        overflow_scores.append(bright * 0.6 + min(luma / 200.0, 1.0) * 0.4)

    if not overflow_scores:
        return 0.0

    max_overflow = max(overflow_scores)
    # Score: how tile-like is the overflow region compared to the slot itself
    # A sideways tile will have overflow scores close to the slot's own scores
    slot_score = slot_bright * 0.6 + min(slot_luma / 200.0, 1.0) * 0.4
    if slot_score < 0.15:
        return 0.0
    return min(max_overflow / slot_score, 1.0)
