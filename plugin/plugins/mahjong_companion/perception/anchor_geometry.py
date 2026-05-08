"""Anchor-relative geometry coefficients for Mahjong Soul layout.

Every ROI position (hand tiles, discard piles, meld area, dora indicators)
is expressed as a pair of ratios ``(rx, ry)`` relative to the hand baseline
anchor's ``(left_x, top_y)``, normalised by image dimensions.  Given a
detected :class:`HandBaselineAnchor` and the image size, downstream code
calls :func:`anchor_derived_rois` to obtain every ROI without reading any
calibration profile.

Coefficients are measured on the ``multi_theme`` fixture set (1920x1080,
two themes, 12 in-match frames).  Y-ratios have sub-0.5% variance; X-ratios
are sub-0.1% when excluding false-positive baseline detections.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hand_baseline import HandBaselineAnchor
from .roi import RoiBox


@dataclass(frozen=True)
class DiscardRoiSpec:
    player: str
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


@dataclass(frozen=True)
class AnchorDerivedLayout:
    hand: RoiBox
    meld_origin: RoiBox
    dora_origin: RoiBox
    discard: dict[str, DiscardRoiSpec]


def anchor_derived_rois(
    baseline: HandBaselineAnchor,
    width: int,
    height: int,
) -> AnchorDerivedLayout:
    """Compute all ROI positions from the detected hand baseline anchor.

    The baseline provides ``(left_x, top_y)`` as a stable reference point.
    Each ROI's origin is ``anchor.left + rx * width`` and
    ``anchor.top + ry * height``, where ``(rx, ry)`` are the empirically
    measured coefficients from the fixture set.
    """
    ref_x, ref_y = baseline.left_x, baseline.top_y

    hand_left = _px(ref_x, _HAND_RX, width)
    hand_top = _px(ref_y, _HAND_RY, height)
    hand_w = _dim(width, _HAND_TILE_W)
    hand_h = _dim(height, _HAND_TILE_H)

    meld_left = _px(ref_x, _MELD_RX, width)
    meld_top = _px(ref_y, _MELD_RY, height)
    meld_w = _dim(width, _MELD_TILE_W)
    meld_h = _dim(height, _MELD_TILE_H)

    dora_left = _px(ref_x, _DORA_RX, width)
    dora_top = _px(ref_y, _DORA_RY, height)
    dora_w = _dim(width, _DORA_TILE_W)
    dora_h = _dim(height, _DORA_TILE_H)

    discard_specs = {}
    for player, coeffs in _DISCARD_COEFFS.items():
        origin_left = _px(ref_x, coeffs["rx"], width)
        origin_top = _px(ref_y, coeffs["ry"], height)
        tw = _dim(width, coeffs["tile_wr"])
        th = _dim(height, coeffs["tile_hr"])
        sx = _dim(width, coeffs["step_xr"])
        sy = _dim(height, coeffs["step_yr"])
        discard_specs[player] = DiscardRoiSpec(
            player=player,
            origin_left=origin_left,
            origin_top=origin_top,
            tile_width=tw,
            tile_height=th,
            step_x=sx,
            step_y=sy,
            columns=coeffs["cols"],
            rows=coeffs["rows"],
            orientation=coeffs["orient"],
            order=coeffs["order"],
        )

    return AnchorDerivedLayout(
        hand=RoiBox(
            name="hand_origin",
            left=hand_left,
            top=hand_top,
            width=hand_w,
            height=hand_h,
        ),
        meld_origin=RoiBox(
            name="meld_origin",
            left=meld_left,
            top=meld_top,
            width=meld_w,
            height=meld_h,
        ),
        dora_origin=RoiBox(
            name="dora_origin",
            left=dora_left,
            top=dora_top,
            width=dora_w,
            height=dora_h,
        ),
        discard=discard_specs,
    )


def _px(anchor_px: int, ratio: float, dim: int) -> int:
    return anchor_px + int(round(ratio * dim))


def _dim(base_dim: int, ratio: float) -> int:
    """Dimension ratio → pixels.  Preserves sign for negative steps (top opponent)."""
    return max(1, int(round(abs(ratio) * base_dim))) * (1 if ratio >= 0 else -1)


# Measured on multi_theme fixtures (1920x1080, 12 in-match frames).
# Reference point = baseline (left_x=214, top_y=927).
# Ratios = (absolute_pixel - ref_pixel) / image_dimension.

_HAND_RX = 0.0281
_HAND_RY = -0.1389
_HAND_TILE_W = 0.036
_HAND_TILE_H = 0.112

_MELD_RX = 0.6083
_MELD_RY = -0.3185
_MELD_TILE_W = 0.042
_MELD_TILE_H = 0.10

_DORA_RX = 0.3182
_DORA_RY = -0.7583
_DORA_TILE_W = 0.034
_DORA_TILE_H = 0.09

_DISCARD_COEFFS: dict[str, dict[str, float | int | str]] = {
    "self": {
        "rx": 0.2854,
        "ry": -0.3565,
        "tile_wr": 0.0302,
        "tile_hr": 0.0648,
        "step_xr": 0.0333,
        "step_yr": 0.0648,
        "cols": 6,
        "rows": 3,
        "orient": "bottom",
        "order": "row_major",
    },
    "left_opponent": {
        "rx": 0.2135,
        "ry": -0.5898,
        "tile_wr": 0.0438,
        "tile_hr": 0.0537,
        "step_xr": 0.0427,
        "step_yr": 0.0574,
        "cols": 3,
        "rows": 6,
        "orient": "left",
        "order": "column_major",
    },
    "top_opponent": {
        "rx": 0.3063,
        "ry": -0.6343,
        "tile_wr": 0.0302,
        "tile_hr": 0.0648,
        "step_xr": 0.0333,
        "step_yr": -0.0648,
        "cols": 6,
        "rows": 3,
        "orient": "top",
        "order": "row_major",
    },
    "right_opponent": {
        "rx": 0.4865,
        "ry": -0.5898,
        "tile_wr": 0.0438,
        "tile_hr": 0.0537,
        "step_xr": 0.0427,
        "step_yr": 0.0574,
        "cols": 3,
        "rows": 6,
        "orient": "right",
        "order": "column_major",
    },
}
