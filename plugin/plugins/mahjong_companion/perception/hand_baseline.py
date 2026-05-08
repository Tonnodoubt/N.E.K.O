"""Detect the self-hand baseline as the primary localisation anchor.

The bottom row of the player's own concealed hand is a long, ivory,
horizontal strip that survives every theme, every resolution, and every
client zoom level - making it by far the most reliable visual anchor in
a Mahjong Soul screenshot. Every downstream geometry (4-player discard
ROIs, dora area, meld area) is parameterised relative to this baseline
rather than to absolute pixel offsets, so non-1920x1080 resolutions and
themed tablecloths no longer require a hand-tuned calibration profile.

Detection strategy (validated on the multi-theme fixture set):

  1. Build an "ivory" predicate: warm pixels brighter than mid-grey
     (R >= 165, G >= 150, B >= 110, R >= B - 5). Tile faces and the
     concealed-hand back-row stripe both pass; table felt and UI chrome
     do not.
  2. Restrict to the bottom band (default: lower 35% of the image) and
     find the topmost row whose ivory ratio crosses ``row_ratio_threshold``
     (default 0.30). That row's y is the baseline top.
  3. Aggregate the next ~one-tile-tall slab via per-column ``any`` so
     dark tile edges and printed glyphs don't break runs.
  4. Close gaps within ``gap_close_ratio * image_width`` columns, then
     return the longest contiguous run as ``[left_x, right_x)``.

Empirical stability on ``tests/fixtures/multi_theme`` (1920x1080,
in-match frames): ``top_y`` drops into a 20-pixel window across 13/14
screenshots; ``(left_x, right_x)`` is pixel-identical across 9 of those.
The remaining frames are non-in-match scenes (results dialog, etc.)
where the baseline genuinely does not exist - returning ``None`` is the
correct behaviour, not a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


# Thresholds tuned on tests/fixtures/multi_theme (1920x1080).
DEFAULT_BAND_TOP_RATIO = 0.65
DEFAULT_ROW_RATIO_THRESHOLD = 0.30
DEFAULT_TILE_HEIGHT_RATIO = 0.10
DEFAULT_IVORY_MIN_RED = 165
DEFAULT_IVORY_MIN_GREEN = 150
DEFAULT_IVORY_MIN_BLUE = 110
DEFAULT_GAP_CLOSE_RATIO = 0.005

# Public ivory predicate constants — imported by tile_detector and
# discard_quad_finder so thresholds stay in one place.
IVORY_MIN_RED = 165
IVORY_MIN_GREEN = 150
IVORY_MIN_BLUE = 110


@dataclass(frozen=True)
class HandBaselineAnchor:
    """Geometric anchor describing the self-hand's bottom-edge strip.

    Downstream code uses ``top_y`` to fix the hand layout's vertical
    origin, ``left_x`` / ``right_x`` to fix horizontal extent, and
    ``tile_height_estimate`` to size the scan window for related ROIs
    (dora indicators, meld tiles).
    """

    top_y: int
    left_x: int
    right_x: int
    image_size: tuple[int, int]   # (width, height)
    tile_height_estimate: int

    @property
    def width(self) -> int:
        return self.right_x - self.left_x

    @property
    def center_x(self) -> int:
        return (self.left_x + self.right_x) // 2

    def to_dict(self) -> dict[str, object]:
        return {
            "top_y": self.top_y,
            "left_x": self.left_x,
            "right_x": self.right_x,
            "width": self.width,
            "center_x": self.center_x,
            "image_size": list(self.image_size),
            "tile_height_estimate": self.tile_height_estimate,
        }


def detect_hand_baseline(
    image: Image.Image,
    *,
    band_top_ratio: float = DEFAULT_BAND_TOP_RATIO,
    row_ratio_threshold: float = DEFAULT_ROW_RATIO_THRESHOLD,
    tile_height_ratio: float = DEFAULT_TILE_HEIGHT_RATIO,
    ivory_min_red: int = DEFAULT_IVORY_MIN_RED,
    ivory_min_green: int = DEFAULT_IVORY_MIN_GREEN,
    ivory_min_blue: int = DEFAULT_IVORY_MIN_BLUE,
    gap_close_ratio: float = DEFAULT_GAP_CLOSE_RATIO,
) -> HandBaselineAnchor | None:
    """Detect the self-hand baseline strip, or ``None`` if absent.

    All thresholds are exposed as keyword arguments so callers (tests,
    diagnostic dumps, alternative themes) can tighten or relax the gates
    without forking the algorithm.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    height, width, _ = arr.shape

    ivory = _ivory_mask(
        arr,
        min_red=ivory_min_red,
        min_green=ivory_min_green,
        min_blue=ivory_min_blue,
    )

    band_top = int(height * band_top_ratio)
    band = ivory[band_top:, :]
    row_ratio = band.mean(axis=1)
    above = np.where(row_ratio >= row_ratio_threshold)[0]
    if above.size == 0:
        return None
    top_y = band_top + int(above[0])

    tile_height = max(40, int(height * tile_height_ratio))
    bottom_y = min(height, top_y + tile_height)
    column_presence = ivory[top_y:bottom_y, :].any(axis=0)

    closed = _close_horizontal_gaps(
        column_presence,
        gap_columns=max(3, int(width * gap_close_ratio)),
    )

    run = _longest_true_run(closed)
    if run is None:
        return None
    left_x, right_x = run

    return HandBaselineAnchor(
        top_y=top_y,
        left_x=left_x,
        right_x=right_x,
        image_size=(width, height),
        tile_height_estimate=tile_height,
    )


def _ivory_mask(
    arr: np.ndarray,
    *,
    min_red: int,
    min_green: int,
    min_blue: int,
) -> np.ndarray:
    """Boolean ``(H, W)`` mask of "ivory tile-face" pixels.

    The R-vs-B floor (``arr[..., 0] >= arr[..., 2] - 5``) excludes blue
    and cyan UI chrome that would otherwise satisfy the minimum
    brightness gates - tile ivory always reads warmer than cool.
    """
    return (
        (arr[..., 0] >= min_red)
        & (arr[..., 1] >= min_green)
        & (arr[..., 2] >= min_blue)
        & (arr[..., 0] >= arr[..., 2] - 5)
    )


def _close_horizontal_gaps(presence: np.ndarray, *, gap_columns: int) -> np.ndarray:
    """Treat any column within ``gap_columns`` of an ivory column as
    ivory, so dark tile edges and printed glyphs don't break runs."""
    width = presence.shape[0]
    cumsum = np.concatenate(([0], np.cumsum(presence.astype(np.int32))))
    closed = np.zeros(width, dtype=bool)
    for x in range(width):
        lo = max(0, x - gap_columns)
        hi = min(width, x + gap_columns + 1)
        closed[x] = (cumsum[hi] - cumsum[lo]) > 0
    return closed


def _longest_true_run(presence: np.ndarray) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the longest True run, or None."""
    width = int(presence.shape[0])
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x in range(width):
        if presence[x] and start is None:
            start = x
        elif not presence[x] and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, width))
    if not runs:
        return None
    return max(runs, key=lambda run: run[1] - run[0])
