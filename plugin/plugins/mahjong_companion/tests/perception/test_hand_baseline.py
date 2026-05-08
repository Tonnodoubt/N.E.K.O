"""Pin the contract for the self-hand baseline detector.

The baseline is the **primary** localisation anchor in the new
anchor-driven layout pipeline (PIPELINE.md flagged crop misalignment as
the real bottleneck, not the classifier). Any drift in detection is a
P0 regression for hand recognition accuracy.

Synthetic-image tests cover:

  * a centred ivory band on a dark background is detected and its
    bbox plus tile_height_estimate are reported correctly;
  * the baseline ``top_y`` reflects the actual top of the band, not
    the bottom of the search region;
  * detection survives small dark gaps inside the band (printed glyphs,
    tile edges) thanks to gap closing;
  * an off-centre band produces correct ``left_x`` / ``right_x``;
  * a fully dark or fully cool-coloured (blue) image returns ``None``
    — the cool gate must not let UI chrome qualify as ivory;
  * threshold overrides let callers loosen / tighten without forking.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.perception.hand_baseline import (
    HandBaselineAnchor,
    detect_hand_baseline,
)


def _canvas(width: int = 1920, height: int = 1080, *, bg: tuple[int, int, int] = (40, 30, 25)) -> np.ndarray:
    return np.full((height, width, 3), bg, dtype=np.uint8)


def _paint_band(
    arr: np.ndarray,
    *,
    top: int,
    bottom: int,
    left: int,
    right: int,
    fill: tuple[int, int, int] = (220, 200, 170),
) -> np.ndarray:
    arr[top:bottom, left:right] = fill
    return arr


@pytest.mark.unit
def test_detects_centred_ivory_band() -> None:
    arr = _paint_band(_canvas(), top=920, bottom=1040, left=210, right=1470)
    image = Image.fromarray(arr)
    anchor = detect_hand_baseline(image)
    assert isinstance(anchor, HandBaselineAnchor)
    assert anchor.image_size == image.size
    # top_y must point at the actual band top, not the search-region top.
    assert 915 <= anchor.top_y <= 925
    assert 200 <= anchor.left_x <= 220
    assert 1460 <= anchor.right_x <= 1480
    assert anchor.tile_height_estimate >= 40


@pytest.mark.unit
def test_off_centre_band_left_right_reflect_actual_extent() -> None:
    arr = _paint_band(_canvas(), top=920, bottom=1040, left=900, right=1700)
    image = Image.fromarray(arr)
    anchor = detect_hand_baseline(image)
    assert anchor is not None
    assert 890 <= anchor.left_x <= 910
    assert 1690 <= anchor.right_x <= 1710
    assert anchor.center_x == (anchor.left_x + anchor.right_x) // 2


@pytest.mark.unit
def test_small_dark_gaps_inside_band_are_closed() -> None:
    """Tile-edge shadows / glyph strokes shouldn't shrink the run."""
    arr = _paint_band(_canvas(), top=920, bottom=1040, left=210, right=1470)
    # Carve thin dark vertical gaps every ~110 px (approx tile width).
    for gap_x in range(220, 1460, 110):
        arr[920:1040, gap_x:gap_x + 4] = (30, 25, 20)
    image = Image.fromarray(arr)
    anchor = detect_hand_baseline(image)
    assert anchor is not None
    assert anchor.width >= 1200  # gap closing keeps the run mostly intact


@pytest.mark.unit
def test_to_dict_round_trip_exposes_geometry() -> None:
    arr = _paint_band(_canvas(), top=920, bottom=1040, left=210, right=1470)
    anchor = detect_hand_baseline(Image.fromarray(arr))
    assert anchor is not None
    payload = anchor.to_dict()
    assert payload["top_y"] == anchor.top_y
    assert payload["left_x"] == anchor.left_x
    assert payload["right_x"] == anchor.right_x
    assert payload["width"] == anchor.width
    assert payload["center_x"] == anchor.center_x
    assert payload["image_size"] == list(anchor.image_size)
    assert payload["tile_height_estimate"] == anchor.tile_height_estimate


@pytest.mark.unit
def test_dark_image_returns_none() -> None:
    image = Image.fromarray(_canvas(bg=(20, 18, 15)))
    assert detect_hand_baseline(image) is None


@pytest.mark.unit
def test_cool_blue_band_does_not_qualify_as_ivory() -> None:
    """The R >= B - 5 gate must reject saturated blue UI chrome even
    when its luma is high enough to pass the brightness floors."""
    arr = _paint_band(_canvas(), top=920, bottom=1040, left=210, right=1470, fill=(150, 170, 230))
    image = Image.fromarray(arr)
    assert detect_hand_baseline(image) is None


@pytest.mark.unit
def test_band_too_short_to_meet_row_ratio_returns_none() -> None:
    """An ivory smear too narrow to push a row above the threshold must
    not trip detection on a stray pixel cluster."""
    arr = _paint_band(_canvas(), top=920, bottom=1040, left=900, right=1000)
    image = Image.fromarray(arr)
    # 100/1920 ≈ 5% of width — well below default row_ratio_threshold=0.30.
    assert detect_hand_baseline(image) is None


@pytest.mark.unit
def test_threshold_override_can_loosen_row_ratio() -> None:
    """A 100-pixel ivory smear is invisible at default thresholds but
    detectable when the caller relaxes the gate."""
    arr = _paint_band(_canvas(), top=920, bottom=1040, left=900, right=1000)
    image = Image.fromarray(arr)
    anchor = detect_hand_baseline(image, row_ratio_threshold=0.04)
    assert anchor is not None
    assert anchor.left_x >= 890
    assert anchor.right_x <= 1010


@pytest.mark.unit
def test_band_only_in_top_half_returns_none() -> None:
    """The detector restricts itself to the bottom band; an ivory strip
    near the top of the screen must not be misread as the hand."""
    arr = _paint_band(_canvas(), top=80, bottom=200, left=210, right=1470)
    image = Image.fromarray(arr)
    assert detect_hand_baseline(image) is None
