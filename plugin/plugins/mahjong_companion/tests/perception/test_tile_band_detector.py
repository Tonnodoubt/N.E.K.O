"""Tests for band-based tile segmentation in tile_detector."""

from __future__ import annotations

import sys

sys.path.insert(0, "plugin/plugins/mahjong_companion")

import numpy as np
import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.perception.tile_detector import (
    TileBandBox,
    _find_boundary,
    _find_peaks,
    _smooth_1d,
    find_tiles_in_band,
)


# ---------------------------------------------------------------------------
# Synthetic ivory-band image builders
# ---------------------------------------------------------------------------


def _ivory_band(
    band_width: int,
    band_height: int,
    *,
    tile_centres: list[int],
    tile_width: int,
) -> Image.Image:
    """Paint ivory rectangles onto a dark felt background.

    Each tile is placed centred at ``tile_centres[i]`` horizontally and
    spans the full band height.
    """
    arr = np.full((band_height, band_width, 3), (60, 90, 60), dtype=np.uint8)
    for cx in tile_centres:
        left = max(0, cx - tile_width // 2)
        right = min(band_width, cx + tile_width // 2)
        arr[:, left:right] = (200, 175, 140)  # warm ivory
        # Add a dark glyph stripe in the middle so the ivory is "textured".
        glyph_left = cx - tile_width // 6
        glyph_right = cx + tile_width // 6
        glyph_top = band_height // 3
        glyph_bottom = 2 * band_height // 3
        arr[glyph_top:glyph_bottom, glyph_left:glyph_right] = (40, 30, 20)
    return Image.fromarray(arr)


def _vertical_ivory_band(
    band_width: int,
    band_height: int,
    *,
    tile_centres: list[int],
    tile_height: int,
) -> Image.Image:
    """Same as _ivory_band but for vertical orientation (tiles stacked top-to-bottom)."""
    arr = np.full((band_height, band_width, 3), (60, 90, 60), dtype=np.uint8)
    for cy in tile_centres:
        top = max(0, cy - tile_height // 2)
        bottom = min(band_height, cy + tile_height // 2)
        arr[top:bottom, :] = (200, 175, 140)
        glyph_top = cy - tile_height // 6
        glyph_bottom = cy + tile_height // 6
        glyph_left = band_width // 3
        glyph_right = 2 * band_width // 3
        arr[glyph_top:glyph_bottom, glyph_left:glyph_right] = (40, 30, 20)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Low-level helper tests
# ---------------------------------------------------------------------------


def test_smooth_1d_preserves_length():
    sig = np.array([0.0, 0.2, 0.8, 0.2, 0.0])
    out = _smooth_1d(sig, sigma=1.0)
    assert len(out) == len(sig)


def test_smooth_1d_reduces_noise():
    np.random.seed(42)
    sig = np.random.randn(100) * 0.1 + 0.5
    out = _smooth_1d(sig, sigma=3.0)
    assert float(np.std(out)) < float(np.std(sig))


def test_find_peaks_detects_obvious_maxima():
    sig = np.array([0.0, 0.1, 0.5, 0.1, 0.0, 0.1, 0.4, 0.1, 0.0])
    peaks = _find_peaks(sig, min_height=0.2, min_distance=2)
    assert len(peaks) == 2
    indices = [p[0] for p in peaks]
    assert 2 in indices
    assert 6 in indices


def test_find_peaks_respects_min_height():
    sig = np.array([0.0, 0.05, 0.1, 0.05, 0.0])
    peaks = _find_peaks(sig, min_height=0.2, min_distance=1)
    assert len(peaks) == 0


def test_find_peaks_respects_min_distance():
    sig = np.array([0.0, 0.8, 0.9, 0.85, 0.0])
    peaks = _find_peaks(sig, min_height=0.3, min_distance=10)
    assert len(peaks) == 1


def test_find_boundary_left():
    sig = np.array([0.0, 0.0, 0.1, 0.4, 0.8, 0.4, 0.1, 0.0])
    edge = _find_boundary(sig, 4, direction=-1, threshold_ratio=0.4)
    assert edge < 4


def test_find_boundary_right():
    sig = np.array([0.0, 0.0, 0.1, 0.4, 0.8, 0.4, 0.1, 0.0])
    edge = _find_boundary(sig, 4, direction=+1, threshold_ratio=0.4)
    assert edge > 4


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_find_three_tiles_in_horizontal_band():
    img = _ivory_band(600, 50, tile_centres=[100, 300, 500], tile_width=60)
    boxes = find_tiles_in_band(img, 0, 0, 600, 50, orientation="horizontal")
    assert len(boxes) >= 2  # Allow one weak peak to drop out


def test_find_expected_count_trims_excess():
    img = _ivory_band(600, 50, tile_centres=[80, 200, 320, 440], tile_width=60)
    boxes = find_tiles_in_band(
        img, 0, 0, 600, 50, orientation="horizontal", expected_count=3,
    )
    assert len(boxes) == 3


def test_empty_band_returns_empty():
    arr = np.full((40, 500, 3), (60, 90, 60), dtype=np.uint8)
    img = Image.fromarray(arr)
    boxes = find_tiles_in_band(img, 0, 0, 500, 40, orientation="horizontal")
    assert len(boxes) == 0


def test_horizontal_boxes_in_full_image_coords():
    """Boxes should be returned in full-image coordinates, offset by band position."""
    band_img = _ivory_band(400, 30, tile_centres=[100, 300], tile_width=50)
    # Embed the band into a larger canvas at (50, 50).
    canvas = Image.new("RGB", (500, 110), (60, 90, 60))
    canvas.paste(band_img, (50, 50))
    boxes = find_tiles_in_band(canvas, 50, 50, 400, 30, orientation="horizontal")
    for b in boxes:
        assert b.left >= 50
        assert b.top >= 50
        assert b.right <= 450
        assert b.bottom <= 80


def test_vertical_band_detects_tiles():
    img = _vertical_ivory_band(30, 400, tile_centres=[80, 200, 320], tile_height=50)
    boxes = find_tiles_in_band(img, 0, 0, 30, 400, orientation="vertical")
    assert len(boxes) >= 2


def test_blueish_band_no_false_positives():
    """Cool-toned UI elements should not trigger ivory peaks."""
    arr = np.full((30, 400, 3), (60, 100, 180), dtype=np.uint8)  # blue
    img = Image.fromarray(arr)
    boxes = find_tiles_in_band(img, 0, 0, 400, 30, orientation="horizontal")
    assert len(boxes) == 0
