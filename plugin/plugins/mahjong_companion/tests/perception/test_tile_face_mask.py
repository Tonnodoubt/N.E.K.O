"""Pin the tile-face mask behaviour on the patches that matter:

  * Ivory tile faces (low saturation, high luma) survive — the standard
    Otsu + saturation-gate path.
  * Red-five centers (saturated red-dominant pixels) **also** survive,
    so quad refinement no longer skips around the red-dora area.
  * Generic high-saturation non-red colours (green felt, blue UI chrome,
    yellow riichi sticks) are still rejected — the new red-dominant
    path is a narrow opening, not a free pass for any saturated colour.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.perception.discard_quad_finder import _tile_face_mask


def _solid(rgb: tuple[int, int, int], size: int = 24) -> Image.Image:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    return Image.fromarray(arr)


def _two_tone(
    bright: tuple[int, int, int],
    dark: tuple[int, int, int],
    *,
    size: int = 32,
    bright_fraction: float = 0.55,
) -> Image.Image:
    """Otsu picks a meaningful threshold only when there are two distinct
    populations, so every test image carries a dark band."""
    arr = np.full((size, size, 3), dark, dtype=np.uint8)
    cut = int(size * bright_fraction)
    arr[:cut, :, :] = bright
    return Image.fromarray(arr)


@pytest.mark.unit
def test_ivory_tile_face_survives_mask() -> None:
    crop = _two_tone((215, 205, 185), (40, 40, 40))
    mask = _tile_face_mask(crop)
    # ivory band ~55% of pixels; should mostly pass.
    assert mask[:10, :].mean() > 0.85


@pytest.mark.unit
def test_red_five_center_survives_via_red_dominant_path() -> None:
    """Red-dora hot center used to be rejected by ``saturation <= 100``."""
    crop = _two_tone((210, 50, 50), (30, 30, 30))
    mask = _tile_face_mask(crop)
    # The red band must mostly pass — that's the whole point of the patch.
    assert mask[:10, :].mean() > 0.85


@pytest.mark.unit
def test_saturated_green_does_not_leak_through() -> None:
    """The new red-dominant path must NOT open the door for table felt
    or other saturated greens (which are common majsoul backgrounds)."""
    crop = _two_tone((30, 180, 60), (30, 30, 30))
    mask = _tile_face_mask(crop)
    # Bright green region should be rejected by both paths.
    assert mask[:10, :].mean() < 0.15


@pytest.mark.unit
def test_saturated_blue_does_not_leak_through() -> None:
    crop = _two_tone((40, 60, 200), (30, 30, 30))
    mask = _tile_face_mask(crop)
    assert mask[:10, :].mean() < 0.15


@pytest.mark.unit
def test_yellow_riichi_stick_does_not_qualify_as_red() -> None:
    """Riichi sticks are saturated yellow (high R+G, low B) — must not be
    misread as red-dominant which only accepts R>>G AND R>>B."""
    crop = _two_tone((220, 200, 30), (30, 30, 30))
    mask = _tile_face_mask(crop)
    # Yellow has high luma but high green too → saturation > 100 AND
    # not red-dominant (R - G is small) → rejected.
    assert mask[:10, :].mean() < 0.15


@pytest.mark.unit
def test_dark_red_below_brightness_floor_does_not_qualify_as_red_dominant() -> None:
    """The red-dominant path requires ``red >= 150`` on top of the
    R-vs-{G,B} margin. A muddy dark red (e.g. wood texture, shadow on
    a red logo) must not slip through just because it's red-skewed."""
    crop = _two_tone((140, 30, 30), (30, 30, 30))  # dark muddy red + black
    mask = _tile_face_mask(crop)
    # Bright band's red=140 < 150 → red-dominant fails; saturation=110 > 100
    # → the low-sat gate also fails. Whole image rejected.
    assert mask.mean() < 0.15
