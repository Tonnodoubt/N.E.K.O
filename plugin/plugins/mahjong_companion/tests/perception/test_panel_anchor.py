"""Pin the contract for ``detect_score_panel`` / ``detect_score_panel_anchor``.

The score panel is the geometric anchor for the upcoming
anchor-driven localisation rewrite, so any drift in its detection
behaviour is a P0 regression. These tests cover:

  * a centred dark rectangle of plausible size + aspect is detected;
  * the high-level wrapper exposes a usable ``ScorePanelAnchor``;
  * size, aspect, and fill-ratio gates each reject pathological inputs;
  * an empty (all-white) image returns ``None`` cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.perception.panel_anchor import (
    ScorePanelAnchor,
    detect_score_panel,
    detect_score_panel_anchor,
)


def _canvas(width: int = 800, height: int = 600, *, bg: int = 200) -> Image.Image:
    arr = np.full((height, width, 3), bg, dtype=np.uint8)
    return Image.fromarray(arr)


def _paint_rect(
    image: Image.Image,
    *,
    bbox: tuple[int, int, int, int],
    fill: tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    arr = np.array(image)
    left, top, right, bottom = bbox
    arr[top:bottom, left:right] = fill
    return Image.fromarray(arr)


@pytest.mark.unit
def test_detects_centred_dark_panel() -> None:
    image = _paint_rect(_canvas(), bbox=(280, 230, 520, 290))
    bbox = detect_score_panel(image)
    assert bbox is not None
    left, top, right, bottom = bbox
    assert 270 <= left <= 290
    assert 220 <= top <= 240
    assert 510 <= right <= 530
    assert 280 <= bottom <= 300


@pytest.mark.unit
def test_high_level_wrapper_returns_anchor_with_geometry() -> None:
    image = _paint_rect(_canvas(), bbox=(280, 230, 520, 290))
    anchor = detect_score_panel_anchor(image)
    assert isinstance(anchor, ScorePanelAnchor)
    assert anchor.image_size == image.size
    cx, cy = anchor.center
    assert 390 <= cx <= 410
    assert 250 <= cy <= 270
    assert anchor.width >= 200
    assert anchor.height >= 50


@pytest.mark.unit
def test_to_dict_serialisation_round_trip() -> None:
    image = _paint_rect(_canvas(), bbox=(280, 230, 520, 290))
    anchor = detect_score_panel_anchor(image)
    assert anchor is not None
    payload = anchor.to_dict()
    assert payload["bbox"] == list(anchor.bbox)
    assert payload["center"] == list(anchor.center)
    assert payload["image_size"] == list(anchor.image_size)
    assert payload["width"] == anchor.width
    assert payload["height"] == anchor.height


@pytest.mark.unit
def test_too_small_dark_blob_rejected_by_min_area() -> None:
    """A 10×4 dark speck is below ``min_area_ratio`` and must not be
    promoted to "the score panel" just because it's centred."""
    image = _paint_rect(_canvas(), bbox=(395, 298, 405, 302))
    assert detect_score_panel(image) is None


@pytest.mark.unit
def test_too_large_dark_blob_rejected_by_max_area() -> None:
    """A panel that fills 20% of the image is implausible — score panel
    is small/medium UI furniture."""
    image = _paint_rect(_canvas(), bbox=(80, 60, 720, 540))
    assert detect_score_panel(image) is None


@pytest.mark.unit
def test_tall_dark_blob_rejected_by_aspect_ratio() -> None:
    """A 60×240 vertical strip has aspect ~0.25, well below 1.5."""
    image = _paint_rect(_canvas(), bbox=(370, 180, 430, 420))
    assert detect_score_panel(image) is None


@pytest.mark.unit
def test_low_fill_ratio_rejected() -> None:
    """A hollow rectangle (only the border is dark) should fail the fill
    test even if the bounding box is the right size."""
    image = _canvas()
    arr = np.array(image)
    # Draw only the 2-pixel border of the would-be panel.
    arr[230:232, 280:520] = (20, 20, 20)
    arr[288:290, 280:520] = (20, 20, 20)
    arr[230:290, 280:282] = (20, 20, 20)
    arr[230:290, 518:520] = (20, 20, 20)
    image = Image.fromarray(arr)
    assert detect_score_panel(image) is None


@pytest.mark.unit
def test_empty_bright_image_returns_none() -> None:
    image = _canvas(bg=240)
    assert detect_score_panel(image) is None
    assert detect_score_panel_anchor(image) is None


@pytest.mark.unit
def test_threshold_overrides_let_callers_loosen_dark_gate() -> None:
    """A medium-grey panel (luma ~110) is invisible at the default
    threshold but should be detectable when the caller relaxes the gate."""
    image = _paint_rect(_canvas(), bbox=(280, 230, 520, 290), fill=(110, 110, 110))
    assert detect_score_panel(image) is None
    bbox = detect_score_panel(image, dark_luma_threshold=140)
    assert bbox is not None
