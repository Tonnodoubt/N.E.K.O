"""Tests for the tile classifier dispatch layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch import (
    _apply_red_five,
    _to_template_match,
    classify_hand_tile,
    classify_tile,
    classify_tiles_batch,
    detect_red_five,
)
from plugin.plugins.mahjong_companion.perception.tile_templates import (
    DEFAULT_MAX_DISTANCE,
    TileTemplateMatch,
)


def _fake_vit_prediction(tile: str = "5m", confidence: float = 0.92, top_k: list[dict[str, Any]] | None = None):
    class _Pred:
        pass
    p = _Pred()
    p.tile = tile
    p.label = tile
    p.confidence = confidence
    p.top_k = top_k or [{"tile": tile, "score": confidence}]
    return p


def _template_payload() -> dict[str, Any]:
    return {"version": "test", "templates": {}}


@pytest.mark.unit
def test_falls_back_to_templates_when_onnx_unavailable() -> None:
    with patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_ready", return_value=False):
        crop = Image.new("RGB", (32, 32), (128, 128, 128))
        result = classify_tile(crop, _template_payload())
        # No templates loaded → returns None
        assert result is None


@pytest.mark.unit
def test_returns_onnx_result_when_available() -> None:
    fake_pred = _fake_vit_prediction("7p", 0.88, [
        {"tile": "7p", "score": 0.88},
        {"tile": "8p", "score": 0.07},
    ])
    with patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_ready", return_value=True), \
         patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_single", return_value=_to_template_match(fake_pred)):
        crop = Image.new("RGB", (32, 32), (128, 128, 128))
        result = classify_tile(crop, _template_payload())
    assert result is not None
    assert result.tile == "7p"
    assert result.confidence == pytest.approx(0.88, abs=0.01)


@pytest.mark.unit
def test_batch_empty_input() -> None:
    assert classify_tiles_batch([], _template_payload()) == []


@pytest.mark.unit
def test_batch_onnx_path() -> None:
    fake_pred = _fake_vit_prediction("1s", 0.95)
    expected = _to_template_match(fake_pred)
    with patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_ready", return_value=True), \
         patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_batch", return_value=[expected]):
        crops = [Image.new("RGB", (32, 32), (128, 128, 128))]
        results = classify_tiles_batch(crops, _template_payload())
    assert len(results) == 1
    assert results[0] is not None
    assert results[0].tile == "1s"


@pytest.mark.unit
def test_hand_classification_skips_onnx_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAHJONG_COMPANION_ONNX_HAND_ENABLED", raising=False)
    fake_pred = _fake_vit_prediction("7p", 0.88)
    crop = Image.new("RGB", (32, 32), (128, 128, 128))
    with patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_ready", return_value=True), \
         patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_single", return_value=_to_template_match(fake_pred)), \
         patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch.classify_tile_from_templates", return_value=None) as template_mock:
        result = classify_hand_tile(crop, _template_payload())
    assert result is None
    template_mock.assert_called_once()


@pytest.mark.unit
def test_hand_classification_can_opt_into_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAHJONG_COMPANION_ONNX_HAND_ENABLED", "1")
    fake_pred = _fake_vit_prediction("7p", 0.88)
    crop = Image.new("RGB", (32, 32), (128, 128, 128))
    with patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_ready", return_value=True), \
         patch("plugin.plugins.mahjong_companion.perception.tile_classifier_dispatch._onnx_single", return_value=_to_template_match(fake_pred)):
        result = classify_hand_tile(crop, _template_payload())
    assert result is not None
    assert result.tile == "7p"


@pytest.mark.unit
def test_to_template_match_distance_mapping() -> None:
    pred = _fake_vit_prediction("3m", 0.8)
    match = _to_template_match(pred)
    assert match.distance == pytest.approx((1.0 - 0.8) * DEFAULT_MAX_DISTANCE, abs=0.01)
    assert match.confidence == 0.8


@pytest.mark.unit
def test_to_template_match_runner_up() -> None:
    pred = _fake_vit_prediction("2z", 0.75, [
        {"tile": "2z", "score": 0.75},
        {"tile": "3z", "score": 0.15},
    ])
    match = _to_template_match(pred)
    assert match.runner_up_tile == "3z"
    assert match.runner_up_distance == pytest.approx((1.0 - 0.15) * DEFAULT_MAX_DISTANCE, abs=0.01)


@pytest.mark.unit
def test_to_template_match_no_runner_up() -> None:
    pred = _fake_vit_prediction("9m", 0.99, [{"tile": "9m", "score": 0.99}])
    match = _to_template_match(pred)
    assert match.runner_up_tile == ""
    assert match.runner_up_distance is None


# ---------------------------------------------------------------------------
# Red-five color detection
# ---------------------------------------------------------------------------


def _make_tile_crop(center_r: int, center_g: int, center_b: int) -> Image.Image:
    """32x32 tile with ivory border and a coloured centre patch."""
    arr = np.full((32, 32, 3), (200, 175, 140), dtype=np.uint8)
    # Centre 12x12 patch with the test colour.
    arr[10:22, 10:22, 0] = center_r
    arr[10:22, 10:22, 1] = center_g
    arr[10:22, 10:22, 2] = center_b
    return Image.fromarray(arr)


def test_red_five_detects_red_5p():
    crop = _make_tile_crop(180, 80, 70)
    assert detect_red_five(crop, "5p") == "0p"


def test_red_five_ignores_non_red_5p():
    crop = _make_tile_crop(60, 55, 50)
    assert detect_red_five(crop, "5p") is None


def test_red_five_ignores_non_five_tile():
    crop = _make_tile_crop(180, 80, 70)
    assert detect_red_five(crop, "1m") is None
    assert detect_red_five(crop, "7p") is None
    assert detect_red_five(crop, "5z") is None


def test_red_five_detects_5m():
    crop = _make_tile_crop(175, 90, 85)
    assert detect_red_five(crop, "5m") == "0m"


def test_red_five_detects_5s():
    crop = _make_tile_crop(170, 85, 75)
    assert detect_red_five(crop, "5s") == "0s"


def test_red_five_dark_red_does_not_trigger():
    crop = _make_tile_crop(100, 70, 65)
    assert detect_red_five(crop, "5p") is None


def test_apply_red_five_rewrites_tile():
    crop = _make_tile_crop(180, 80, 70)
    match = TileTemplateMatch(tile="5p", confidence=0.9, distance=5.0)
    result = _apply_red_five(crop, match)
    assert result.tile == "0p"
    assert result.confidence == 0.9


def test_apply_red_five_passes_through_non_red():
    crop = _make_tile_crop(60, 55, 50)
    match = TileTemplateMatch(tile="5p", confidence=0.9, distance=5.0)
    result = _apply_red_five(crop, match)
    assert result.tile == "5p"
