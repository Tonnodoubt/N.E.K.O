"""Verify ``parse_discards_from_image`` issues batched classifier calls.

Before the refactor, every occupied slot triggered one ``classify_tile``
call (plus an optional second call for the refined crop), giving
``O(occupied + refined)`` ONNX forwards per frame. The batched version
issues at most two calls:

1. one batched call across all occupied base crops, and
2. one batched call across plans whose base match was below the
   minimum confidence and that have a quad refinement available.

These tests pin that contract while leaving the per-slot acceptance /
rejection logic untouched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.perception.discard_layout import DiscardSlot
from plugin.plugins.mahjong_companion.perception.discard_parser import parse_discards_from_image
from plugin.plugins.mahjong_companion.perception.discard_quad_finder import DiscardQuadRefinement
from plugin.plugins.mahjong_companion.perception.roi import RoiBox
from plugin.plugins.mahjong_companion.perception.tile_templates import (
    DEFAULT_MAX_DISTANCE,
    TileTemplateMatch,
)


_OCCUPIED_FILL = (210, 200, 180)


def _make_slot(slot_id: str, left: int, *, turn_index: int) -> DiscardSlot:
    return DiscardSlot(
        slot_id=slot_id,
        player="self",
        turn_index=turn_index,
        orientation="bottom",
        box=RoiBox(name="discard_slot", left=left, top=10, width=40, height=60),
    )


def _occupied_image() -> Image.Image:
    """Bright + high-variance image so the slot occupancy gate fires.

    The gate requires ``mean_luma >= 88``, ``bright_ratio >= 0.12 OR
    white_ratio >= 0.04``, ``dark_ratio <= 0.62`` and ``stddev >= 14``.
    Horizontal 3-row stripes at periods that don't align with the metric
    sampler's ``sample_step=4``, so slot occupancy is position-invariant.
    """
    image = Image.new("RGB", (200, 120), _OCCUPIED_FILL)
    for y in range(0, 120):
        if y % 3 == 0:
            row_color = (60, 60, 60)
        elif y % 3 == 1:
            row_color = (255, 255, 255)
        else:
            row_color = (180, 180, 180)
        for x in range(0, 200):
            image.putpixel((x, y), row_color)
    return image


def _make_match(tile: str, confidence: float, runner_up: str = "") -> TileTemplateMatch:
    distance = (1.0 - confidence) * DEFAULT_MAX_DISTANCE
    return TileTemplateMatch(
        tile=tile,
        confidence=confidence,
        distance=distance,
        runner_up_tile=runner_up,
        runner_up_distance=(1.0 - 0.10) * DEFAULT_MAX_DISTANCE if runner_up else None,
    )


def _template_payload() -> dict[str, Any]:
    # Non-empty payload so parse_discards_from_image gets past the early
    # missing-template short-circuit. Real signatures are mocked away.
    return {"version": "test", "templates": {"placeholder": object()}}


def _make_refinement(left: int) -> DiscardQuadRefinement:
    return DiscardQuadRefinement(
        quad=(
            (left, 12),
            (left, 68),
            (left + 36, 68),
            (left + 36, 12),
        ),
        bbox=[left, 12, left + 36, 68],
        search_box=RoiBox(name="search", left=left - 4, top=8, width=44, height=64),
        confidence=0.84,
        component_area=400,
    )


@pytest.mark.unit
def test_strong_base_runs_single_batch_no_refined() -> None:
    """Two slots, both with strong base matches and no refinement → 1 batch."""
    image = _occupied_image()
    layout = {"self": [_make_slot("d-self-0", 10, turn_index=0), _make_slot("d-self-1", 80, turn_index=1)]}
    base_matches = [_make_match("5p", 0.92), _make_match("3m", 0.88)]
    batch_calls: list[int] = []

    def fake_batch(crops, payload):  # noqa: ANN001 - test stub
        batch_calls.append(len(crops))
        return list(base_matches)

    with patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.refine_discard_slot_quad",
        return_value=None,
    ), patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.classify_tiles_batch",
        side_effect=fake_batch,
    ), patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser._onnx_backend_active",
        return_value=False,
    ):
        result = parse_discards_from_image(image, _template_payload(), layout=layout)

    assert batch_calls == [2], f"expected one batch of 2; got {batch_calls}"
    self_pile = result.discard_piles.get("self", [])
    accepted_tiles = [item["tile"] for item in self_pile]
    assert accepted_tiles == ["5p", "3m"]


@pytest.mark.unit
def test_weak_base_triggers_refined_batch_only_for_weak_slots() -> None:
    """Two slots: one strong (skip refined), one weak with refinement (refined batch of size 1)."""
    image = _occupied_image()
    slot_strong = _make_slot("d-self-0", 10, turn_index=0)
    slot_weak = _make_slot("d-self-1", 80, turn_index=1)
    layout = {"self": [slot_strong, slot_weak]}

    base_matches = [_make_match("5p", 0.92), _make_match("6p", 0.40)]
    refined_match = _make_match("6p", 0.81)
    batch_call_sizes: list[int] = []

    def fake_batch(crops, payload):  # noqa: ANN001 - test stub
        batch_call_sizes.append(len(crops))
        if len(crops) == 2:
            return list(base_matches)
        return [refined_match]

    def fake_refine(_image, slot):  # noqa: ANN001 - test stub
        if slot.slot_id == "d-self-1":
            return _make_refinement(slot.box.left)
        return None

    with patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.refine_discard_slot_quad",
        side_effect=fake_refine,
    ), patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.classify_tiles_batch",
        side_effect=fake_batch,
    ), patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser._onnx_backend_active",
        return_value=False,
    ):
        result = parse_discards_from_image(image, _template_payload(), layout=layout)

    assert batch_call_sizes == [2, 1], f"expected base batch=2 then refined batch=1; got {batch_call_sizes}"
    self_pile = result.discard_piles.get("self", [])
    accepted_tiles = [item["tile"] for item in self_pile]
    assert accepted_tiles == ["5p", "6p"]
    weak_item = self_pile[1]
    assert weak_item["quad_source"] == "refined_tile_surface"
    assert weak_item["confidence"] == pytest.approx(0.81)


@pytest.mark.unit
def test_onnx_occupancy_gate_rejects_low_confidence_match() -> None:
    image = _occupied_image()
    layout = {"self": [_make_slot("d-self-0", 10, turn_index=0)]}
    base_matches = [_make_match("5p", 0.89)]

    with patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.refine_discard_slot_quad",
        return_value=None,
    ), patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.classify_tiles_batch",
        return_value=base_matches,
    ), patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser._onnx_backend_active",
        return_value=True,
    ):
        result = parse_discards_from_image(image, _template_payload(), layout=layout)

    assert result.discard_piles == {}
    assert result.raw_detections[0]["rejection_reason"] == "onnx_occupancy_gate"


@pytest.mark.unit
def test_no_occupied_slots_skips_classification() -> None:
    """All slots empty → zero batch calls."""
    image = Image.new("RGB", (200, 120), (0, 0, 0))  # dark image fails occupancy gate
    layout = {"self": [_make_slot("d-self-0", 10, turn_index=0)]}
    batch_calls: list[int] = []

    def fake_batch(crops, payload):  # noqa: ANN001 - test stub
        batch_calls.append(len(crops))
        return []

    with patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.refine_discard_slot_quad",
        return_value=None,
    ), patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.classify_tiles_batch",
        side_effect=fake_batch,
    ):
        result = parse_discards_from_image(image, _template_payload(), layout=layout)

    assert batch_calls == []
    assert result.discard_piles == {}


@pytest.mark.unit
def test_empty_template_payload_short_circuits() -> None:
    """Missing template payload returns immediately, no batch calls."""
    image = _occupied_image()
    batch_calls: list[int] = []

    def fake_batch(crops, payload):  # noqa: ANN001 - test stub
        batch_calls.append(len(crops))
        return []

    with patch(
        "plugin.plugins.mahjong_companion.perception.discard_parser.classify_tiles_batch",
        side_effect=fake_batch,
    ):
        result = parse_discards_from_image(image, {})

    assert batch_calls == []
    assert result.discard_piles == {}
    assert result.analysis_hints["discard_parser_available"] is False
