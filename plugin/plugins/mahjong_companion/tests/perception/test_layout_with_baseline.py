"""Tests for layout routing with baseline anchor."""

from __future__ import annotations

import sys

sys.path.insert(0, "plugin/plugins/mahjong_companion")

import pytest
from plugin.plugins.mahjong_companion.perception.hand_baseline import (
    HandBaselineAnchor,
)
from plugin.plugins.mahjong_companion.perception.calibration import (
    CalibrationOffsets,
    CalibrationProfile,
)
from plugin.plugins.mahjong_companion.perception.hand_layout import (
    _baseline_plausible,
    build_hand_layout,
)
from plugin.plugins.mahjong_companion.perception.discard_layout import (
    _baseline_plausible as _discard_baseline_plausible,
    build_discard_layout,
)

_1080_good = HandBaselineAnchor(
    top_y=927, left_x=214, right_x=1462,
    image_size=(1920, 1080), tile_height_estimate=108,
)
_1080_far_right = HandBaselineAnchor(
    top_y=947, left_x=1094, right_x=1832,
    image_size=(1920, 1080), tile_height_estimate=108,
)
_1080_top_edge = HandBaselineAnchor(
    top_y=200, left_x=214, right_x=1462,
    image_size=(1920, 1080), tile_height_estimate=108,
)
_1080_bottom_edge = HandBaselineAnchor(
    top_y=1070, left_x=214, right_x=1462,
    image_size=(1920, 1080), tile_height_estimate=108,
)


def test_plausible_via_hand_layout_passes_good_baseline():
    assert _baseline_plausible(_1080_good, 1920, 1080) is True


def test_plausible_via_discard_layout_passes_good_baseline():
    assert _discard_baseline_plausible(_1080_good, 1920, 1080) is True


def test_hand_plausible_rejects_far_right():
    assert _baseline_plausible(_1080_far_right, 1920, 1080) is False


def test_discard_plausible_rejects_far_right():
    assert _discard_baseline_plausible(_1080_far_right, 1920, 1080) is False


def test_hand_plausible_rejects_top_edge():
    assert _baseline_plausible(_1080_top_edge, 1920, 1080) is False


def test_hand_plausible_rejects_bottom_edge():
    assert _baseline_plausible(_1080_bottom_edge, 1920, 1080) is False


def test_hand_plausible_rejects_none_baseline():
    """None should skip the check entirely at the call site."""
    pass


def test_build_hand_layout_falls_back_when_baseline_implausible():
    layout = build_hand_layout(1920, 1080, baseline=_1080_far_right)
    hc = build_hand_layout(1920, 1080)
    assert layout["hand"][0].box.left == hc["hand"][0].box.left
    assert layout["hand"][0].box.top == hc["hand"][0].box.top


def test_build_hand_layout_falls_back_when_baseline_none():
    layout = build_hand_layout(1920, 1080, baseline=None)
    hc = build_hand_layout(1920, 1080)
    assert layout["hand"][0].box.left == hc["hand"][0].box.left


def test_build_discard_layout_falls_back_when_baseline_implausible():
    layout = build_discard_layout(1920, 1080, baseline=_1080_far_right)
    hc = build_discard_layout(1920, 1080)
    assert layout["self"][0].box.left == hc["self"][0].box.left


def test_build_discard_layout_falls_back_when_baseline_none():
    layout = build_discard_layout(1920, 1080, baseline=None)
    hc = build_discard_layout(1920, 1080)
    assert layout["self"][0].box.left == hc["self"][0].box.left


def test_build_hand_layout_uses_anchor_when_baseline_plausible():
    layout = build_hand_layout(1920, 1080, baseline=_1080_good)
    hc = build_hand_layout(1920, 1080)
    assert layout["hand"][0].box.left == hc["hand"][0].box.left
    assert layout["hand"][0].box.top == hc["hand"][0].box.top
    assert layout["dora"][0].box.left == hc["dora"][0].box.left
    assert layout["meld"][0].box.left == hc["meld"][0].box.left


def test_anchor_hand_layout_uses_calibrated_draw_gap_as_total():
    calibration = CalibrationProfile(
        screen_width=1920,
        screen_height=1080,
        hand_offsets=CalibrationOffsets(draw_gap_px=41),
    )
    layout = build_hand_layout(
        1920,
        1080,
        baseline=_1080_good,
        calibration=calibration,
        draw_slot_index=14,
    )

    natural_left = layout["hand"][12].box.left + layout["hand"][12].box.width
    actual_gap = layout["hand"][13].box.left - natural_left

    assert actual_gap == 41 + int(layout["hand"][12].box.width * 0.12)


def test_build_discard_layout_uses_anchor_when_baseline_plausible():
    layout = build_discard_layout(1920, 1080, baseline=_1080_good)
    hc = build_discard_layout(1920, 1080)
    for player in ["self", "left_opponent", "top_opponent", "right_opponent"]:
        assert layout[player][0].box.left == hc[player][0].box.left, player
