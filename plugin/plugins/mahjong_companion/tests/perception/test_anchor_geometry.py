"""Tests for anchor_geometry module."""

from __future__ import annotations

import sys

sys.path.insert(0, "plugin/plugins/mahjong_companion")

import pytest
from plugin.plugins.mahjong_companion.perception.anchor_geometry import (
    anchor_derived_rois,
)
from plugin.plugins.mahjong_companion.perception.hand_baseline import (
    HandBaselineAnchor,
)

_1080 = HandBaselineAnchor(
    top_y=927, left_x=214, right_x=1462,
    image_size=(1920, 1080), tile_height_estimate=108,
)


def test_hand_origin_matches_hardcoded():
    layout = anchor_derived_rois(_1080, 1920, 1080)
    hand = layout.hand
    expected_left = int(1920 * 0.14)
    expected_top = int(1080 * 0.72)
    assert hand.left == expected_left
    assert hand.top == expected_top


def test_discard_self_origin_matches_hardcoded():
    layout = anchor_derived_rois(_1080, 1920, 1080)
    self_spec = layout.discard["self"]
    assert self_spec.origin_left == 762
    assert self_spec.origin_top == 542


def test_discard_all_players_origin():
    layout = anchor_derived_rois(_1080, 1920, 1080)
    expected = {
        "self": (762, 542),
        "left_opponent": (624, 290),
        "top_opponent": (802, 242),
        "right_opponent": (1148, 290),
    }
    for player, (ex, ey) in expected.items():
        spec = layout.discard[player]
        assert (spec.origin_left, spec.origin_top) == (ex, ey), player


def test_discard_tile_dims():
    layout = anchor_derived_rois(_1080, 1920, 1080)
    self_spec = layout.discard["self"]
    assert self_spec.tile_width == 58
    assert self_spec.tile_height == 70
    assert self_spec.step_x == 64
    assert self_spec.step_y == 70
    assert self_spec.columns == 6
    assert self_spec.rows == 3


def test_meld_and_dora_origins():
    layout = anchor_derived_rois(_1080, 1920, 1080)
    expected_meld_left = int(1920 * 0.72)
    expected_meld_top = int(1080 * 0.54)
    assert layout.meld_origin.left == expected_meld_left
    assert layout.meld_origin.top == expected_meld_top

    expected_dora_left = int(1920 * 0.43)
    expected_dora_top = int(1080 * 0.10)
    assert layout.dora_origin.left == expected_dora_left
    assert layout.dora_origin.top == expected_dora_top


def test_scaling_to_different_resolution():
    half = HandBaselineAnchor(
        top_y=463, left_x=107, right_x=731,
        image_size=(960, 540), tile_height_estimate=54,
    )
    layout_half = anchor_derived_rois(half, 960, 540)
    layout_full = anchor_derived_rois(_1080, 1920, 1080)

    for player in ["self", "left_opponent", "top_opponent", "right_opponent"]:
        full = layout_full.discard[player]
        half_spec = layout_half.discard[player]
        assert abs(half_spec.origin_left - full.origin_left // 2) <= 1, player
        assert abs(half_spec.origin_top - full.origin_top // 2) <= 1, player


def test_discard_order_field():
    layout = anchor_derived_rois(_1080, 1920, 1080)
    assert layout.discard["self"].order == "row_major"
    assert layout.discard["left_opponent"].order == "column_major"
    assert layout.discard["right_opponent"].order == "column_major"
