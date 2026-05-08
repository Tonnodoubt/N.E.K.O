"""Integration test for the discard-pile temporal smoothing applier.

Pins the contract that ``apply_temporal_smoothing_to_discard_piles``:

  * keys slots by ``f"{player}:{turn_index}"`` (turn_index is the stable
    cross-frame anchor for the discard pile);
  * leaves consistent predictions untouched but adds counters;
  * overwrites a single-frame mistake when the smoothed estimate is more
    confident, recording the original tile + confidence for diagnostics;
  * never overwrites with a less-confident estimate.
"""

from __future__ import annotations

import pytest

from plugin.plugins.mahjong_companion.perception.temporal_tracker import (
    TemporalTileTracker,
    apply_temporal_smoothing_to_discard_piles,
)


def _pile(tile: str, *, turn_index: int, confidence: float) -> dict:
    return {
        "tile": tile,
        "turn_index": turn_index,
        "confidence": confidence,
        "player": "self",
    }


@pytest.mark.unit
def test_consistent_frames_leave_tile_unchanged_and_count_stable() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    discard_piles = {"self": [_pile("5p", turn_index=1, confidence=0.9)]}

    apply_temporal_smoothing_to_discard_piles(discard_piles, tracker)
    counters = apply_temporal_smoothing_to_discard_piles(discard_piles, tracker)

    assert discard_piles["self"][0]["tile"] == "5p"
    assert "temporal_smoothed" not in discard_piles["self"][0]
    assert counters["stable"] >= 1


@pytest.mark.unit
def test_one_noisy_frame_gets_overwritten_by_smoothed_majority() -> None:
    """Three good frames, then a wrong one with lower confidence — the
    tracker should restore the consistent tile and tag the override."""
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    for _ in range(3):
        apply_temporal_smoothing_to_discard_piles(
            {"self": [_pile("5p", turn_index=1, confidence=0.9)]},
            tracker,
        )
    bad = {"self": [_pile("6p", turn_index=1, confidence=0.55)]}
    counters = apply_temporal_smoothing_to_discard_piles(bad, tracker)

    item = bad["self"][0]
    assert item["tile"] == "5p"
    assert item["temporal_smoothed"] is True
    assert item["temporal_original_tile"] == "6p"
    # EMA score lives on a different scale than raw frame confidence — we
    # gate on support, not on score magnitude.
    assert item["temporal_support"] >= 2
    assert counters["smoothed"] == 1


@pytest.mark.unit
def test_high_confidence_single_frame_not_overwritten_by_weak_history() -> None:
    """A genuinely-changing high-confidence frame should not be reverted
    to a weakly-supported historic estimate."""
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    apply_temporal_smoothing_to_discard_piles(
        {"self": [_pile("5p", turn_index=1, confidence=0.30)]},
        tracker,
    )
    apply_temporal_smoothing_to_discard_piles(
        {"self": [_pile("5p", turn_index=1, confidence=0.30)]},
        tracker,
    )
    fresh = {"self": [_pile("6p", turn_index=1, confidence=0.99)]}
    apply_temporal_smoothing_to_discard_piles(fresh, tracker)

    assert fresh["self"][0]["tile"] == "6p"


@pytest.mark.unit
def test_distinct_turn_indices_are_independent_slots() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    apply_temporal_smoothing_to_discard_piles(
        {"self": [_pile("5p", turn_index=1, confidence=0.9)]},
        tracker,
    )
    next_frame = {
        "self": [
            _pile("5p", turn_index=1, confidence=0.9),
            _pile("9m", turn_index=2, confidence=0.9),
        ],
    }
    counters = apply_temporal_smoothing_to_discard_piles(next_frame, tracker)
    # First slot has support=2 → stable; second slot has support=1 → no estimate yet
    assert counters["stable"] >= 1
    assert counters["smoothed"] == 0
    assert next_frame["self"][1]["tile"] == "9m"


@pytest.mark.unit
def test_distinct_players_do_not_cross_contaminate() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    for _ in range(3):
        apply_temporal_smoothing_to_discard_piles(
            {"self": [_pile("5p", turn_index=1, confidence=0.9)]},
            tracker,
        )
    cross = {"left_opponent": [_pile("9m", turn_index=1, confidence=0.9)]}
    apply_temporal_smoothing_to_discard_piles(cross, tracker)
    assert cross["left_opponent"][0]["tile"] == "9m"


@pytest.mark.unit
def test_missing_or_invalid_fields_are_skipped_silently() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=1)
    discard_piles = {
        "self": [
            {"tile": "", "turn_index": 1, "confidence": 0.9},          # blank tile
            {"tile": "5p", "turn_index": None, "confidence": 0.9},     # no turn
            {"tile": "5p", "turn_index": "abc", "confidence": 0.9},    # bad turn
            "not a dict",                                              # garbage
        ],
    }
    counters = apply_temporal_smoothing_to_discard_piles(discard_piles, tracker)
    assert counters["observed"] == 0


@pytest.mark.unit
def test_non_dict_pile_returns_zero_counters() -> None:
    tracker = TemporalTileTracker()
    counters = apply_temporal_smoothing_to_discard_piles({"self": "not a list"}, tracker)
    assert counters == {"observed": 0, "smoothed": 0, "stable": 0}


@pytest.mark.unit
def test_non_dict_input_returns_zero_counters() -> None:
    tracker = TemporalTileTracker()
    counters = apply_temporal_smoothing_to_discard_piles("garbage", tracker)  # type: ignore[arg-type]
    assert counters == {"observed": 0, "smoothed": 0, "stable": 0}
