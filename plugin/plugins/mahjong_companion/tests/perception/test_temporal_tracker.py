"""Unit tests for the cross-frame temporal tile tracker.

Goal: pin the behaviour that one bad frame cannot override several good
ones, while a sustained genuine change still flips the estimate within
a small bounded number of frames.
"""

from __future__ import annotations

import pytest

from plugin.plugins.mahjong_companion.perception.temporal_tracker import (
    StableTileEstimate,
    TemporalTileTracker,
    TileObservation,
)


@pytest.mark.unit
def test_single_frame_below_min_support_returns_none() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    tracker.observe("self:1", [TileObservation("5p", 0.9)])
    assert tracker.estimate("self:1") is None


@pytest.mark.unit
def test_two_consistent_frames_produce_estimate() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    tracker.observe("self:1", [TileObservation("5p", 0.9)])
    tracker.observe("self:1", [TileObservation("5p", 0.9)])
    estimate = tracker.estimate("self:1")
    assert isinstance(estimate, StableTileEstimate)
    assert estimate.tile == "5p"
    assert estimate.support == 2
    assert estimate.total_frames == 2
    assert 0.0 < estimate.confidence <= 1.0


@pytest.mark.unit
def test_one_noisy_frame_does_not_flip_estimate() -> None:
    """Three good frames + one bad frame must still resolve to the good tile."""
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    for _ in range(3):
        tracker.observe("self:1", [TileObservation("5p", 0.95)])
    tracker.observe("self:1", [TileObservation("6p", 0.55)])  # one wrong frame
    tracker.observe("self:1", [TileObservation("5p", 0.95)])
    estimate = tracker.estimate("self:1")
    assert estimate is not None
    assert estimate.tile == "5p"


@pytest.mark.unit
def test_sustained_change_eventually_flips_estimate() -> None:
    """If the true tile genuinely changes, the tracker must catch up."""
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    for _ in range(3):
        tracker.observe("self:1", [TileObservation("5p", 0.9)])
    for _ in range(6):
        tracker.observe("self:1", [TileObservation("6p", 0.9)])
    estimate = tracker.estimate("self:1")
    assert estimate is not None
    assert estimate.tile == "6p"


@pytest.mark.unit
def test_independent_slots_do_not_interfere() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=2)
    for _ in range(3):
        tracker.observe("self:1", [TileObservation("5p", 0.9)])
        tracker.observe("self:2", [TileObservation("9m", 0.9)])
    assert tracker.estimate("self:1").tile == "5p"  # type: ignore[union-attr]
    assert tracker.estimate("self:2").tile == "9m"  # type: ignore[union-attr]


@pytest.mark.unit
def test_empty_observation_decays_existing_scores() -> None:
    """Frames with no observation still age the slot's history."""
    tracker = TemporalTileTracker(decay=0.5, min_support=1, prune_threshold=0.05)
    tracker.observe("self:1", [TileObservation("5p", 1.0)])
    initial = tracker.estimate("self:1")
    assert initial is not None
    initial_confidence = initial.confidence
    for _ in range(8):
        tracker.observe("self:1", [])
    aged = tracker.estimate("self:1")
    if aged is not None:
        assert aged.confidence < initial_confidence


@pytest.mark.unit
def test_top_k_observations_accumulate_runner_up() -> None:
    """Passing top-2 observations lets the runner-up build score too."""
    tracker = TemporalTileTracker(decay=0.6, min_support=2)
    for _ in range(4):
        tracker.observe(
            "self:1",
            [TileObservation("6p", 0.55), TileObservation("5p", 0.40)],
        )
    estimate = tracker.estimate("self:1")
    assert estimate is not None
    assert estimate.tile == "6p"


@pytest.mark.unit
def test_reset_specific_slot_keeps_others() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=1)
    tracker.observe("self:1", [TileObservation("5p", 0.9)])
    tracker.observe("self:2", [TileObservation("9m", 0.9)])
    tracker.reset("self:1")
    assert tracker.estimate("self:1") is None
    assert tracker.estimate("self:2") is not None


@pytest.mark.unit
def test_reset_all_slots() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=1)
    tracker.observe("self:1", [TileObservation("5p", 0.9)])
    tracker.observe("self:2", [TileObservation("9m", 0.9)])
    tracker.reset()
    assert tracker.slots() == []


@pytest.mark.unit
def test_invalid_decay_rejected() -> None:
    with pytest.raises(ValueError):
        TemporalTileTracker(decay=0.0)
    with pytest.raises(ValueError):
        TemporalTileTracker(decay=1.0)


@pytest.mark.unit
def test_invalid_min_support_rejected() -> None:
    with pytest.raises(ValueError):
        TemporalTileTracker(min_support=0)


@pytest.mark.unit
def test_blank_or_whitespace_tile_observation_ignored() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=1)
    tracker.observe("self:1", [TileObservation("", 0.9), TileObservation("   ", 0.9)])
    assert tracker.estimate("self:1") is None


@pytest.mark.unit
def test_confidence_is_clamped_to_unit_interval() -> None:
    tracker = TemporalTileTracker(decay=0.7, min_support=1)
    tracker.observe("self:1", [TileObservation("5p", 5.0)])
    estimate = tracker.estimate("self:1")
    assert estimate is not None
    assert estimate.confidence <= 1.0


@pytest.mark.unit
def test_unseen_slot_returns_none() -> None:
    tracker = TemporalTileTracker()
    assert tracker.estimate("never:seen") is None
