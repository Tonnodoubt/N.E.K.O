"""Cross-frame temporal smoothing for tile classifications.

Single-frame perception is brittle: animation mid-frames, riichi glow
effects, and ambiguous tile pairs (5p/6p, 6s/9s, ...) all flicker enough
to flip the predicted tile for one or two frames at a time. This module
keeps a per-slot exponentially-weighted score map so a momentarily
wrong frame cannot override several stable ones.

The tracker is deliberately backend-agnostic: callers feed it
``(slot_id, [(tile, confidence), ...])`` tuples and read back a
:class:`StableTileEstimate`. How callers form ``slot_id`` is up to them
- the discard path uses ``f"{player}:{turn_index}"`` because
``turn_index`` is a stable anchor, while the hand path needs multiset
alignment (handled at the call site, not here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TileObservation:
    """One classifier output for a single tile slot in one frame."""

    tile: str
    confidence: float


@dataclass(frozen=True)
class StableTileEstimate:
    """Smoothed prediction for one slot across recent frames."""

    tile: str
    confidence: float
    support: int
    total_frames: int


@dataclass
class _SlotState:
    scores: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    total_frames: int = 0


class TemporalTileTracker:
    """Per-slot exponentially-weighted vote over recent tile observations.

    Algorithm (per slot):

      - On each ``observe(slot_id, observations)``:

          * decay all existing scores by ``decay`` (e.g. ``0.7``);
          * for each observed ``(tile, confidence)`` add
            ``confidence * (1 - decay)`` to that tile's score.

      - ``estimate(slot_id)`` returns the highest-scoring tile, provided
        it has been observed at least ``min_support`` times.

    A smaller ``decay`` reacts faster to genuine tile changes; a larger
    ``decay`` smooths harder. The default of ``0.7`` gives roughly three
    frames to respond to a sustained switch.
    """

    def __init__(
        self,
        *,
        decay: float = 0.7,
        min_support: int = 2,
        prune_threshold: float = 0.01,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay!r}")
        if min_support < 1:
            raise ValueError(f"min_support must be >= 1; got {min_support!r}")
        self._decay = float(decay)
        self._min_support = int(min_support)
        self._prune_threshold = max(0.0, float(prune_threshold))
        self._slots: dict[str, _SlotState] = {}

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def min_support(self) -> int:
        return self._min_support

    def observe(self, slot_id: str, observations: Iterable[TileObservation]) -> None:
        """Record one frame of observations for the given slot.

        Observations should already be filtered to the top-k candidates
        the caller cares about (typically top-1 or top-2). Empty
        iterables still count as a frame so silence decays old scores.
        """
        slot = self._slots.setdefault(slot_id, _SlotState())
        slot.total_frames += 1
        for tile in list(slot.scores.keys()):
            slot.scores[tile] *= self._decay
            if slot.scores[tile] < self._prune_threshold:
                del slot.scores[tile]
                slot.support.pop(tile, None)
        weight = 1.0 - self._decay
        for obs in observations:
            tile = str(obs.tile or "").strip()
            if not tile:
                continue
            confidence = max(0.0, min(1.0, float(obs.confidence)))
            slot.scores[tile] = slot.scores.get(tile, 0.0) + confidence * weight
            slot.support[tile] = slot.support.get(tile, 0) + 1

    def estimate(self, slot_id: str) -> StableTileEstimate | None:
        """Best current estimate for ``slot_id``, or None if untracked."""
        slot = self._slots.get(slot_id)
        if slot is None or not slot.scores:
            return None
        tile, score = max(slot.scores.items(), key=lambda kv: kv[1])
        support = slot.support.get(tile, 0)
        if support < self._min_support:
            return None
        return StableTileEstimate(
            tile=tile,
            confidence=round(min(1.0, score), 4),
            support=support,
            total_frames=slot.total_frames,
        )

    def reset(self, slot_id: str | None = None) -> None:
        """Forget history for one slot, or all slots if ``None``."""
        if slot_id is None:
            self._slots.clear()
        else:
            self._slots.pop(slot_id, None)

    def slots(self) -> list[str]:
        """All slot ids the tracker has seen."""
        return list(self._slots.keys())


SINGLE_FRAME_TRUST_THRESHOLD = 0.85
"""Above this single-frame confidence we trust the new frame even when
tracker history disagrees - lets genuinely-changing tiles flip without
fighting the smoother."""


def apply_temporal_smoothing_to_discard_piles(
    discard_piles: dict[str, list[dict]],
    tracker: TemporalTileTracker,
    *,
    single_frame_trust_threshold: float = SINGLE_FRAME_TRUST_THRESHOLD,
) -> dict[str, int]:
    """Observe every discard item into ``tracker`` and overwrite its tile
    in place when tracker history outvotes a low-confidence single frame.

    The discard slot key is ``f"{player}:{turn_index}"`` because
    ``turn_index`` is a stable anchor across frames (the discard pile
    only grows; tiles never shift index).

    Override rule: overwrite only when ``tracker.estimate`` returns a
    different tile (which already requires ``support >= min_support``,
    enforced inside the tracker) AND the single-frame confidence is
    below ``single_frame_trust_threshold``. The EMA score is not
    compared directly to the single-frame confidence because the two
    live on different scales (EMA score saturates at the input
    confidence; comparing them directly would let any sub-saturation
    history get overruled).

    Returns counters for diagnostics:

    - ``observed``: items fed into the tracker this frame
    - ``smoothed``: items whose tile was overwritten by the tracker
    - ``stable``: items confirmed by the tracker (same tile, no change)
    """
    counters = {"observed": 0, "smoothed": 0, "stable": 0}
    if not isinstance(discard_piles, dict):
        return counters

    for player, pile in discard_piles.items():
        if not isinstance(pile, list):
            continue
        for item in pile:
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            turn_index = item.get("turn_index")
            if not tile or turn_index is None:
                continue
            try:
                turn_int = int(turn_index)
            except (TypeError, ValueError):
                continue
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            slot_key = f"{player}:{turn_int}"
            tracker.observe(slot_key, [TileObservation(tile, confidence)])
            counters["observed"] += 1

            estimate = tracker.estimate(slot_key)
            if estimate is None:
                continue
            if estimate.tile == tile:
                counters["stable"] += 1
                continue
            if confidence < single_frame_trust_threshold:
                item["tile"] = estimate.tile
                item["temporal_smoothed"] = True
                item["temporal_original_tile"] = tile
                item["temporal_original_confidence"] = round(confidence, 4)
                item["temporal_confidence"] = estimate.confidence
                item["temporal_support"] = estimate.support
                counters["smoothed"] += 1
    return counters
