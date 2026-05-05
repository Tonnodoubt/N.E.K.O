from __future__ import annotations

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.generator import build_decision


def test_riichi_decision_pushes_good_wait_without_pressure() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.88,
        buttons=["riichi", "skip"],
        analysis_hints={
            "tile_level_available": True,
            "shanten_estimate": 0,
            "candidate_discards": [
                {
                    "tile": "1s",
                    "ukeire_estimate": 7,
                    "post_discard_shanten": 0,
                    "wait_quality_bonus": 2,
                    "safety_hint": "medium",
                }
            ],
        },
    )

    decision = build_decision(state)

    assert decision.recommended_focus == "riichi_decision"
    assert decision.engine_meta["recommended_button_types"] == ["riichi"]
    assert "立直" in decision.suggestion


def test_riichi_decision_skips_bad_wait_under_riichi_pressure() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.88,
        buttons=["riichi", "skip"],
        riichi_players=["right_opponent"],
        analysis_hints={
            "tile_level_available": True,
            "shanten_estimate": 0,
            "candidate_discards": [
                {
                    "tile": "1z",
                    "ukeire_estimate": 2,
                    "post_discard_shanten": 0,
                    "wait_quality_bonus": -1,
                    "safety_hint": "low",
                }
            ],
        },
    )

    decision = build_decision(state)

    assert decision.recommended_focus == "riichi_decision"
    assert decision.engine_meta["recommended_button_types"] == ["skip"]
    assert "跳过立直" in decision.suggestion
