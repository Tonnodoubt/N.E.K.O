from __future__ import annotations

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.generator import build_decision
from plugin.plugins.mahjong_companion.decision.tile_efficiency import (
    _calculate_discard_ukeire,
    build_incremental_draw_candidates,
)


HAND = ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "2s", "3s", "4s", "5z", "9m"]


def test_discard_ukeire_uses_remaining_tile_counts() -> None:
    assert _calculate_discard_ukeire(HAND, "9m") == 3
    assert _calculate_discard_ukeire(HAND, "9m", visible_tiles=["5z", "5z", "5z"]) == 0


def test_complete_deck_state_enables_real_ukeire_for_candidate_ordering() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=HAND,
        analysis_hints={
            "deck_state_complete": True,
            "visible_tiles": ["5z", "5z", "5z"],
        },
    )

    decision = build_decision(state)

    assert decision.mahjong_analysis["candidate_discards"][0]["tile"] == "5z"
    assert decision.mahjong_analysis["candidate_discards"][0]["ukeire_estimate"] == 3
    assert decision.mahjong_analysis["ukeire_estimate"] == 3


def test_riichi_risk_alert_mentions_confirmed_genbutsu_without_suji_or_wall_claims() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=HAND,
        riichi_players=["right_opponent"],
        analysis_hints={
            "genbutsu_tiles": ["9m"],
        },
    )

    decision = build_decision(state)
    alerts = " ".join(decision.mahjong_analysis["defense_alerts"])

    assert "立直" in decision.mahjong_analysis["defense_alerts"][0]
    assert "现物" in alerts
    assert "九万" in alerts
    assert "9m" not in alerts
    assert "筋牌" not in alerts
    assert "壁牌" not in alerts


def test_known_genbutsu_state_field_feeds_defense_alerts() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=HAND,
        riichi_players=["right_opponent"],
        known_genbutsu_tiles=["9m"],
    )

    decision = build_decision(state)
    alerts = " ".join(decision.mahjong_analysis["defense_alerts"])

    assert "现物" in alerts
    assert "九万" in alerts
    assert "9m" not in alerts


def test_self_riichi_does_not_create_defense_pressure() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=HAND,
        riichi_players=["self"],
        known_genbutsu_tiles=["9m"],
    )

    decision = build_decision(state)
    alerts = " ".join(decision.mahjong_analysis["defense_alerts"])

    assert "立直压力" not in alerts
    assert "现物" not in alerts


def test_legacy_opposite_riichi_alias_feeds_defense_alerts() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=HAND,
        riichi_players=["opposite"],
        known_genbutsu_tiles=["5z"],
    )

    decision = build_decision(state)
    alerts = " ".join(decision.mahjong_analysis["defense_alerts"])

    assert "立直" in alerts
    assert "现物" in alerts
    assert "白板" in alerts
    assert "5z" not in alerts


def test_tile_efficiency_suggestion_uses_human_tile_names() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        analysis_hints={
            "tile_level_available": True,
            "analysis_confidence": 0.82,
            "recognized_hand_tile_count": 14,
            "candidate_discards": [
                {
                    "tile": "1z",
                    "score": 1.0,
                    "ukeire_estimate": 8,
                    "safety_hint": "medium",
                    "reason": "1z 更像单张孤张",
                }
            ],
        },
    )

    decision = build_decision(state)
    text = " ".join(
        [
            decision.suggestion,
            decision.detail,
            *decision.mahjong_analysis["teaching_points"],
        ]
    )

    assert decision.decision_type == "tile_efficiency_hint"
    assert "东风" in text
    assert "1z" not in text


def test_incremental_draw_candidates_compares_cached_plan_with_drawn_tile() -> None:
    cached_candidates = [
        {
            "tile": "5z",
            "score": 1.0,
            "ukeire_estimate": 4,
            "safety_hint": "high",
            "reason": "5z 鏇村儚鍗曞紶瀛ゅ紶",
        }
    ]

    candidates = build_incremental_draw_candidates(
        HAND[:-1],
        "9m",
        cached_candidates,
    )

    assert {item["tile"] for item in candidates} == {"5z", "9m"}
    assert {item["source"] for item in candidates} == {"preturn_cached", "drawn_tile"}
