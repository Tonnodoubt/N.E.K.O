from __future__ import annotations

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.generator import build_decision


def test_call_decision_recommends_chi_when_hand_is_close_and_unpressured() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        buttons=["chi", "skip"],
        hand_tiles=[
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "3s",
            "4s",
            "5s",
            "7s",
            "8s",
            "2z",
            "2z",
        ],
        analysis_hints={
            "tile_level_available": True,
            "recognized_hand_tile_count": 13,
            "shanten_estimate": 2,
        },
    )

    decision = build_decision(state)

    assert decision.recommended_focus == "call_decision"
    assert decision.engine_meta["recommended_button_types"] == ["chi"]
    assert "可以考虑吃" in decision.suggestion


def test_call_decision_still_skips_under_pressure() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        buttons=["pon", "skip"],
        hand_tiles=[
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "3s",
            "4s",
            "5s",
            "7s",
            "8s",
            "2z",
            "2z",
        ],
        riichi_players=["right_opponent"],
        analysis_hints={
            "tile_level_available": True,
            "recognized_hand_tile_count": 13,
            "shanten_estimate": 2,
        },
    )

    decision = build_decision(state)

    assert decision.recommended_focus == "call_decision"
    assert decision.engine_meta["recommended_button_types"] == ["skip"]
    assert "跳过" in decision.suggestion


def test_call_decision_skips_far_open_hand_without_yaku_signal() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        buttons=["chi", "skip"],
        hand_tiles=[
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "3s",
            "4s",
            "5s",
            "7s",
            "8s",
            "1p",
            "9m",
        ],
        analysis_hints={
            "tile_level_available": True,
            "recognized_hand_tile_count": 13,
            "shanten_estimate": 2,
        },
    )

    decision = build_decision(state)

    assert decision.recommended_focus == "call_decision"
    assert decision.engine_meta["recommended_button_types"] == ["skip"]
    assert "无役" in decision.detail


def test_call_decision_allows_tanyao_open_route() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        buttons=["chi", "skip"],
        hand_tiles=[
            "2m",
            "3m",
            "4m",
            "4p",
            "5p",
            "6p",
            "3s",
            "4s",
            "5s",
            "6s",
            "7s",
            "2p",
            "8p",
        ],
        analysis_hints={
            "tile_level_available": True,
            "recognized_hand_tile_count": 13,
            "shanten_estimate": 2,
        },
    )

    decision = build_decision(state)

    assert decision.recommended_focus == "call_decision"
    assert decision.engine_meta["recommended_button_types"] == ["chi"]
    assert "断幺" in decision.detail
