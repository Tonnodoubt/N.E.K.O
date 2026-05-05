from __future__ import annotations

from collections import Counter

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.tile_efficiency import (
    _defensive_safety_hint,
    build_incremental_draw_candidates,
    build_mahjong_analysis,
)


def test_candidate_discards_rank_post_discard_shanten_before_shape_score() -> None:
    hand = [
        "1s",
        "3s",
        "3s",
        "4s",
        "5s",
        "6s",
        "6m",
        "7m",
        "8m",
        "7p",
        "7p",
        "8p",
        "8p",
        "9p",
    ]
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=hand,
        analysis_hints={
            "tile_level_available": True,
            "recognized_hand_tile_count": 14,
        },
    )

    analysis = build_mahjong_analysis(state)
    candidates = analysis.candidate_discards

    assert candidates[0]["tile"] == "1s"
    assert candidates[0]["post_discard_shanten"] == 0
    assert candidates[1]["post_discard_shanten"] == 1
    assert candidates[1]["ukeire_estimate"] > candidates[0]["ukeire_estimate"]


def test_incremental_draw_candidates_rank_drawn_tile_when_it_preserves_tenpai() -> None:
    waiting_hand = [
        "3s",
        "3s",
        "4s",
        "5s",
        "6s",
        "6m",
        "7m",
        "8m",
        "7p",
        "7p",
        "8p",
        "8p",
        "9p",
    ]
    cached_candidates = [
        {
            "tile": "9p",
            "score": 1.0,
            "ukeire_estimate": 6,
            "safety_hint": "high",
            "reason": "cached raw top",
        },
        {
            "tile": "8m",
            "score": 0.5,
            "ukeire_estimate": 4,
            "safety_hint": "low",
            "reason": "cached fallback",
        },
    ]

    candidates = build_incremental_draw_candidates(
        waiting_hand,
        "1s",
        cached_candidates,
    )

    assert candidates[0]["tile"] == "1s"
    assert candidates[0]["source"] == "drawn_tile"
    assert candidates[0]["post_discard_shanten"] == 0
    assert candidates[1]["tile"] == "9p"
    assert candidates[1]["post_discard_shanten"] == 1


def test_riichi_pressure_prioritizes_genbutsu_without_losing_shanten() -> None:
    hand = [
        "4m",
        "6m",
        "6m",
        "6s",
        "2p",
        "2s",
        "8p",
        "5p",
        "3m",
        "2p",
        "1z",
        "8s",
        "6z",
        "6s",
    ]
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=hand,
        riichi_players=["right_opponent"],
        analysis_hints={
            "tile_level_available": True,
            "known_genbutsu_tiles": ["8s"],
        },
    )

    candidates = build_mahjong_analysis(state).candidate_discards

    assert candidates[0]["tile"] == "8s"
    assert candidates[0]["safety_hint"] == "genbutsu"
    assert candidates[0]["post_discard_shanten"] == candidates[1]["post_discard_shanten"]


def test_riichi_pressure_uses_suji_and_dead_tile_safety_ladder() -> None:
    counts = Counter(["1m", "5p", "9s", "2z"])

    assert _defensive_safety_hint(
        "1m",
        counts=counts,
        visible_tiles=[],
        genbutsu_tiles={"4m"},
        has_riichi_pressure=True,
    ) == "suji"
    assert _defensive_safety_hint(
        "5p",
        counts=counts,
        visible_tiles=["5p", "5p", "5p"],
        genbutsu_tiles=set(),
        has_riichi_pressure=True,
    ) == "dead"


def test_riichi_pressure_prioritizes_dead_tile_without_genbutsu() -> None:
    hand = [
        "4m",
        "6m",
        "6m",
        "6s",
        "2p",
        "2s",
        "8p",
        "5p",
        "3m",
        "2p",
        "1z",
        "8s",
        "6z",
        "6s",
    ]
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=hand,
        riichi_players=["right_opponent"],
        analysis_hints={
            "tile_level_available": True,
            "deck_state_complete": True,
            "visible_tiles": ["8s", "8s", "8s"],
        },
    )

    candidates = build_mahjong_analysis(state).candidate_discards

    assert candidates[0]["tile"] == "8s"
    assert candidates[0]["safety_hint"] == "dead"


def test_unpressured_strategy_preserves_dora_when_efficiency_is_equal() -> None:
    hand = [
        "4s",
        "2m",
        "9p",
        "2p",
        "3s",
        "7m",
        "5p",
        "9p",
        "1s",
        "8m",
        "5m",
        "4z",
        "6m",
        "5s",
    ]
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=hand,
        dora_indicators=["3z"],
        analysis_hints={"tile_level_available": True},
    )

    candidates = build_mahjong_analysis(state).candidate_discards

    assert candidates[0]["tile"] != "4z"
    assert candidates[0]["post_discard_shanten"] == candidates[1]["post_discard_shanten"]
    assert candidates[0]["ukeire_estimate"] == candidates[1]["ukeire_estimate"]


def test_trusted_visible_tiles_reduce_dead_wait_ukeire() -> None:
    hand = [
        "1m",
        "2m",
        "3m",
        "4m",
        "5m",
        "6m",
        "2p",
        "3p",
        "4p",
        "6p",
        "7p",
        "8p",
        "4s",
        "9s",
    ]
    base_state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=hand,
        analysis_hints={"tile_level_available": True},
    )
    visible_state = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=hand,
        analysis_hints={
            "tile_level_available": True,
            "deck_state_complete": True,
            "visible_tiles": ["9s", "9s", "9s"],
        },
    )

    base_4s = next(item for item in build_mahjong_analysis(base_state).candidate_discards if item["tile"] == "4s")
    visible_4s = next(
        item for item in build_mahjong_analysis(visible_state).candidate_discards if item["tile"] == "4s"
    )

    assert visible_4s["ukeire_estimate"] < base_4s["ukeire_estimate"]
