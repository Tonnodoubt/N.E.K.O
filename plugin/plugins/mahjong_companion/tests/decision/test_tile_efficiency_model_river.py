from __future__ import annotations

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.tile_efficiency import build_mahjong_analysis


def test_model_river_tile_overflow_adds_teaching_warning():
    state = PerceivedGameState(
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z"],
        visible_tiles=["8p"] * 5,
        analysis_hints={
            "model_river_tile_overflow_counts": {"8p": 5},
            "visible_tiles": ["8p"] * 5,
            "discard_parser_source": "external_discard_recognizer",
            "recognized_discard_tile_count": 5,
        },
    )

    analysis = build_mahjong_analysis(state)

    assert any("超过四张" in point for point in analysis.teaching_points)


def test_model_river_unknown_count_adds_conservative_teaching_warning():
    state = PerceivedGameState(
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z"],
        visible_tiles=["8p", "9p", "1s"],
        analysis_hints={
            "model_river_unknown_count": 3,
            "visible_tiles": ["8p", "9p", "1s"],
            "discard_parser_source": "external_discard_recognizer",
            "recognized_discard_tile_count": 3,
        },
    )

    analysis = build_mahjong_analysis(state)

    assert any("未知牌" in point and "保守" in point for point in analysis.teaching_points)


def test_model_river_health_adds_strategy_readiness_point():
    state = PerceivedGameState(
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z"],
        visible_tiles=["8p", "9p", "1s", "2z"],
        analysis_hints={
            "discard_parser_source": "model_river_adapter",
            "model_river_known_count": 58,
            "model_river_candidate_count": 60,
            "model_river_unknown_count": 2,
            "visible_tiles": ["8p", "9p", "1s", "2z"],
        },
    )

    analysis = build_mahjong_analysis(state)

    assert any("识别健康度够用" in point and "轻量牌河判断" in point for point in analysis.teaching_points)


def test_model_river_dead_candidate_adds_safety_teaching_point():
    state = PerceivedGameState(
        hand_tiles=["8p", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z"],
        visible_tiles=["8p", "8p", "8p", "8p"],
        analysis_hints={
            "candidate_discards": [
                {
                    "tile": "8p",
                    "score": 0.9,
                    "ukeire_estimate": 6,
                    "safety_hint": "dead",
                    "reason": "visible river wall",
                }
            ],
            "discard_parser_source": "model_river_adapter",
            "model_river_known_count": 4,
            "model_river_candidate_count": 4,
            "model_river_unknown_count": 0,
            "visible_tiles": ["8p", "8p", "8p", "8p"],
        },
    )

    analysis = build_mahjong_analysis(state)

    assert any("已经看完" in point and "绝张" in point for point in analysis.teaching_points)


def test_model_river_riichi_pressure_adds_river_defense_alert():
    state = PerceivedGameState(
        hand_tiles=["5p", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z"],
        riichi_players=["right_opponent"],
        analysis_hints={
            "candidate_discards": [
                {
                    "tile": "5p",
                    "score": 0.9,
                    "ukeire_estimate": 8,
                    "safety_hint": "unknown",
                    "reason": "best shape",
                }
            ],
            "discard_parser_source": "model_river_adapter",
            "model_river_known_count": 30,
            "model_river_candidate_count": 30,
            "model_river_unknown_count": 0,
        },
    )

    analysis = build_mahjong_analysis(state)

    assert any("牌河策略" in alert and "没有明确安全线索" in alert for alert in analysis.defense_alerts)


def test_shanten_preserving_connected_discard_reason_is_not_self_contradictory():
    state = PerceivedGameState(
        hand_tiles=["7s", "5s", "6s", "9p", "9p", "9p", "3p", "4p", "5p", "2p"],
        analysis_hints={
            "recognized_meld_group_count": 1,
            "post_meld_hand_shape": "waiting",
        },
    )

    analysis = build_mahjong_analysis(state)

    assert analysis.candidate_discards
    assert analysis.candidate_discards[0]["recommendation_strength"] in {"strong", "medium", "weak"}
    reason = str(analysis.candidate_discards[0]["reason"])
    assert "不建议" not in reason
    assert "不退向听" in reason


def test_shape_loss_candidate_reason_is_framed_as_relative_candidate():
    state = PerceivedGameState(
        hand_tiles=["2p", "3p", "4p", "6p", "6p", "6s", "6s", "7s", "7s", "8s"],
        analysis_hints={
            "recognized_meld_group_count": 1,
            "post_meld_hand_shape": "waiting",
        },
    )

    analysis = build_mahjong_analysis(state)

    assert analysis.candidate_discards
    assert analysis.candidate_discards[0]["recommendation_strength"] in {"medium", "weak"}
    reason = str(analysis.candidate_discards[0]["reason"])
    assert "不建议" not in reason
    assert "相对候选" in reason or "不退向听" in reason


def test_candidate_discards_preserve_recommendation_strength_from_hints():
    state = PerceivedGameState(
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z"],
        analysis_hints={
            "candidate_discards": [
                {
                    "tile": "1z",
                    "score": 0.8,
                    "ukeire_estimate": 12,
                    "safety_hint": "medium",
                    "recommendation_strength": "weak",
                    "reason": "external model candidate",
                }
            ],
        },
    )

    analysis = build_mahjong_analysis(state)

    assert analysis.candidate_discards[0]["recommendation_strength"] == "weak"
