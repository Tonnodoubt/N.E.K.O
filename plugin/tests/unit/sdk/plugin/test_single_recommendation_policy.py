from __future__ import annotations

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.generator import build_decision
from plugin.plugins.mahjong_companion.status_snapshot import current_screen_overlays


def test_call_window_has_exactly_one_button_recommendation() -> None:
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

    assert len(decision.engine_meta["recommended_button_types"]) == 1
    assert decision.engine_meta["single_recommendation"]["kind"] == "button"


def test_skip_alone_does_not_become_action_advice() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        buttons=["skip"],
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

    assert decision.recommended_focus == "skip_observe"
    assert decision.engine_meta["recommended_button_types"] == []
    assert decision.engine_meta["single_recommendation"]["kind"] == "observe"


def test_status_snapshot_suppresses_live_screen_overlays() -> None:
    assert current_screen_overlays(
        {
            "screen_overlays": [
                {
                    "kind": "action_button_recommendation",
                    "box": {"left": 1, "top": 2, "width": 3, "height": 4},
                }
            ]
        }
    ) == []
