from __future__ import annotations

from plugin.plugins.mahjong_companion.overlay.view import _advice_text, _advice_view


def _status_with_strength(strength: str) -> dict:
    return {
        "window_bound": True,
        "runtime_mode": "watching",
        "last_decision": {
            "decision_type": "tile_efficiency_hint",
            "mahjong_analysis": {
                "candidate_discards": [
                    {
                        "tile": "7p",
                        "ukeire_estimate": 12,
                        "recommendation_strength": strength,
                        "reason": "打出后不退向听。",
                    }
                ]
            },
        },
    }


def test_advice_text_uses_strong_recommendation_label():
    assert _advice_text(_status_with_strength("strong")).startswith("优先考虑：七筒")


def test_advice_text_uses_medium_recommendation_label():
    assert _advice_text(_status_with_strength("medium")).startswith("可以考虑：七筒")


def test_advice_view_uses_weak_recommendation_label():
    view = _advice_view(_status_with_strength("weak"))

    assert view["primary"] == "七筒"
    assert view["reason"].startswith("仅作参考：")
