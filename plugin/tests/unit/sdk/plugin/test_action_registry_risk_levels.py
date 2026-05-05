from __future__ import annotations

from plugin.plugins.mahjong_companion.action.action_registry import ActionRegistry


def test_dangerous_action_requires_explicit_confirmation() -> None:
    registry = ActionRegistry()

    ok, reason = registry.validate(
        "dialog_confirm",
        current_scene="dialog",
        action_mode="assist",
        session_running=False,
        user_confirmed=False,
    )

    assert ok is False
    assert "dangerous" in reason


def test_dangerous_action_allows_confirmed_user() -> None:
    registry = ActionRegistry()

    ok, reason = registry.validate(
        "dialog_confirm",
        current_scene="dialog",
        action_mode="assist",
        session_running=False,
        user_confirmed=True,
    )

    assert ok is True
    assert reason == "allowed"


def test_game_button_action_is_not_registered_for_advice_only_scope() -> None:
    registry = ActionRegistry()

    ok, reason = registry.validate(
        "ui_pon",
        current_scene="in_match",
        action_mode="assist",
        session_running=True,
        user_confirmed=True,
    )

    assert ok is False
    assert "unknown action_id" in reason


def test_safe_action_ignores_legacy_confirmation_flag() -> None:
    registry = ActionRegistry([
        {
            "action_id": "safe_requires_confirmation_legacy",
            "category": "test",
            "label": "safe legacy",
            "allowed_contexts": ["menu"],
            "requires_confirmation": True,
            "risk_level": "safe",
        }
    ])

    ok, reason = registry.validate(
        "safe_requires_confirmation_legacy",
        current_scene="menu",
        action_mode="assist",
        session_running=False,
        user_confirmed=False,
    )

    assert ok is True
    assert reason == "allowed"
