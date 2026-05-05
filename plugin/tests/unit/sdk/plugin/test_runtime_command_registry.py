from __future__ import annotations

from typing import Any

import pytest

from plugin.plugins.mahjong_companion.runtime.command_registry import RuntimeCommandRegistry


def test_register_rejects_duplicate_command_name() -> None:
    registry = RuntimeCommandRegistry()

    registry.register("refresh_status", lambda payload: {"ok": True})

    with pytest.raises(ValueError, match="command already registered: refresh_status"):
        registry.register(" REFRESH_STATUS ", lambda payload: {"ok": True})


def test_register_rejects_empty_command_name() -> None:
    registry = RuntimeCommandRegistry()

    with pytest.raises(ValueError, match="command name must be non-empty"):
        registry.register("  ", lambda payload: {"ok": True})


def test_dispatch_unknown_command_returns_existing_error_shape() -> None:
    registry = RuntimeCommandRegistry()

    result = registry.dispatch("missing_action", {})

    assert result == {"ok": False, "error": "unsupported runtime action: missing_action"}


def test_dispatch_empty_command_returns_existing_error_shape() -> None:
    registry = RuntimeCommandRegistry()

    result = registry.dispatch("  ", {})

    assert result == {"ok": False, "error": "runtime action is empty"}


def test_dispatch_calls_handler_with_payload() -> None:
    registry = RuntimeCommandRegistry()
    seen: dict[str, Any] = {}

    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        seen["payload"] = payload
        return {"ok": True, "mode": payload["mode"]}

    registry.register("set_mode", handler)
    payload = {"mode": "replay"}

    result = registry.dispatch(" SET_MODE ", payload)

    assert result == {"ok": True, "mode": "replay"}
    assert seen["payload"] is payload
    assert registry.known_commands() == ["set_mode"]


def test_dispatch_lets_handler_exceptions_surface() -> None:
    registry = RuntimeCommandRegistry()

    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    registry.register("explode", handler)

    with pytest.raises(RuntimeError, match="boom"):
        registry.dispatch("explode", {})
