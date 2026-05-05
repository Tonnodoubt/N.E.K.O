from __future__ import annotations

import logging
from pathlib import Path

import pytest

from plugin.plugins.mahjong_companion.action.action_log import load_action_log
from plugin.plugins.mahjong_companion.orchestrator import SessionOrchestrator


class _FakePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logger = logging.getLogger("mahjong-companion-action-audit-test")
        self.statuses: list[dict[str, object]] = []

    def data_path(self, *parts: str) -> Path:
        path = self.root / "data"
        if parts:
            path = path.joinpath(*parts)
        return path

    def report_status(self, payload: dict[str, object]) -> None:
        self.statuses.append(dict(payload))


@pytest.mark.asyncio
async def test_dangerous_action_rejection_is_audited(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.state.action_mode = "assist"
    orchestrator.state.scene = "dialog"

    result = await orchestrator.execute_assist_action("dialog_confirm")

    assert result.value["ok"] is False
    assert result.value["risk_level"] == "dangerous"
    entries = load_action_log(plugin.data_path("session_cache"))
    assert entries[-1]["risk_level"] == "dangerous"
    assert entries[-1]["confirmation_chain"][0]["step"] == "user_explicit"
    assert entries[-1]["confirmation_chain"][1]["step"] == "registry_validate"
    assert "rejected" in entries[-1]["confirmation_chain"][1]["value"]


@pytest.mark.asyncio
async def test_confirmed_dry_run_records_confirmation_chain(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.state.action_mode = "assist"
    orchestrator.state.scene = "dialog"

    result = await orchestrator.execute_assist_action(
        "dialog_confirm",
        dry_run=True,
        user_confirmed=True,
    )

    assert result.value["ok"] is True
    entries = load_action_log(plugin.data_path("session_cache"))
    assert entries[-1]["risk_level"] == "dangerous"
    assert [item["step"] for item in entries[-1]["confirmation_chain"]] == [
        "user_explicit",
        "registry_validate",
        "dry_run",
    ]
