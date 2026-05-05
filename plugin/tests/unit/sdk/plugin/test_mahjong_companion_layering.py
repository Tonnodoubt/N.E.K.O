from __future__ import annotations

import logging
from pathlib import Path

import pytest

from plugin.plugins.mahjong_companion import MahjongCompanionPlugin
from plugin.plugins.mahjong_companion.config_defaults import DEFAULT_CONFIG, merge_runtime_config
from plugin.plugins.mahjong_companion.gates import DefaultFrameChangeGate
from plugin.plugins.mahjong_companion.narration.events import NarrationEvent
from plugin.plugins.mahjong_companion.orchestrator import SessionOrchestrator


class _FakePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logger = logging.getLogger("mahjong-companion-layering-test")
        self.statuses: list[dict[str, object]] = []
        self.messages: list[dict[str, object]] = []

    def data_path(self, *parts: str) -> Path:
        path = self.root / "data"
        if parts:
            path = path.joinpath(*parts)
        return path

    def report_status(self, payload: dict[str, object]) -> None:
        self.statuses.append(dict(payload))

    def push_message(self, **kwargs: object) -> dict[str, object]:
        self.messages.append(dict(kwargs))
        return {"ok": True}


class _FakeCtx:
    plugin_id = "mahjong_companion"

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.logger = logging.getLogger("mahjong-companion-plugin-test")
        self.metadata = {}
        self.bus = {"messages": "bus"}
        self._effective_config = {
            "plugin": {"store": {"enabled": True}, "database": {"enabled": True, "name": "data.db"}},
            "plugin_state": {"backend": "file"},
        }
        self.status: dict[str, object] | None = None

    async def get_own_config(self, timeout: float = 5.0) -> dict[str, object]:
        return {}

    async def get_own_base_config(self, timeout: float = 5.0) -> dict[str, object]:
        return {}

    def update_status(self, status: dict[str, object]) -> None:
        self.status = dict(status)

    def push_message(self, **kwargs: object) -> dict[str, object]:
        return {"ok": True}


def _sample_frame_path(name: str) -> Path:
    return Path(__file__).resolve().parents[4] / "plugins" / "mahjong_companion" / "data" / "debug_samples" / name


def test_frame_change_gate_skips_identical_frames() -> None:
    gate = DefaultFrameChangeGate()
    frame_path = _sample_frame_path("20260415-050905-263947-frame.png")

    first = gate.evaluate(frame_path, enabled=True, min_change_distance=1, stable_skip_limit=10)
    second = gate.evaluate(frame_path, enabled=True, min_change_distance=1, stable_skip_limit=10)

    assert first.should_process is True
    assert second.should_process is False
    assert second.reason == "frame_unchanged"


def test_frame_change_gate_allows_changed_frames() -> None:
    gate = DefaultFrameChangeGate()
    first_path = _sample_frame_path("20260415-050905-263947-frame.png")
    second_path = _sample_frame_path("20260415-071314-863534-frame.png")

    first = gate.evaluate(first_path, enabled=True, min_change_distance=1, stable_skip_limit=10)
    second = gate.evaluate(second_path, enabled=True, min_change_distance=1, stable_skip_limit=10)

    assert first.should_process is True
    assert second.should_process is True
    assert second.reason in {"frame_changed", "action_bar_changed"}


def test_frame_change_gate_detects_bottom_action_bar_change_even_when_table_is_stable(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    gate = DefaultFrameChangeGate()
    first_path = tmp_path / "frame-1.png"
    second_path = tmp_path / "frame-2.png"

    base = Image.new("RGB", (1280, 720), color=(40, 90, 60))
    draw = ImageDraw.Draw(base)
    draw.rectangle((160, 480, 1120, 700), fill=(200, 200, 190))
    base.save(first_path)

    changed = base.copy()
    draw_changed = ImageDraw.Draw(changed)
    draw_changed.rectangle((760, 560, 1080, 655), fill=(240, 150, 60))
    changed.save(second_path)

    first = gate.evaluate(first_path, enabled=True, min_change_distance=3, stable_skip_limit=10)
    second = gate.evaluate(second_path, enabled=True, min_change_distance=3, stable_skip_limit=10)

    assert first.should_process is True
    assert second.should_process is True
    assert second.reason in {"action_bar_changed", "frame_changed"}


def test_debug_dispatch_bypasses_runtime_guards(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    orchestrator.state.last_decision = {"priority": 80}
    orchestrator.state.last_decision_type = "action_available"

    event = NarrationEvent(
        event_type="action_available",
        channel="companion",
        delivery="proactive_notification",
        priority=80,
        summary="测试消息",
        detail="调试态允许绕过会话保护。",
        risk_level="medium",
        scene="menu",
        buttons=["confirm"],
        text="我这边先把整条链路打通给主人看。",
        speakable=True,
        dedupe_key="debug|pipeline",
    )

    result = orchestrator._dispatch_debug_narration_locked(event)

    assert result["ok"] is True
    assert plugin.messages
    assert plugin.messages[0]["content"] == event.text
    assert orchestrator.state.last_notification_ok is True


def test_get_status_exposes_host_status_and_runtime_status(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    orchestrator.state.running = True
    orchestrator.state.status = "scanning"
    orchestrator.state.scene = "in_match"

    payload = orchestrator.get_status()

    assert payload["status"] == "in_match"
    assert payload["runtime_status"] == "scanning"
    assert payload["scene"] == "in_match"


@pytest.mark.asyncio
async def test_startup_copies_and_registers_static_ui(tmp_path: Path) -> None:
    config_path = tmp_path / "mahjong_companion" / "plugin.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = _FakeCtx(config_path)
    plugin = MahjongCompanionPlugin(ctx)

    result = await plugin.startup()

    static_index = config_path.parent / "static" / "index.html"

    assert result.value["status"] == "ready"
    assert static_index.is_file()
    assert "Mahjong Companion" in static_index.read_text(encoding="utf-8")
    assert plugin.get_static_ui_config() is not None
    assert plugin.get_static_ui_config()["plugin_id"] == "mahjong_companion"
