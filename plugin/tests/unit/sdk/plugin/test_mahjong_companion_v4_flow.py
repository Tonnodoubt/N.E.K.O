from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.config_defaults import DEFAULT_CONFIG, merge_runtime_config
from plugin.plugins.mahjong_companion.contracts import DecisionResult, FramePacket, PerceivedGameState
from plugin.plugins.mahjong_companion.narration.events import NarrationEvent
from plugin.plugins.mahjong_companion.narration.generator import generate_narration
from plugin.plugins.mahjong_companion.narration.speech_policy import apply_speech_policy
from plugin.plugins.mahjong_companion.orchestrator import SessionOrchestrator, _prune_debug_samples
from plugin.plugins.mahjong_companion.session_state import now_iso
from plugin.plugins.mahjong_companion.window_binding import WindowBindingResult


class _FakePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logger = logging.getLogger("mahjong-companion-test")
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


class _FakeOverlay:
    def __init__(self, commands: list[str] | None = None) -> None:
        self.commands = list(commands or [])
        self.started = 0
        self.stopped = 0
        self.statuses: list[dict[str, object]] = []
        self.running = False

    def start(self) -> bool:
        self.started += 1
        self.running = True
        return True

    def stop(self) -> None:
        self.stopped += 1
        self.running = False

    def update_status(self, status: dict[str, object]) -> None:
        self.statuses.append(dict(status))

    def drain_commands(self) -> list[str]:
        commands = list(self.commands)
        self.commands = []
        return commands

    def is_running(self) -> bool:
        return self.running


def _iso_seconds_ago(seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _speech_cfg(**overrides: object) -> dict[str, object]:
    cfg = {
        "voice_enabled": True,
        "voice_mode": "key_events_only",
        "normal_voice_cooldown_sec": 18,
        "danger_voice_cooldown_sec": 5,
        "normal_notification_cooldown_sec": 18,
        "danger_notification_cooldown_sec": 5,
        "dedupe_window_sec": 8,
        "auto_dispatch_enabled": True,
    }
    cfg.update(overrides)
    return cfg


def _sample_frame_path(name: str) -> Path:
    return Path(__file__).resolve().parents[4] / "plugins" / "mahjong_companion" / "data" / "debug_samples" / name


def test_apply_speech_policy_suppresses_recent_action_notification() -> None:
    event = NarrationEvent(
        event_type="action_available",
        channel="nudge",
        delivery="silent_ui",
        priority=60,
        summary="有可操作按钮",
        detail="底部有按钮。",
        risk_level="medium",
        scene="in_match",
        buttons=["skip"],
        text="现在像是轮到你操作了。",
        speakable=True,
        dedupe_key="action_available|in_match|medium|skip",
    )

    updated = apply_speech_policy(
        event,
        _speech_cfg(),
        last_notified_at=_iso_seconds_ago(3),
        last_notified_key=event.dedupe_key,
        last_notified_text=event.text,
    )

    assert updated.delivery == "silent_ui"
    assert updated.channel == "nudge"
    assert updated.speakable is False


def test_apply_speech_policy_allows_recent_call_window_notification() -> None:
    event = NarrationEvent(
        event_type="action_available",
        channel="nudge",
        delivery="silent_ui",
        priority=72,
        summary="有副露选择",
        detail="底部出现吃碰窗口。",
        risk_level="medium",
        scene="in_match",
        buttons=["chi", "skip"],
        text="建议按跳过处理，这口先别吃。",
        speakable=True,
        dedupe_key="action_available|in_match|medium|chi,skip",
    )

    updated = apply_speech_policy(
        event,
        _speech_cfg(),
        last_notified_at=_iso_seconds_ago(4),
        last_notified_key="other-action",
        last_notified_text="上一条普通提示",
    )

    assert updated.delivery == "proactive_notification"
    assert updated.channel == "nudge"
    assert updated.speakable is True


def test_apply_speech_policy_keeps_uncertain_state_silent() -> None:
    event = NarrationEvent(
        event_type="uncertain_state",
        channel="nudge",
        delivery="silent_ui",
        text="这一帧我还没看太清。",
        speakable=True,
        dedupe_key="uncertain",
    )

    updated = apply_speech_policy(event, _speech_cfg())

    assert updated.delivery == "silent_ui"
    assert updated.channel == "silent_ui"
    assert updated.speakable is False


def test_apply_speech_policy_accepts_naive_iso_timestamps() -> None:
    event = NarrationEvent(
        event_type="action_available",
        channel="nudge",
        delivery="proactive_notification",
        priority=55,
        summary="可以操作",
        detail="测试无时区时间戳。",
        risk_level="low",
        scene="in_match",
        buttons=["skip"],
        text="我看到了一个操作窗口。",
        speakable=True,
        dedupe_key="naive-time",
    )

    updated = apply_speech_policy(
        event,
        _speech_cfg(),
        last_spoken_at="2026-04-30T12:00:00",
        last_notified_at="2026-04-30T12:00:00",
        last_notified_key="other",
    )

    assert updated.event_type == "action_available"


def test_tile_efficiency_hint_uses_discard_suggestion_and_notifies() -> None:
    decision = DecisionResult(
        decision_type="tile_efficiency_hint",
        priority=62,
        risk_level="low",
        action_required=True,
        speakable=False,
        summary="tile efficiency is available",
        detail="structured hand state is available",
        suggestion="Discard 9m first.",
        recommended_focus="tile_efficiency",
        scene="in_match",
    )

    event, view_model, _debug = generate_narration(decision)
    updated = apply_speech_policy(event, _speech_cfg())

    assert event.text == "Discard 9m first."
    assert event.channel == "nudge"
    assert view_model.suggestion_level == "nudge"
    assert updated.delivery == "proactive_notification"
    assert updated.channel == "nudge"


def test_dispatch_narration_updates_notification_and_voice_state(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    orchestrator.state.running = True
    orchestrator.state.window_bound = True
    orchestrator.state.last_decision = {"priority": 90}
    orchestrator.state.last_decision_type = "danger_action"

    event = NarrationEvent(
        event_type="danger_action",
        channel="warning",
        delivery="voice_candidate",
        priority=90,
        summary="关键操作",
        detail="检测到高优先级按钮。",
        risk_level="high",
        scene="in_match",
        buttons=["ron"],
        text="这里像是有关键操作，我们先看清楚再点。",
        speakable=True,
        dedupe_key="danger_action|in_match|high|ron",
    )

    result = orchestrator._dispatch_narration_locked(event)

    assert result["ok"] is True
    assert plugin.messages
    assert plugin.messages[0]["content"] == event.text
    assert plugin.messages[0]["metadata"]["delivery"] == "voice_candidate"
    assert orchestrator.state.last_notification_ok is True
    assert orchestrator.state.last_spoken_text == event.text
    assert orchestrator.state.last_speak_ok is True


@pytest.mark.asyncio
async def test_speak_last_narration_rejects_when_window_unbound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    orchestrator.state.running = True
    orchestrator.state.last_decision = {"priority": 90}
    orchestrator.state.last_decision_ok = True
    orchestrator.state.last_decision_type = "danger_action"
    orchestrator.state.last_narration_ok = True
    orchestrator.state.last_narration = {
        "event_type": "danger_action",
        "channel": "warning",
        "delivery": "voice_candidate",
        "priority": 90,
        "summary": "关键操作",
        "detail": "检测到高优先级按钮。",
        "risk_level": "high",
        "scene": "in_match",
        "buttons": ["ron"],
        "text": "这里像是有关键操作，我们先看清楚再点。",
        "speakable": True,
        "dedupe_key": "danger_action|in_match|high|ron",
    }
    orchestrator.state.last_narration_delivery = "voice_candidate"
    orchestrator.state.last_narration_text = "这里像是有关键操作，我们先看清楚再点。"

    monkeypatch.setattr(
        orchestrator,
        "_bind_window",
        lambda: WindowBindingResult(bound=False, error="active window does not match keywords"),
    )

    result = await orchestrator.speak_last_narration()

    assert result.value["ok"] is False
    assert "not currently bound" in result.value["error"]
    assert plugin.messages == []


@pytest.mark.asyncio
async def test_run_loop_executes_cycle_before_first_sleep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    orchestrator.state.running = True

    calls = {"count": 0}

    def _fake_cycle() -> None:
        calls["count"] += 1
        orchestrator.state.running = False
        orchestrator.state.last_notification_at = now_iso()

    monkeypatch.setattr(orchestrator, "_run_live_cycle_locked", _fake_cycle)
    monkeypatch.setattr(orchestrator, "_get_sample_interval_ms", lambda: 999999)

    await orchestrator._run_loop()

    assert calls["count"] == 1
    assert plugin.statuses


@pytest.mark.asyncio
async def test_bind_window_auto_starts_session_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    orchestrator.state.runtime_mode = "standby"
    fake_overlay = _FakeOverlay()
    orchestrator._overlay = fake_overlay  # type: ignore[assignment]

    async def _fake_loop() -> None:
        return None

    monkeypatch.setattr(
        orchestrator,
        "_bind_window",
        lambda: WindowBindingResult(bound=True, window_title="Mahjong Soul", match_keyword="Mahjong Soul"),
    )
    monkeypatch.setattr(orchestrator, "_run_loop", _fake_loop)

    result = await orchestrator.bind_window(window_title="Mahjong Soul")

    assert result.value["bound"] is True
    assert result.value["auto_started"] is True
    assert result.value["selected_window_title"] == "Mahjong Soul"
    assert result.value["runtime_activated"] is True
    assert result.value["runtime_mode"] == "active"
    assert result.value["overlay"]["overlay_visible"] is True
    assert orchestrator.state.running is True
    assert orchestrator.state.runtime_mode == "active"
    assert orchestrator._task is not None
    assert fake_overlay.started == 1
    assert plugin.statuses[-1]["overlay_visible"] is True

    await orchestrator.stop()


@pytest.mark.asyncio
async def test_show_hide_overlay_controls_lifecycle(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    fake_overlay = _FakeOverlay()
    orchestrator._overlay = fake_overlay  # type: ignore[assignment]

    shown = await orchestrator.show_overlay()
    hidden = await orchestrator.hide_overlay()

    assert shown.value["ok"] is True
    assert shown.value["overlay_visible"] is True
    assert hidden.value["ok"] is True
    assert hidden.value["overlay_visible"] is False
    assert fake_overlay.started == 1
    assert fake_overlay.stopped == 1


def test_emit_status_updates_visible_overlay(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    fake_overlay = _FakeOverlay()
    fake_overlay.start()
    orchestrator._overlay = fake_overlay  # type: ignore[assignment]
    orchestrator._overlay_visible = True
    orchestrator.state.window_bound = True
    orchestrator.state.window_title = "Mahjong Soul"

    orchestrator._emit_status()

    assert fake_overlay.statuses
    assert fake_overlay.statuses[-1]["window_bound"] is True
    assert fake_overlay.statuses[-1]["window_title"] == "Mahjong Soul"


def test_overlay_refresh_button_runs_advice_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    fake_overlay = _FakeOverlay(["refresh_advice"])
    fake_overlay.start()
    orchestrator._overlay = fake_overlay  # type: ignore[assignment]
    orchestrator._overlay_visible = True
    orchestrator.state.running = True
    orchestrator.state.runtime_mode = "standby"
    calls: list[dict[str, object]] = []

    def _fake_pipeline(
        frame_path: str,
        *,
        capture: bool,
        dispatch: bool,
        force_reply: bool,
        persist_review_artifacts: bool | None = None,
    ) -> dict[str, object]:
        calls.append({
            "frame_path": frame_path,
            "capture": capture,
            "dispatch": dispatch,
            "force_reply": force_reply,
            "persist_review_artifacts": persist_review_artifacts,
        })
        orchestrator.state.last_decision_ok = True
        orchestrator.state.last_decision = {"suggestion": "优先考虑处理 1m。"}
        return {"ok": True, "stage": "fake_pipeline"}

    def _unexpected_live_cycle() -> None:
        raise AssertionError("manual overlay refresh should consume this runtime tick")

    monkeypatch.setattr(orchestrator, "_run_companion_pipeline_locked", _fake_pipeline)
    monkeypatch.setattr(orchestrator, "_run_live_cycle_locked", _unexpected_live_cycle)

    orchestrator._run_runtime_cycle_locked()

    assert calls == [{
        "frame_path": "",
        "capture": True,
        "dispatch": True,
        "force_reply": True,
        "persist_review_artifacts": True,
    }]
    assert orchestrator.state.last_runtime_command_source == "overlay"
    assert orchestrator.state.last_runtime_command_action == "refresh_advice"
    assert orchestrator.state.last_runtime_command_ok is True
    assert orchestrator.state.runtime_mode == "active"
    assert orchestrator.state.last_runtime_command_result["runtime_mode_before_refresh"] == "standby"


def test_stable_frame_skip_keeps_last_advice_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    frame_path = plugin.data_path("debug_samples") / "stable-frame.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"not a real image but enough for this state test")
    orchestrator.state.running = True
    orchestrator.state.last_perception_ok = True
    orchestrator.state.last_perception = {"scene": "in_match", "confidence": 0.8}
    orchestrator.state.last_decision_ok = True
    orchestrator.state.last_decision = {"suggestion": "继续保留两面，先处理孤张。"}
    orchestrator.state.last_narration_ok = True
    orchestrator.state.last_narration_text = "继续保留两面，先处理孤张。"
    orchestrator.state.last_narration = {"text": "继续保留两面，先处理孤张。"}

    class _FakeCaptureProvider:
        def capture_frame(self, *, samples_dir: Path, binding_result: WindowBindingResult, save_format: str) -> FramePacket:
            return FramePacket(
                timestamp_ms=1,
                image_path=str(frame_path),
                window_title="Mahjong Soul",
                width=2560,
                height=1440,
                source="fake",
            )

    orchestrator._capture_provider = _FakeCaptureProvider()  # type: ignore[assignment]
    monkeypatch.setattr(
        orchestrator,
        "_bind_window",
        lambda: WindowBindingResult(bound=True, window_title="Mahjong Soul", match_keyword="Mahjong Soul"),
    )
    monkeypatch.setattr(orchestrator, "_should_process_frame_locked", lambda _frame_path: False)

    orchestrator._run_live_cycle_locked()

    assert orchestrator.state.last_perception_ok is True


def test_discard_marker_clears_when_recommended_tile_region_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {
        "mahjong_companion": {
            "debug_samples": {"auto_prune_enabled": False},
        },
    }))
    live_dir = plugin.data_path("debug_samples", "live")
    live_dir.mkdir(parents=True, exist_ok=True)
    frame_one = live_dir / "marker-before-frame.png"
    frame_two = live_dir / "marker-after-frame.png"

    def _write_frame(path: Path, region_color: tuple[int, int, int]) -> None:
        image = Image.new("RGB", (120, 80), (24, 24, 24))
        image.paste(region_color, (10, 10, 34, 46))
        image.save(path)

    _write_frame(frame_one, (245, 245, 245))
    _write_frame(frame_two, (30, 30, 30))
    now = time.time()
    os.utime(frame_one, (now - 1, now - 1))
    os.utime(frame_two, (now, now))

    orchestrator.state.running = True
    orchestrator.state.last_frame_path = str(frame_one)
    orchestrator.state.window_left = 0
    orchestrator.state.window_top = 0
    perceived = PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=True,
        hand_tiles=["9m"],
        analysis_hints={
            "hand_tile_slots": [
                {
                    "slot_id": "hand_1",
                    "tile": "9m",
                    "index": 0,
                    "box": {"left": 10, "top": 10, "width": 24, "height": 36},
                },
            ],
        },
    )
    orchestrator.state.last_perception_ok = True
    orchestrator.state.last_perception = perceived.to_dict()
    orchestrator._apply_decision_result(DecisionResult(
        decision_type="tile_efficiency_hint",
        mahjong_analysis={"candidate_discards": [{"tile": "9m"}]},
    ))
    assert orchestrator.get_status()["screen_overlays"]

    class _FakeCaptureProvider:
        def capture_frame(self, *, samples_dir: Path, binding_result: WindowBindingResult, save_format: str) -> FramePacket:
            return FramePacket(
                timestamp_ms=2,
                image_path=str(frame_two),
                window_title="Mahjong Soul",
                width=120,
                height=80,
                source="fake",
            )

    orchestrator._capture_provider = _FakeCaptureProvider()  # type: ignore[assignment]
    monkeypatch.setattr(
        orchestrator,
        "_bind_window",
        lambda: WindowBindingResult(bound=True, window_title="Mahjong Soul", match_keyword="Mahjong Soul"),
    )
    monkeypatch.setattr(orchestrator, "_should_process_frame_locked", lambda _frame_path: False)

    orchestrator._run_live_cycle_locked()

    assert orchestrator.state.last_decision_ok is True
    assert orchestrator.get_status()["screen_overlays"] == []


def test_call_buttons_arm_fast_poll_and_force_frame_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {
        "mahjong_companion": {
            "sample_interval_ms": 700,
            "fast_poll": {
                "enabled": True,
                "interval_ms": 220,
                "duration_sec": 2,
                "force_process": True,
            },
        },
    }))

    orchestrator._apply_perception_result(PerceivedGameState(
        scene="in_match",
        confidence=0.9,
        is_user_turn=False,
        buttons=["chi"],
    ))

    assert orchestrator._get_current_sample_interval_ms() == 220
    monkeypatch.setattr(
        orchestrator._frame_change_gate,
        "evaluate",
        lambda *_args, **_kwargs: SimpleNamespace(should_process=False, reason="frame_unchanged"),
    )
    assert orchestrator._should_process_frame_locked(tmp_path / "stable-frame.png") is True


def test_debug_sample_prune_only_removes_old_live_frame_groups(tmp_path: Path) -> None:
    live_dir = tmp_path / "data" / "debug_samples" / "live"
    root_sample = tmp_path / "data" / "debug_samples" / "fixture-frame.png"
    now = 1_700_000_000.0
    root_sample.parent.mkdir(parents=True, exist_ok=True)
    root_sample.write_bytes(b"keep fixture")

    old_one = live_dir / "20260503-120000-000001-frame.png"
    old_two = live_dir / "20260503-120001-000001-frame.png"
    fresh = live_dir / "20260503-120002-000001-frame.png"
    for index, path in enumerate([old_one, old_two, fresh]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frame")
        overlay = path.with_suffix("").with_name(path.with_suffix("").name + "-overlay.json")
        overlay.write_text("{}", encoding="utf-8")
        timestamp = now - 100 + index
        if path == fresh:
            timestamp = now
        os.utime(path, (timestamp, timestamp))
        os.utime(overlay, (timestamp, timestamp))

    removed = _prune_debug_samples(live_dir, max_frames=1, max_age_sec=60, now=now)

    assert removed == 4
    assert root_sample.exists()
    assert not old_one.exists()
    assert not old_one.with_suffix("").with_name(old_one.with_suffix("").name + "-overlay.json").exists()
    assert not old_two.exists()
    assert fresh.exists()
    assert fresh.with_suffix("").with_name(fresh.with_suffix("").name + "-overlay.json").exists()


@pytest.mark.asyncio
async def test_public_pipeline_keeps_cached_state_clean_between_steps(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

    result = await orchestrator.analyze_frame_path(str(_sample_frame_path("20260415-050905-263947-frame.png")))
    assert result.value["ok"] is True
    assert orchestrator.state.last_perception_ok is True

    decision = await orchestrator.generate_decision()
    assert decision.value["ok"] is True
    assert decision.value["decision_type"] == "waiting_state"
    assert orchestrator.state.last_decision_ok is True

    narration = await orchestrator.generate_narration()
    assert narration.value["ok"] is True
    assert narration.value["event_type"] == "waiting_state"
    assert orchestrator.state.last_narration_ok is True

    preview = await orchestrator.preview_companion_view()
    assert preview.value["ok"] is True
    assert preview.value["data"]["headline"]
    assert preview.value["data"]["delivery"] == "silent_ui"


@pytest.mark.asyncio
async def test_run_companion_pipeline_can_force_debug_reply_from_image(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

    result = await orchestrator.run_companion_pipeline(
        frame_path=str(_sample_frame_path("20260415-050905-263947-frame.png")),
        dispatch=True,
        force_reply=True,
    )

    assert result.value["ok"] is True
    assert result.value["perception"]["ok"] is True
    assert result.value["decision"]["ok"] is True
    assert result.value["narration"]["ok"] is True
    assert result.value["dispatch"]["ok"] is True
    assert result.value["dispatch"]["delivery"] == "proactive_notification"
    assert plugin.messages
    assert plugin.messages[0]["content"] == result.value["narration"]["text"]


@pytest.mark.asyncio
async def test_manual_pipeline_does_not_stage_review_or_memory_bridge_when_session_not_running(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

    manual_frame = plugin.data_path("debug_samples") / "manual-frame.png"
    manual_frame.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_sample_frame_path("20260415-071312-089224-frame.png"), manual_frame)

    result = await orchestrator.run_companion_pipeline(
        frame_path=str(manual_frame),
        dispatch=False,
        force_reply=False,
    )

    assert result.value["ok"] is True
    assert "review_candidates_path" not in result.value["decision"]
    assert "memory_bridge" not in result.value["decision"]
    assert not (plugin.data_path("session_cache") / "review_candidates.json").exists()
    assert not (plugin.data_path("session_cache") / "memory_bridge_queue.json").exists()


def test_pipeline_can_explicitly_persist_review_artifacts_for_overlay_refresh(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

    manual_frame = plugin.data_path("debug_samples") / "overlay-refresh-frame.png"
    manual_frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(20, 30, 40)).save(manual_frame)
    perceived = PerceivedGameState(
        scene="in_match",
        confidence=0.91,
        is_user_turn=True,
        buttons=["ron"],
        notes=["overlay refresh fixture"],
    )
    orchestrator._perception_adapter = SimpleNamespace(  # type: ignore[assignment]
        analyze=lambda _path: (perceived, {"source": "test"})
    )
    orchestrator._decision_adapter = SimpleNamespace(  # type: ignore[assignment]
        suggest=lambda _perceived: DecisionResult(
            decision_type="action_available",
            priority=88,
            risk_level="high",
            action_required=True,
            summary="可复盘的悬浮窗刷新建议",
            detail="用户主动刷新时，这一轮建议应进入复盘候选。",
            suggestion="先确认和牌窗口。",
            scene="in_match",
            buttons=["ron"],
            review_tags=["overlay_refresh", "win_window"],
        )
    )

    result = orchestrator._run_companion_pipeline_locked(
        str(manual_frame),
        capture=False,
        dispatch=False,
        force_reply=False,
        persist_review_artifacts=True,
    )

    assert result["ok"] is True
    assert "review_candidates_path" in result["decision"]
    assert Path(result["decision"]["review_candidates_path"]).exists()


@pytest.mark.asyncio
async def test_manual_pipeline_rejects_frame_outside_debug_samples(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

    manual_frame = tmp_path / "manual-frame.png"
    shutil.copyfile(_sample_frame_path("20260415-071312-089224-frame.png"), manual_frame)

    result = await orchestrator.run_companion_pipeline(
        frame_path=str(manual_frame),
        dispatch=False,
        force_reply=False,
    )

    assert result.value["ok"] is False
    assert "debug_samples" in result.value["error"]
