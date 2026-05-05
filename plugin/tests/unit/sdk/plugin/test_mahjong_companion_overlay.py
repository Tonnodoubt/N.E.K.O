from __future__ import annotations

import queue

import plugin.plugins.mahjong_companion.overlay as overlay_module
from plugin.plugins.mahjong_companion.overlay import (
    CompanionOverlay,
    _DragState,
    _advice_text,
    _overlay_platform_supported,
    _parse_marker_box,
)
from plugin.plugins.mahjong_companion.tile_labels import format_tile_label


def test_advice_text_prefers_call_decision_over_discard_candidate() -> None:
    status = {
        "runtime_mode": "active",
        "window_bound": True,
        "last_decision_type": "action_available",
        "last_narration_text": "这口先判断要不要吃碰，再决定是否副露。",
        "last_decision": {
            "decision_type": "action_available",
            "recommended_focus": "call_decision",
            "suggestion": "这口先判断要不要吃碰，再决定是否副露。",
            "mahjong_analysis": {
                "candidate_discards": [
                    {
                        "tile": "9m",
                        "reason": "孤张边张，改良偏弱。",
                        "ukeire_estimate": 4,
                    },
                ],
            },
        },
    }

    assert _advice_text(status) == "这口先判断要不要吃碰，再决定是否副露。"


def test_advice_text_keeps_discard_hint_priority_for_tile_efficiency() -> None:
    status = {
        "runtime_mode": "active",
        "window_bound": True,
        "last_decision": {
            "decision_type": "tile_efficiency_hint",
            "recommended_focus": "tile_efficiency",
            "suggestion": "先别急着副露。",
            "mahjong_analysis": {
                "candidate_discards": [
                    {
                        "tile": "9m",
                        "reason": "孤张边张，改良偏弱。",
                        "ukeire_estimate": 4,
                    },
                ],
            },
        },
    }

    advice = _advice_text(status)

    assert format_tile_label("9m") in advice
    assert "改良偏弱" in advice


def test_advice_text_does_not_show_discard_candidate_outside_tile_efficiency_hint() -> None:
    status = {
        "runtime_mode": "active",
        "window_bound": True,
        "last_narration_text": "先看最新摸牌、牌河和副露，再决定下一步。",
        "last_decision": {
            "decision_type": "scene_update",
            "recommended_focus": "turn_observe",
            "suggestion": "先看最新摸牌、牌河和副露，再决定下一步。",
            "mahjong_analysis": {
                "candidate_discards": [
                    {
                        "tile": "9m",
                        "reason": "孤张边张，改良偏弱。",
                        "ukeire_estimate": 4,
                    },
                ],
            },
        },
    }

    assert _advice_text(status) == "先看最新摸牌、牌河和副露，再决定下一步。"


def test_advice_text_mentions_recognized_hand_count_while_waiting_for_turn_shape() -> None:
    status = {
        "runtime_mode": "active",
        "window_bound": True,
        "last_perception": {
            "analysis_hints": {
                "recognized_hand_tile_count": 11,
            },
        },
        "last_decision": {},
    }

    assert "已识别到 11 张手牌" in _advice_text(status)


def test_parse_marker_box_accepts_screen_overlay_box() -> None:
    assert _parse_marker_box({"left": 100, "top": 200, "width": 64, "height": 92}) == {
        "left": 100,
        "top": 200,
        "width": 64,
        "height": 92,
    }
    assert _parse_marker_box({"left": 100, "top": 200, "width": 0, "height": 92}) == {}


def test_drag_state_is_attribute_based() -> None:
    state = _DragState()

    state.value = True
    state.dx = 12
    state.dy = 34

    assert state.value is True
    assert state.dx == 12
    assert state.dy == 34


def test_overlay_platform_supported_on_gui_platforms(monkeypatch) -> None:
    monkeypatch.setattr(overlay_module.platform, "system", lambda: "Darwin")
    assert _overlay_platform_supported() is True

    monkeypatch.setattr(overlay_module.platform, "system", lambda: "Windows")
    assert _overlay_platform_supported() is True

    monkeypatch.setattr(overlay_module.platform, "system", lambda: "Linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert _overlay_platform_supported() is False

    monkeypatch.setenv("DISPLAY", ":0")
    assert _overlay_platform_supported() is True


def test_companion_overlay_starts_process_and_sends_latest_status(monkeypatch) -> None:
    monkeypatch.setattr(overlay_module, "_overlay_platform_supported", lambda: True)
    context = _FakeContext()
    overlay = CompanionOverlay(mp_context=context)

    assert overlay.start() is True
    overlay.update_status({"primary": "first"})
    overlay.update_status({"primary": "second"})

    assert context.process is not None
    assert context.process.started is True
    assert context.status_queue is not None
    assert context.status_queue.get_nowait() == {"primary": "first"}
    assert context.status_queue.get_nowait() == {"primary": "second"}

    overlay.stop()

    assert context.process.terminated is True


def test_companion_overlay_drains_commands_and_logs_child_errors(monkeypatch) -> None:
    monkeypatch.setattr(overlay_module, "_overlay_platform_supported", lambda: True)
    context = _FakeContext()
    logger = _FakeLogger()
    overlay = CompanionOverlay(logger=logger, mp_context=context)

    assert overlay.start() is True
    context.command_queue.put_nowait("hide_overlay")
    context.command_queue.put_nowait({"__control__": "error", "error": "tk failed"})

    assert overlay.drain_commands() == ["hide_overlay"]
    assert logger.warnings


class _FakeContext:
    def __init__(self) -> None:
        self.status_queue: queue.Queue | None = None
        self.command_queue: queue.Queue | None = None
        self.process: _FakeProcess | None = None

    def Queue(self, maxsize: int = 0) -> queue.Queue:
        created: queue.Queue = queue.Queue(maxsize=maxsize)
        if self.status_queue is None:
            self.status_queue = created
        else:
            self.command_queue = created
        return created

    def Process(self, *, target, args, name: str) -> "_FakeProcess":
        self.process = _FakeProcess(target=target, args=args, name=name)
        return self.process


class _FakeProcess:
    def __init__(self, *, target, args, name: str) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = False
        self.started = False
        self.terminated = False
        self._alive = False

    def start(self) -> None:
        self.started = True
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | int | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, tuple[object, ...]]] = []

    def warning(self, message: str, *args: object) -> None:
        self.warnings.append((message, args))
