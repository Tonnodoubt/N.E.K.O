from __future__ import annotations

from plugin.plugins.mahjong_companion.action.human_override_guard import HumanOverrideGuard


class _FocusProvider:
    def __init__(self, focused: bool) -> None:
        self.focused = focused

    def is_target_window_focused(self, expected_title: str) -> bool:
        return self.focused


def test_human_override_guard_aborts_when_window_unfocused() -> None:
    guard = HumanOverrideGuard(
        pointer_provider=lambda: (100, 100),
        focus_provider=_FocusProvider(False),
    )
    guard.arm(
        enabled=True,
        active_window_sec=1.0,
        movement_threshold_px=18,
        expected_window_title="Mahjong Soul",
        check_window_focus=True,
        now_monotonic=1.0,
    )

    decision = guard.evaluate(pointer=(100, 100), now_monotonic=1.1)

    assert decision.should_abort is True
    assert decision.reason == "window_unfocused"


def test_human_override_guard_keeps_pointer_protection_when_focused() -> None:
    guard = HumanOverrideGuard(
        pointer_provider=lambda: (100, 100),
        focus_provider=_FocusProvider(True),
    )
    guard.arm(
        enabled=True,
        active_window_sec=1.0,
        movement_threshold_px=18,
        expected_window_title="Mahjong Soul",
        check_window_focus=True,
        now_monotonic=1.0,
    )

    decision = guard.evaluate(pointer=(140, 100), now_monotonic=1.1)

    assert decision.should_abort is True
    assert decision.reason == "human_override_detected"
