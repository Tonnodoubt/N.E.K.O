from __future__ import annotations

import sys
from types import SimpleNamespace

from plugin.plugins.mahjong_companion import window_binding


class _FakeWindow:
    def __init__(
        self,
        title: str,
        *,
        active: bool = False,
        minimized: bool = False,
        left: int = 0,
        top: int = 0,
        width: int = 1280,
        height: int = 720,
        hwnd: int = 12345,
    ) -> None:
        self.title = title
        self.isActive = active
        self.isMinimized = minimized
        self._hWnd = hwnd
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.activate_calls = 0
        self.restore_calls = 0

    def activate(self) -> None:
        self.activate_calls += 1

    def restore(self) -> None:
        self.restore_calls += 1
        self.isMinimized = False


def test_bind_window_from_keywords_finds_matching_windows_window_when_plugin_ui_is_active(monkeypatch) -> None:
    active_plugin_window = _FakeWindow("插件详情 - N.E.K.O 插件管理", active=True, width=1280, height=900)
    majsoul_window = _FakeWindow("Mahjong Soul", left=20, top=30, width=1920, height=1080)
    fake_gw = SimpleNamespace(
        getActiveWindow=lambda: active_plugin_window,
        getAllWindows=lambda: [active_plugin_window, majsoul_window],
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.setattr(window_binding.platform, "system", lambda: "Windows")
    monkeypatch.setattr(window_binding.time, "sleep", lambda _seconds: None)

    result = window_binding.bind_window_from_keywords(["雀魂", "Mahjong Soul"])

    assert result.bound is True
    assert result.window_title == "Mahjong Soul"
    assert result.match_keyword == "Mahjong Soul"
    assert result.source == "pygetwindow-all"
    assert result.hwnd == 12345
    assert result.left == 20
    assert result.top == 30
    assert result.width == 1920
    assert result.height == 1080
    assert majsoul_window.activate_calls == 1


def test_bind_window_from_keywords_reports_active_window_when_no_window_matches(monkeypatch) -> None:
    active_plugin_window = _FakeWindow("插件详情 - N.E.K.O 插件管理", active=True, width=1280, height=900)
    fake_gw = SimpleNamespace(
        getActiveWindow=lambda: active_plugin_window,
        getAllWindows=lambda: [active_plugin_window],
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.setattr(window_binding.platform, "system", lambda: "Windows")

    result = window_binding.bind_window_from_keywords(["Mahjong Soul"])

    assert result.bound is False
    assert result.window_title == "插件详情 - N.E.K.O 插件管理"
    assert result.error == "active window does not match keywords"


def test_list_window_candidates_marks_keyword_matches(monkeypatch) -> None:
    active_plugin_window = _FakeWindow("插件详情 - N.E.K.O 插件管理", active=True, width=1280, height=900)
    majsoul_window = _FakeWindow("雀魂麻將", left=20, top=30, width=1920, height=1080, hwnd=67890)
    fake_gw = SimpleNamespace(
        getActiveWindow=lambda: active_plugin_window,
        getAllWindows=lambda: [active_plugin_window, majsoul_window],
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.setattr(window_binding.platform, "system", lambda: "Windows")

    candidates = window_binding.list_window_candidates(["雀魂", "Mahjong Soul"])

    assert candidates[0]["title"] == "雀魂麻將"
    assert candidates[0]["matches_keywords"] is True
    assert candidates[0]["match_keyword"] == "雀魂"
    assert candidates[0]["hwnd"] == 67890


def test_bind_window_from_title_uses_selected_window(monkeypatch) -> None:
    active_plugin_window = _FakeWindow("插件详情 - N.E.K.O 插件管理", active=True, width=1280, height=900)
    majsoul_window = _FakeWindow("雀魂麻將", left=20, top=30, width=1920, height=1080, hwnd=67890)
    fake_gw = SimpleNamespace(
        getActiveWindow=lambda: active_plugin_window,
        getAllWindows=lambda: [active_plugin_window, majsoul_window],
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.setattr(window_binding.platform, "system", lambda: "Windows")
    monkeypatch.setattr(window_binding.time, "sleep", lambda _seconds: None)

    result = window_binding.bind_window_from_title("雀魂麻將", ["雀魂", "Mahjong Soul"])

    assert result.bound is True
    assert result.window_title == "雀魂麻將"
    assert result.source == "pygetwindow-selected"
    assert result.hwnd == 67890
    assert majsoul_window.activate_calls == 1


def test_bind_window_ignores_companion_overlay_even_when_active(monkeypatch) -> None:
    window_binding._ACTIVATION_LAST_AT.clear()
    overlay_window = _FakeWindow("雀魂陪伴", active=True, width=376, height=217)
    majsoul_window = _FakeWindow("雀魂麻將", left=20, top=30, width=1920, height=1080)
    fake_gw = SimpleNamespace(
        getActiveWindow=lambda: overlay_window,
        getAllWindows=lambda: [overlay_window, majsoul_window],
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.setattr(window_binding.platform, "system", lambda: "Windows")
    monkeypatch.setattr(window_binding.time, "sleep", lambda _seconds: None)

    result = window_binding.bind_window_from_keywords(["雀魂", "Mahjong Soul"])
    candidates = window_binding.list_window_candidates(["雀魂", "Mahjong Soul"])

    assert result.bound is True
    assert result.window_title == "雀魂麻將"
    assert majsoul_window.activate_calls == 1
    assert all(candidate["title"] != "雀魂陪伴" for candidate in candidates)


def test_bind_window_rejects_selected_companion_overlay(monkeypatch) -> None:
    overlay_window = _FakeWindow("NEKO 牌局建议", active=True, width=376, height=217)
    fake_gw = SimpleNamespace(
        getActiveWindow=lambda: overlay_window,
        getAllWindows=lambda: [overlay_window],
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake_gw)
    monkeypatch.setattr(window_binding.platform, "system", lambda: "Windows")

    result = window_binding.bind_window_from_title("NEKO 牌局建议", ["雀魂", "Mahjong Soul"])

    assert result.bound is False
    assert "not a viable Mahjong Soul game window" in result.error
