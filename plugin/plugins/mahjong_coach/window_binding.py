from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any


IGNORED_WINDOW_TITLE_FRAGMENTS = (
    "Mahjong Coach",
    "Mahjong Coach Overlay",
    "雀魂陪伴",
    "牌局建议",
    "NEKO 牌局建议",
    "store.steampowered.com",
    "steamcommunity.com",
    "Steam",
)
MIN_GAME_WINDOW_WIDTH = 640
MIN_GAME_WINDOW_HEIGHT = 360


@dataclass
class WindowBindingResult:
    bound: bool
    window_title: str = ""
    app_name: str = ""
    match_keyword: str = ""
    source: str = ""
    error: str = ""
    hwnd: int | None = None
    left: int | None = None
    top: int | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_bounds(self) -> bool:
        return (
            isinstance(self.left, int)
            and isinstance(self.top, int)
            and isinstance(self.width, int)
            and self.width > 0
            and isinstance(self.height, int)
            and self.height > 0
        )


@dataclass
class WindowCandidate:
    title: str
    app_name: str = ""
    source: str = ""
    is_active: bool = False
    match_keyword: str = ""
    matches_keywords: bool = False
    hwnd: int | None = None
    left: int | None = None
    top: int | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bind_window_from_keywords(keywords: list[str]) -> WindowBindingResult:
    cleaned = _clean_keywords(keywords)
    if cleaned and platform.system().lower() == "windows":
        matched = _find_matching_window_windows(cleaned)
        if matched is not None:
            return matched

    probe = get_active_window_info()
    if not probe.window_title and not probe.app_name:
        return WindowBindingResult(
            bound=False,
            source=probe.source,
            error=probe.error or "no active window info available",
        )
    ignored = _window_should_be_ignored(probe.window_title, probe.app_name, probe.width, probe.height)
    match = _match_keyword(probe.window_title, probe.app_name, cleaned)
    if (not cleaned or match) and not ignored:
        probe.bound = True
        probe.match_keyword = match
        probe.error = ""
        return probe
    return WindowBindingResult(
        bound=False,
        window_title=probe.window_title,
        app_name=probe.app_name,
        source=probe.source,
        error="active window does not match keywords" if not ignored else "active window is ignored",
        hwnd=probe.hwnd,
        left=probe.left,
        top=probe.top,
        width=probe.width,
        height=probe.height,
    )


def list_window_candidates(keywords: list[str]) -> list[dict[str, Any]]:
    cleaned = _clean_keywords(keywords)
    if platform.system().lower() == "windows":
        candidates: list[WindowCandidate] = []
        for window in _all_windows_windows():
            try:
                if bool(getattr(window, "isMinimized", False)):
                    continue
                candidate = _window_candidate_from_object(window, cleaned)
            except Exception:
                candidate = None
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                item.matches_keywords,
                item.is_active,
                int(item.width or 0) * int(item.height or 0),
            ),
            reverse=True,
        )
        return [candidate.to_dict() for candidate in candidates[:20]]

    active = get_active_window_info()
    if not active.window_title and not active.app_name:
        return []
    match = _match_keyword(active.window_title, active.app_name, cleaned)
    return [
        WindowCandidate(
            title=active.window_title or active.app_name,
            app_name=active.app_name,
            source=active.source,
            is_active=True,
            match_keyword=match,
            matches_keywords=bool(match),
            hwnd=active.hwnd,
            left=active.left,
            top=active.top,
            width=active.width,
            height=active.height,
        ).to_dict()
    ]


def get_active_window_info() -> WindowBindingResult:
    system = platform.system().lower()
    if system == "windows":
        return _get_active_window_windows()
    if system == "darwin":
        return _get_active_window_macos()
    if system == "linux":
        return _get_active_window_linux()
    return WindowBindingResult(bound=False, source="unknown", error=f"unsupported platform: {system}")


def _find_matching_window_windows(keywords: list[str]) -> WindowBindingResult | None:
    candidates: list[tuple[tuple[int, int], Any, WindowBindingResult]] = []
    for window in _all_windows_windows():
        try:
            if bool(getattr(window, "isMinimized", False)):
                continue
            title = _normalize_title(getattr(window, "title", "") or "")
            if not title:
                continue
            left, top, width, height = _window_geometry_from_object(window)
            if _window_should_be_ignored(title, "", width, height):
                continue
            match = _match_keyword(title, "", keywords)
            if not match:
                continue
            result = WindowBindingResult(
                bound=True,
                window_title=title,
                match_keyword=match,
                source="pygetwindow-all",
                hwnd=_window_hwnd_from_object(window),
                left=left,
                top=top,
                width=width if width and width > 0 else None,
                height=height if height and height > 0 else None,
            )
            candidates.append((_window_candidate_score(window, result), window, result))
        except Exception:
            continue
    if not candidates:
        return None
    _score, window, result = max(candidates, key=lambda item: item[0])
    _activate_window_best_effort(window)
    return result


def _get_active_window_windows() -> WindowBindingResult:
    try:
        import pygetwindow as gw  # type: ignore[import-not-found]
    except Exception as exc:
        return WindowBindingResult(bound=False, source="pygetwindow", error=f"pygetwindow unavailable: {exc}")
    try:
        active = gw.getActiveWindow()
        if active is None:
            return WindowBindingResult(bound=False, source="pygetwindow", error="no active window")
        title = _normalize_title(getattr(active, "title", "") or "")
        left, top, width, height = _window_geometry_from_object(active)
        return WindowBindingResult(
            bound=False,
            window_title=title,
            source="pygetwindow",
            hwnd=_window_hwnd_from_object(active),
            left=left,
            top=top,
            width=width if width and width > 0 else None,
            height=height if height and height > 0 else None,
        )
    except Exception as exc:
        return WindowBindingResult(bound=False, source="pygetwindow", error=str(exc))


def _get_active_window_macos() -> WindowBindingResult:
    if shutil.which("osascript") is None:
        return WindowBindingResult(bound=False, source="osascript", error="osascript unavailable")
    script = r'''
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    set appName to name of frontApp
    set winTitle to ""
    set winPos to "0,0"
    set winSize to "0,0"
    try
        set winTitle to name of front window of frontApp
        set winPosList to position of front window of frontApp
        set winSizeList to size of front window of frontApp
        set winPos to (item 1 of winPosList as text) & "," & (item 2 of winPosList as text)
        set winSize to (item 1 of winSizeList as text) & "," & (item 2 of winSizeList as text)
    end try
    return appName & "||" & winTitle & "||" & winPos & "||" & winSize
end tell
'''
    try:
        output = subprocess.check_output(["osascript", "-e", script], text=True, stderr=subprocess.STDOUT, timeout=3.0).strip()
        parts = output.split("||")
        app_name = _normalize_title(parts[0]) if len(parts) > 0 else ""
        title = _normalize_title(parts[1]) if len(parts) > 1 else ""
        left, top = _split_pair(parts[2] if len(parts) > 2 else "0,0")
        width, height = _split_pair(parts[3] if len(parts) > 3 else "0,0")
        return WindowBindingResult(bound=False, window_title=title, app_name=app_name, source="osascript", left=left, top=top, width=width, height=height)
    except Exception as exc:
        return WindowBindingResult(bound=False, source="osascript", error=str(exc))


def _get_active_window_linux() -> WindowBindingResult:
    if shutil.which("xdotool") is None:
        return WindowBindingResult(bound=False, source="xdotool", error="xdotool unavailable")
    try:
        window_id = subprocess.check_output(["xdotool", "getactivewindow"], text=True, stderr=subprocess.STDOUT, timeout=3.0).strip()
        title = subprocess.check_output(["xdotool", "getwindowname", window_id], text=True, stderr=subprocess.STDOUT, timeout=3.0).strip()
        geometry_raw = subprocess.check_output(["xdotool", "getwindowgeometry", "--shell", window_id], text=True, stderr=subprocess.STDOUT, timeout=3.0)
        geometry: dict[str, int] = {}
        for line in geometry_raw.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if value.strip().lstrip("-").isdigit():
                    geometry[key.strip()] = int(value)
        return WindowBindingResult(bound=False, window_title=_normalize_title(title), source="xdotool", left=geometry.get("X"), top=geometry.get("Y"), width=geometry.get("WIDTH"), height=geometry.get("HEIGHT"))
    except Exception as exc:
        return WindowBindingResult(bound=False, source="xdotool", error=str(exc))


def _all_windows_windows() -> list[Any]:
    try:
        import pygetwindow as gw  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        return list(gw.getAllWindows())
    except Exception:
        return []


def _window_candidate_from_object(window: Any, keywords: list[str]) -> WindowCandidate | None:
    title = _normalize_title(getattr(window, "title", "") or "")
    if not title:
        return None
    left, top, width, height = _window_geometry_from_object(window)
    if _window_should_be_ignored(title, "", width, height):
        return None
    match = _match_keyword(title, "", keywords)
    return WindowCandidate(
        title=title,
        source="pygetwindow-all",
        is_active=bool(getattr(window, "isActive", False)),
        match_keyword=match,
        matches_keywords=bool(match),
        hwnd=_window_hwnd_from_object(window),
        left=left,
        top=top,
        width=width if width and width > 0 else None,
        height=height if height and height > 0 else None,
    )


def _normalize_title(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _clean_keywords(keywords: list[str]) -> list[str]:
    return [str(item).strip() for item in keywords if str(item).strip()]


def _match_keyword(window_title: str, app_name: str, keywords: list[str]) -> str:
    haystack = f"{window_title} {app_name}".lower()
    for keyword in keywords:
        if keyword.lower() in haystack:
            return keyword
    return ""


def _window_should_be_ignored(window_title: str, app_name: str, width: int | None, height: int | None) -> bool:
    haystack = f"{window_title} {app_name}".casefold()
    if any(fragment.casefold() in haystack for fragment in IGNORED_WINDOW_TITLE_FRAGMENTS):
        return True
    return (
        isinstance(width, int)
        and isinstance(height, int)
        and width > 0
        and height > 0
        and (width < MIN_GAME_WINDOW_WIDTH or height < MIN_GAME_WINDOW_HEIGHT)
    )


def _window_geometry_from_object(window: Any) -> tuple[int | None, int | None, int | None, int | None]:
    return (_coerce_int(getattr(window, "left", None)), _coerce_int(getattr(window, "top", None)), _coerce_int(getattr(window, "width", None)), _coerce_int(getattr(window, "height", None)))


def _window_hwnd_from_object(window: Any) -> int | None:
    hwnd = _coerce_int(getattr(window, "_hWnd", None))
    return hwnd if hwnd and hwnd > 0 else None


def _window_candidate_score(window: Any, result: WindowBindingResult) -> tuple[int, int]:
    active_score = 1000 if bool(getattr(window, "isActive", False)) else 0
    bounds_score = 100 if result.has_bounds() else 0
    area = int(result.width or 0) * int(result.height or 0)
    return active_score + bounds_score, area


def _activate_window_best_effort(window: Any) -> None:
    try:
        if bool(getattr(window, "isMinimized", False)) and hasattr(window, "restore"):
            window.restore()
    except Exception:
        pass
    try:
        if bool(getattr(window, "isActive", False)):
            return
    except Exception:
        pass
    try:
        if hasattr(window, "activate"):
            window.activate()
            time.sleep(0.08)
    except Exception:
        pass


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_pair(value: str) -> tuple[int | None, int | None]:
    left, _, right = str(value).partition(",")
    return _coerce_int(left.strip()), _coerce_int(right.strip())
