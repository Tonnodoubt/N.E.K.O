from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

IGNORED_WINDOW_TITLE_FRAGMENTS = (
    "雀魂陪伴",
    "牌局建议",
    "NEKO 牌局建议",
    "Mahjong Companion Overlay",
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
    hwnd: Optional[int] = None
    left: Optional[int] = None
    top: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

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
    hwnd: Optional[int] = None
    left: Optional[int] = None
    top: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _is_ignored_window_title(window_title: str, app_name: str = "") -> bool:
    haystack = f"{window_title} {app_name}".casefold()
    return any(fragment.casefold() in haystack for fragment in IGNORED_WINDOW_TITLE_FRAGMENTS)


def _bounds_too_small(width: Optional[int], height: Optional[int]) -> bool:
    return (
        isinstance(width, int)
        and isinstance(height, int)
        and width > 0
        and height > 0
        and (width < MIN_GAME_WINDOW_WIDTH or height < MIN_GAME_WINDOW_HEIGHT)
    )


def _window_should_be_ignored(window_title: str, app_name: str, width: Optional[int], height: Optional[int]) -> bool:
    return _is_ignored_window_title(window_title, app_name) or _bounds_too_small(width, height)


def _window_geometry_from_object(window: Any) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return (
        _coerce_int(getattr(window, "left", None)),
        _coerce_int(getattr(window, "top", None)),
        _coerce_int(getattr(window, "width", None)),
        _coerce_int(getattr(window, "height", None)),
    )


def _window_hwnd_from_object(window: Any) -> Optional[int]:
    try:
        hwnd = int(getattr(window, "_hWnd", 0) or 0)
    except (TypeError, ValueError):
        return None
    return hwnd if hwnd > 0 else None


def _window_candidate_score(window: Any, result: WindowBindingResult) -> tuple[int, int]:
    active_score = 1000 if bool(getattr(window, "isActive", False)) else 0
    bounds_score = 100 if result.has_bounds() else 0
    area = int(result.width or 0) * int(result.height or 0)
    return active_score + bounds_score, area


def _window_candidate_from_object(window: Any, keywords: list[str]) -> WindowCandidate | None:
    title = _normalize_title(getattr(window, "title", "") or "")
    if not title:
        return None
    left, top, width, height = _window_geometry_from_object(window)
    if _is_ignored_window_title(title, ""):
        return None
    match_keyword = _match_keyword(title, "", keywords)
    return WindowCandidate(
        title=title,
        source="pygetwindow-all",
        is_active=bool(getattr(window, "isActive", False)),
        match_keyword=match_keyword,
        matches_keywords=bool(match_keyword),
        hwnd=_window_hwnd_from_object(window),
        left=left,
        top=top,
        width=width if width and width > 0 else None,
        height=height if height and height > 0 else None,
    )


# Activation throttle: fast-poll loops call _bind_window every ~300ms; the old
# implementation slept 80ms per call even when the target window was already
# focused (CODE_REVIEW_v1.2 N-M4). We now skip when isActive and rate-limit
# unconditional activate() to once per ACTIVATION_THROTTLE_SECONDS per window.
ACTIVATION_THROTTLE_SECONDS = 1.0
_ACTIVATION_LAST_AT: dict[int, float] = {}


def _window_throttle_key(window: Any) -> int:
    hwnd = _window_hwnd_from_object(window)
    if isinstance(hwnd, int) and hwnd:
        return hwnd
    return id(window)


def _activate_window_best_effort(window: Any) -> None:
    try:
        if bool(getattr(window, "isMinimized", False)) and hasattr(window, "restore"):
            window.restore()
    except Exception:
        pass
    # Already focused? Nothing to do — don't burn 80ms on a sleep.
    try:
        if bool(getattr(window, "isActive", False)):
            return
    except Exception:
        pass
    # Throttle repeated activate() calls on the same window.
    key = _window_throttle_key(window)
    now = time.monotonic()
    last = _ACTIVATION_LAST_AT.get(key, 0.0)
    if now - last < ACTIVATION_THROTTLE_SECONDS:
        return
    _ACTIVATION_LAST_AT[key] = now
    try:
        if hasattr(window, "activate"):
            window.activate()
            time.sleep(0.08)
    except Exception:
        pass


def _all_windows_windows() -> list[Any]:
    try:
        import pygetwindow as gw  # type: ignore[import-not-found]
    except Exception:
        return []

    try:
        return list(gw.getAllWindows())
    except Exception:
        return []


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
            match_keyword = _match_keyword(title, "", keywords)
            if not match_keyword:
                continue
            result = WindowBindingResult(
                bound=True,
                window_title=title,
                match_keyword=match_keyword,
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


def list_window_candidates(keywords: list[str]) -> list[dict[str, Any]]:
    cleaned_keywords = _clean_keywords(keywords)
    system = platform.system().lower()
    if system == "windows":
        candidates: list[WindowCandidate] = []
        seen_titles: set[str] = set()
        for window in _all_windows_windows():
            try:
                if bool(getattr(window, "isMinimized", False)):
                    continue
                candidate = _window_candidate_from_object(window, cleaned_keywords)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            key = candidate.title.casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                item.matches_keywords,
                item.is_active,
                int(item.width or 0) * int(item.height or 0),
                item.title.casefold(),
            ),
            reverse=True,
        )
        return [candidate.to_dict() for candidate in candidates]

    active = get_active_window_info()
    if not active.window_title and not active.app_name:
        return []
    match_keyword = _match_keyword(active.window_title, active.app_name, cleaned_keywords)
    return [
        WindowCandidate(
            title=active.window_title or active.app_name,
            app_name=active.app_name,
            source=active.source,
            is_active=True,
            match_keyword=match_keyword,
            matches_keywords=bool(match_keyword),
            hwnd=active.hwnd,
            left=active.left,
            top=active.top,
            width=active.width,
            height=active.height,
        ).to_dict(),
    ]


def bind_window_from_title(window_title: str, keywords: list[str] | None = None) -> WindowBindingResult:
    target_title = _normalize_title(window_title)
    cleaned_keywords = _clean_keywords(keywords or [])
    if not target_title:
        return WindowBindingResult(bound=False, error="window_title is required")

    system = platform.system().lower()
    if system == "windows":
        candidates: list[tuple[tuple[int, int], Any, WindowBindingResult]] = []
        target_folded = target_title.casefold()
        for window in _all_windows_windows():
            try:
                title = _normalize_title(getattr(window, "title", "") or "")
                if title.casefold() != target_folded:
                    continue
                left, top, width, height = _window_geometry_from_object(window)
                if _window_should_be_ignored(title, "", width, height):
                    return WindowBindingResult(
                        bound=False,
                        window_title=title,
                        source="pygetwindow-selected",
                        error="selected window is not a viable Mahjong Soul game window",
                        hwnd=_window_hwnd_from_object(window),
                        left=left,
                        top=top,
                        width=width,
                        height=height,
                    )
                result = WindowBindingResult(
                    bound=True,
                    window_title=title,
                    match_keyword=_match_keyword(title, "", cleaned_keywords),
                    source="pygetwindow-selected",
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
            return WindowBindingResult(
                bound=False,
                window_title=target_title,
                source="pygetwindow-selected",
                error="selected window not found",
            )

        _score, window, result = max(candidates, key=lambda item: item[0])
        _activate_window_best_effort(window)
        return result

    probe = get_active_window_info()
    if _normalize_title(probe.window_title or probe.app_name).casefold() == target_title.casefold():
        if _window_should_be_ignored(probe.window_title, probe.app_name, probe.width, probe.height):
            return WindowBindingResult(
                bound=False,
                window_title=probe.window_title,
                app_name=probe.app_name,
                source=f"{probe.source}-selected",
                error="selected window is not a viable Mahjong Soul game window",
                hwnd=probe.hwnd,
                left=probe.left,
                top=probe.top,
                width=probe.width,
                height=probe.height,
            )
        return WindowBindingResult(
            bound=True,
            window_title=probe.window_title,
            app_name=probe.app_name,
            match_keyword=_match_keyword(probe.window_title, probe.app_name, cleaned_keywords),
            source=f"{probe.source}-selected",
            hwnd=probe.hwnd,
            left=probe.left,
            top=probe.top,
            width=probe.width,
            height=probe.height,
        )
    return WindowBindingResult(
        bound=False,
        window_title=probe.window_title,
        app_name=probe.app_name,
        source=f"{probe.source}-selected",
        error="selected window is not active",
        hwnd=probe.hwnd,
        left=probe.left,
        top=probe.top,
        width=probe.width,
        height=probe.height,
    )


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
        left = int(getattr(active, "left", 0) or 0)
        top = int(getattr(active, "top", 0) or 0)
        width = int(getattr(active, "width", 0) or 0)
        height = int(getattr(active, "height", 0) or 0)
        return WindowBindingResult(
            bound=False,
            window_title=title,
            source="pygetwindow",
            hwnd=_window_hwnd_from_object(active),
            left=left,
            top=top,
            width=width or None,
            height=height or None,
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
        output = subprocess.check_output(
            ["osascript", "-e", script],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=3.0,
        ).strip()
        parts = output.split("||")
        app_name = _normalize_title(parts[0]) if len(parts) > 0 else ""
        title = _normalize_title(parts[1]) if len(parts) > 1 else ""
        pos = parts[2] if len(parts) > 2 else "0,0"
        size = parts[3] if len(parts) > 3 else "0,0"
        left_s, top_s = (pos.split(",", 1) + ["0"])[:2]
        width_s, height_s = (size.split(",", 1) + ["0"])[:2]
        return WindowBindingResult(
            bound=False,
            window_title=title,
            app_name=app_name,
            source="osascript",
            left=int(left_s) if left_s.strip().lstrip("-").isdigit() else None,
            top=int(top_s) if top_s.strip().lstrip("-").isdigit() else None,
            width=int(width_s) if width_s.strip().isdigit() else None,
            height=int(height_s) if height_s.strip().isdigit() else None,
        )
    except Exception as exc:
        return WindowBindingResult(bound=False, source="osascript", error=str(exc))


def _get_active_window_linux() -> WindowBindingResult:
    if shutil.which("xdotool"):
        try:
            window_id = subprocess.check_output(
                ["xdotool", "getactivewindow"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=3.0,
            ).strip()
            title = subprocess.check_output(
                ["xdotool", "getwindowname", window_id],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=3.0,
            ).strip()
            geometry_raw = subprocess.check_output(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=3.0,
            )
            geometry: dict[str, int] = {}
            for line in geometry_raw.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if value.lstrip("-").isdigit():
                    geometry[key] = int(value)
            return WindowBindingResult(
                bound=False,
                window_title=_normalize_title(title),
                source="xdotool",
                left=geometry.get("X"),
                top=geometry.get("Y"),
                width=geometry.get("WIDTH"),
                height=geometry.get("HEIGHT"),
            )
        except Exception as exc:
            return WindowBindingResult(bound=False, source="xdotool", error=str(exc))

    return WindowBindingResult(bound=False, source="xdotool", error="no supported active-window backend")


def get_active_window_info() -> WindowBindingResult:
    system = platform.system().lower()
    if system == "windows":
        return _get_active_window_windows()
    if system == "darwin":
        return _get_active_window_macos()
    if system == "linux":
        return _get_active_window_linux()
    return WindowBindingResult(bound=False, source="unknown", error=f"unsupported platform: {system}")


def bind_window_from_keywords(keywords: list[str]) -> WindowBindingResult:
    cleaned_keywords = _clean_keywords(keywords)
    if cleaned_keywords and platform.system().lower() == "windows":
        matched_window = _find_matching_window_windows(cleaned_keywords)
        if matched_window is not None:
            return matched_window

    probe = get_active_window_info()
    if not probe.window_title and not probe.app_name:
        return WindowBindingResult(
            bound=False,
            window_title=probe.window_title,
            app_name=probe.app_name,
            source=probe.source,
            error=probe.error or "no active window info available",
            hwnd=probe.hwnd,
            left=probe.left,
            top=probe.top,
            width=probe.width,
            height=probe.height,
        )

    ignored_probe = _window_should_be_ignored(probe.window_title, probe.app_name, probe.width, probe.height)
    if not cleaned_keywords:
        return WindowBindingResult(
            bound=bool(probe.window_title or probe.app_name) and not ignored_probe,
            window_title=probe.window_title,
            app_name=probe.app_name,
            source=probe.source,
            error=(
                "active window is not a viable Mahjong Soul game window"
                if ignored_probe
                else "" if (probe.window_title or probe.app_name) else (probe.error or "no keywords configured")
            ),
            hwnd=probe.hwnd,
            left=probe.left,
            top=probe.top,
            width=probe.width,
            height=probe.height,
        )

    match_keyword = _match_keyword(probe.window_title, probe.app_name, cleaned_keywords)
    if match_keyword and not ignored_probe:
        return WindowBindingResult(
            bound=True,
            window_title=probe.window_title,
            app_name=probe.app_name,
            match_keyword=match_keyword,
            source=probe.source,
            error="",
            hwnd=probe.hwnd,
            left=probe.left,
            top=probe.top,
            width=probe.width,
            height=probe.height,
        )

    return WindowBindingResult(
        bound=False,
        window_title=probe.window_title,
        app_name=probe.app_name,
        source=probe.source,
        error=(
            "active window is not a viable Mahjong Soul game window"
            if ignored_probe and match_keyword
            else "active window does not match keywords"
        ),
        hwnd=probe.hwnd,
        left=probe.left,
        top=probe.top,
        width=probe.width,
        height=probe.height,
    )
