from __future__ import annotations

import multiprocessing
import os
import platform
from typing import Any, Callable

from .ipc import CLOSE_MESSAGE, drain_queue, put_latest
from .process import run_overlay_process
from .view import (
    _DragState,
    _advice_text,
    _advice_view,
    _parse_marker_box,
    _reason_line,
    _render_screen_markers,
)


def _overlay_platform_supported() -> bool:
    system = platform.system()
    if system in {"Windows", "Darwin"}:
        return True
    if system == "Linux":
        return bool(os.environ.get("DISPLAY"))
    return False


class CompanionOverlay:
    def __init__(
        self,
        logger: Any | None = None,
        *,
        mp_context: Any | None = None,
        process_target: Callable[[Any, Any], None] = run_overlay_process,
    ) -> None:
        self._logger = logger
        self._context = mp_context or multiprocessing.get_context("spawn")
        self._process_target = process_target
        self._status_queue: Any | None = None
        self._command_queue: Any | None = None
        self._process: Any | None = None
        self._unsupported_logged = False

    def start(self) -> bool:
        if not _overlay_platform_supported():
            if self._logger is not None and not self._unsupported_logged:
                self._logger.warning(
                    "mahjong companion overlay disabled: platform %s has no supported Tk display.",
                    platform.system(),
                )
                self._unsupported_logged = True
            return False

        if self.is_running():
            return True

        self._cleanup_process()
        self._status_queue = self._context.Queue(maxsize=3)
        self._command_queue = self._context.Queue(maxsize=20)
        process = self._context.Process(
            target=self._process_target,
            args=(self._status_queue, self._command_queue),
            name="mahjong-companion-overlay",
        )
        try:
            process.daemon = True
        except Exception:
            pass
        try:
            process.start()
        except Exception as exc:
            self._process = None
            self._status_queue = None
            self._command_queue = None
            if self._logger is not None:
                self._logger.warning("mahjong companion overlay unavailable: %s", exc)
            return False

        self._process = process
        return True

    def stop(self) -> None:
        if self._status_queue is not None:
            put_latest(self._status_queue, CLOSE_MESSAGE)
        process = self._process
        if process is not None and process.is_alive():
            try:
                process.join(timeout=1.5)
            except Exception:
                pass
            if process.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass
                try:
                    process.join(timeout=1.0)
                except Exception:
                    pass
        self._cleanup_process()

    def update_status(self, status: dict[str, Any]) -> None:
        if self._status_queue is None or not self.is_running():
            return
        put_latest(self._status_queue, dict(status))

    def is_running(self) -> bool:
        process = self._process
        return bool(process is not None and process.is_alive())

    def drain_commands(self) -> list[str]:
        if self._command_queue is None:
            return []
        commands: list[str] = []
        for item in drain_queue(self._command_queue):
            if isinstance(item, dict) and item.get("__control__") == "error":
                if self._logger is not None:
                    self._logger.warning("mahjong companion overlay child failed: %s", item.get("error", ""))
                continue
            command = str(item or "").strip()
            if command:
                commands.append(command)
        self._cleanup_process()
        return commands

    def _cleanup_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            return
        try:
            process.join(timeout=0)
        except Exception:
            pass
        self._process = None


__all__ = [
    "CompanionOverlay",
    "_DragState",
    "_advice_text",
    "_advice_view",
    "_overlay_platform_supported",
    "_parse_marker_box",
    "_reason_line",
    "_render_screen_markers",
]

