from __future__ import annotations

from typing import Any

from .session_state import now_iso


class OverlayControllerMixin:
    def _process_overlay_commands_locked(self) -> bool:
        commands = self._overlay.drain_commands()
        forced_refresh = False
        for command in commands:
            command_name = str(command or "").strip()
            if not command_name:
                continue
            self.state.last_runtime_command_id = f"overlay-{now_iso()}"
            self.state.last_runtime_command_source = "overlay"
            self.state.last_runtime_command_action = command_name
            self.state.last_runtime_command_at = now_iso()
            self.state.runtime_status = "processing_overlay_command"
            try:
                result = self._handle_overlay_command_locked(command_name)
                self.state.last_runtime_command_ok = bool(result.get("ok", True))
                self.state.last_runtime_command_result = dict(result)
                if command_name == "refresh_advice":
                    forced_refresh = True
                if not self.state.last_runtime_command_ok and result.get("error"):
                    self.state.last_error = str(result.get("error"))
            except Exception as exc:
                self.logger.exception("overlay command failed")
                self.state.last_runtime_command_ok = False
                self.state.last_runtime_command_result = {
                    "ok": False,
                    "error": str(exc),
                }
                self.state.last_error = str(exc)
            finally:
                self._emit_status()
        return forced_refresh

    def _handle_overlay_command_locked(self, command: str) -> dict[str, Any]:
        if command == "refresh_advice":
            previous_runtime_mode = self.state.runtime_mode
            self.state.runtime_mode = "active"
            self.state.runtime_status = self.state.runtime_mode
            self.state.status = "refreshing_advice"
            self._emit_status()
            result = self._run_companion_pipeline_locked(
                "",
                capture=True,
                dispatch=True,
                force_reply=True,
                persist_review_artifacts=True,
            )
            result["runtime_mode_before_refresh"] = previous_runtime_mode
            result["runtime_mode"] = self.state.runtime_mode
            return result

        if command == "speak":
            return self._dispatch_overlay_narration_locked()

        if command == "toggle_pause":
            next_mode = "standby" if self.state.runtime_mode == "active" else "active"
            self.state.runtime_mode = next_mode
            self.state.runtime_status = self.state.runtime_mode
            if next_mode == "standby":
                self.state.status = "standby"
            elif self.state.running and self.state.status == "standby":
                self.state.status = "scanning"
            return {
                "ok": True,
                "runtime_mode": self.state.runtime_mode,
                "status": self.state.status,
            }

        if command == "hide_overlay":
            self._hide_overlay_locked()
            return {
                "ok": True,
                "overlay_visible": False,
            }

        return {
            "ok": False,
            "error": f"unknown overlay command: {command}",
        }

    def _dispatch_overlay_narration_locked(self) -> dict[str, Any]:
        ready, payload = self._ensure_narration_locked()
        if not ready:
            return {
                "ok": False,
                "error": payload.get("error", "narration_unavailable"),
                "narration": payload,
            }

        event = self._current_narration_event()
        if event is None:
            return {
                "ok": False,
                "error": "no narration available",
            }

        debug_event = self._build_debug_reply_event(event)
        dispatch = self._dispatch_debug_narration_locked(debug_event)
        return {
            "ok": bool(dispatch.get("ok")),
            "text": debug_event.text,
            "dispatch": dispatch,
        }

    def _overlay_enabled(self) -> bool:
        return bool(self._get_overlay_cfg().get("enabled", True))

    def _overlay_auto_show_on_bind(self) -> bool:
        cfg = self._get_overlay_cfg()
        return self._overlay_enabled() and bool(cfg.get("auto_show_on_bind", True))

    def _show_overlay_locked(self) -> dict[str, Any]:
        if not self._overlay_enabled():
            self._overlay_visible = False
            return {
                "ok": False,
                "overlay_visible": False,
                "error": "overlay_disabled",
            }
        try:
            self._overlay_visible = bool(self._overlay.start())
            return {
                "ok": self._overlay_visible,
                "overlay_visible": self._overlay_visible,
            }
        except Exception as exc:
            self.logger.exception("show mahjong companion overlay failed")
            self._overlay_visible = False
            self.state.last_error = str(exc)
            return {
                "ok": False,
                "overlay_visible": False,
                "error": str(exc),
            }

    def _hide_overlay_locked(self) -> None:
        try:
            self._overlay.stop()
        except Exception:
            self.logger.exception("hide mahjong companion overlay failed")
        self._overlay_visible = False
