from __future__ import annotations

from typing import Any, Callable

from ..session_state import SessionState, now_iso
from .events import NarrationEvent


class NarrationDispatcher:
    def __init__(self, plugin: Any, *, plugin_id: str = "mahjong_companion") -> None:
        self._plugin = plugin
        self._plugin_id = plugin_id

    def dispatch(
        self,
        event: NarrationEvent,
        *,
        state: SessionState,
        emit_status: Callable[[], None],
        target_lanlan: str = "",
        require_running: bool = True,
        require_window_bound: bool = True,
    ) -> dict[str, Any]:
        if event.delivery not in {"proactive_notification", "voice_candidate"}:
            return {
                "ok": False,
                "skipped": True,
                "reason": "delivery_suppressed",
                "delivery": event.delivery,
            }

        if require_running and not state.running:
            return {
                "ok": False,
                "skipped": True,
                "reason": "session_not_running",
            }

        if require_window_bound and not state.window_bound:
            return {
                "ok": False,
                "skipped": True,
                "reason": "window_not_bound",
            }

        try:
            self._plugin.push_message(
                source=self._plugin_id,
                message_type="proactive_notification",
                description="雀魂陪伴提醒",
                priority=min(10, max(1, int(state.last_decision.get("priority", 5) / 10) or 5)),
                content=event.text,
                metadata={
                    "plugin_id": self._plugin_id,
                    "decision_type": state.last_decision_type,
                    "narration_type": event.event_type,
                    "delivery": event.delivery,
                    "channel": event.channel,
                    "dedupe_key": event.dedupe_key,
                },
                target_lanlan=target_lanlan or None,
            )
        except Exception as exc:
            state.last_notification_ok = False
            state.last_speak_ok = False
            state.last_error = str(exc)
            emit_status()
            raise

        state.last_notification_at = now_iso()
        state.last_notification_text = event.text
        state.last_notification_key = event.dedupe_key
        state.last_notification_channel = event.channel
        state.last_notification_delivery = event.delivery
        state.last_notification_ok = True
        state.last_error = ""
        if event.delivery == "voice_candidate":
            state.last_spoken_at = state.last_notification_at
            state.last_spoken_text = event.text
            state.last_speak_ok = True
        emit_status()
        return {
            "ok": True,
            "delivery": event.delivery,
            "channel": event.channel,
            "text": event.text,
        }

    def build_debug_reply_event(self, event: NarrationEvent) -> NarrationEvent:
        payload = event.to_dict()
        payload["delivery"] = "proactive_notification"
        payload["channel"] = "companion"
        payload["speakable"] = True
        return NarrationEvent(**payload)

    def apply_debug_reply_event(self, event: NarrationEvent, *, state: SessionState) -> None:
        state.last_narration = event.to_dict()
        state.last_narration_delivery = event.delivery
        state.last_narration_channel = event.channel
        state.last_narration_text = event.text
        if state.last_companion_view:
            state.last_companion_view["delivery"] = event.delivery
            state.last_companion_view["speakable"] = event.speakable
