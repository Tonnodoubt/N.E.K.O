from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .events import NarrationEvent


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _seconds_since(value: str, now: datetime) -> float | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def apply_speech_policy(
    event: NarrationEvent,
    policy_cfg: dict[str, Any],
    *,
    last_spoken_at: str = "",
    last_spoken_text: str = "",
    last_notified_at: str = "",
    last_notified_text: str = "",
    last_notified_key: str = "",
) -> NarrationEvent:
    now = datetime.now(timezone.utc)
    voice_enabled = bool(policy_cfg.get("voice_enabled", True))
    voice_mode = str(policy_cfg.get("voice_mode", "key_events_only")).strip() or "key_events_only"
    normal_voice_cooldown = int(policy_cfg.get("normal_voice_cooldown_sec", 18))
    danger_voice_cooldown = int(policy_cfg.get("danger_voice_cooldown_sec", 5))
    normal_notification_cooldown = int(policy_cfg.get("normal_notification_cooldown_sec", 18))
    danger_notification_cooldown = int(policy_cfg.get("danger_notification_cooldown_sec", 5))
    dedupe_window = int(policy_cfg.get("dedupe_window_sec", 8))

    delivery = "silent_ui"
    channel = event.channel
    speakable = event.speakable
    same_text = bool(last_spoken_text and event.text and last_spoken_text == event.text)
    spoken_delta = _seconds_since(last_spoken_at, now)
    notified_delta = _seconds_since(last_notified_at, now)
    same_notification_text = bool(last_notified_text and event.text and last_notified_text == event.text)
    same_notification_key = bool(last_notified_key and event.dedupe_key and last_notified_key == event.dedupe_key)

    if same_text and spoken_delta is not None and spoken_delta < dedupe_window:
        return replace(event, delivery="silent_ui", channel="silent_ui", speakable=False)

    if (same_notification_key or same_notification_text) and notified_delta is not None and notified_delta < dedupe_window:
        return replace(event, delivery="silent_ui", channel=event.channel, speakable=False)

    if event.event_type == "danger_action":
        channel = "warning"
        if notified_delta is not None and notified_delta < danger_notification_cooldown:
            delivery = "silent_ui"
            speakable = False
        elif voice_enabled and voice_mode != "off":
            cooldown = danger_voice_cooldown
            if spoken_delta is None or spoken_delta >= cooldown:
                delivery = "voice_candidate"
            else:
                delivery = "proactive_notification"
        else:
            delivery = "proactive_notification"
    elif event.event_type in {"action_available", "tile_efficiency_hint"}:
        channel = "nudge"
        is_call_window = _looks_like_call_window(event)
        if event.event_type == "action_available" and not event.speakable:
            channel = "silent_ui"
            delivery = "silent_ui"
            speakable = False
        elif notified_delta is not None and notified_delta < _normal_action_cooldown(
            is_call_window=is_call_window,
            default_cooldown=normal_notification_cooldown,
        ):
            delivery = "silent_ui"
            speakable = False
        else:
            delivery = "proactive_notification"
        if delivery != "silent_ui" and voice_enabled and voice_mode == "companion":
            voice_cooldown = 3 if is_call_window else normal_voice_cooldown
            if spoken_delta is None or spoken_delta >= voice_cooldown:
                delivery = "voice_candidate"
    elif event.event_type in {"waiting_state", "scene_update", "uncertain_state"}:
        channel = "silent_ui"
        delivery = "silent_ui"
        speakable = False

    return replace(event, delivery=delivery, channel=channel, speakable=speakable)


def _looks_like_call_window(event: NarrationEvent) -> bool:
    buttons = {str(button).strip() for button in event.buttons}
    return bool(buttons & {"chi", "pon", "kan", "skip"})


def _normal_action_cooldown(*, is_call_window: bool, default_cooldown: int) -> int:
    if is_call_window:
        return min(max(1, int(default_cooldown)), 3)
    return default_cooldown
