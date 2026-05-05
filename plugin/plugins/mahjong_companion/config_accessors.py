from __future__ import annotations

import time
from typing import Any


class ConfigAccessorMixin:
    def _get_sample_interval_ms(self) -> int:
        companion_cfg = self._config.get("mahjong_companion", {})
        value = companion_cfg.get("sample_interval_ms", 300)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 300
        return max(120, parsed)

    def _get_fast_poll_cfg(self) -> dict[str, Any]:
        companion_cfg = self._config.get("mahjong_companion", {})
        fast_poll_cfg = companion_cfg.get("fast_poll", {})
        return fast_poll_cfg if isinstance(fast_poll_cfg, dict) else {}

    def _get_current_sample_interval_ms(self) -> int:
        base_interval_ms = self._get_sample_interval_ms()
        overlay_interval_ms = self._get_active_overlay_poll_interval_ms()
        if overlay_interval_ms > 0 and self._current_screen_overlays():
            base_interval_ms = min(base_interval_ms, overlay_interval_ms)
        if not self._fast_poll_active():
            return base_interval_ms
        cfg = self._get_fast_poll_cfg()
        fast_interval_ms = _coerce_int(cfg.get("interval_ms", 80), default=80, minimum=120)
        return min(base_interval_ms, fast_interval_ms)

    def _fast_poll_active(self) -> bool:
        return time.monotonic() < self._fast_poll_until

    def _fast_poll_force_process_enabled(self) -> bool:
        cfg = self._get_fast_poll_cfg()
        return bool(cfg.get("force_process", True)) and self._fast_poll_active()

    def _arm_fast_poll_if_needed_locked(
        self,
        *,
        buttons: list[str] | None = None,
        focus: str = "",
    ) -> bool:
        cfg = self._get_fast_poll_cfg()
        if not bool(cfg.get("enabled", True)):
            return False

        trigger_buttons = _coerce_string_set(cfg.get("trigger_buttons"), default={"chi", "pon", "kan", "skip"})
        trigger_focuses = _coerce_string_set(
            cfg.get("trigger_focuses"),
            default={"call_decision", "kan_decision", "confirm_or_skip"},
        )
        clean_buttons = {str(button).strip() for button in (buttons or []) if str(button).strip()}
        clean_focus = str(focus or "").strip()
        if not (clean_buttons & trigger_buttons or clean_focus in trigger_focuses):
            return False

        duration_sec = _coerce_float(cfg.get("duration_sec", 7), default=7.0, minimum=0.2)
        self._fast_poll_until = max(self._fast_poll_until, time.monotonic() + duration_sec)
        return True

    def _get_active_overlay_poll_interval_ms(self) -> int:
        cfg = self._get_overlay_cfg()
        return _coerce_int(cfg.get("discard_marker_poll_interval_ms", 80), default=80, minimum=80)

    def _fast_button_scan_enabled(self) -> bool:
        cfg = self._get_overlay_cfg()
        return bool(cfg.get("fast_button_scan_enabled", True))

    def _get_fast_button_scan_min_interval_ms(self) -> int:
        cfg = self._get_overlay_cfg()
        return _coerce_int(cfg.get("fast_button_scan_min_interval_ms", 60), default=60, minimum=30)

    def _get_keywords(self) -> list[str]:
        companion_cfg = self._config.get("mahjong_companion", {})
        raw = companion_cfg.get("target_window_title_keywords", [])
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    def _get_capture_format(self) -> str:
        companion_cfg = self._config.get("mahjong_companion", {})
        capture_cfg = companion_cfg.get("capture", {})
        if not isinstance(capture_cfg, dict):
            return "png"
        value = str(capture_cfg.get("save_format", "png")).strip().lower()
        return value if value in {"png", "jpg", "jpeg"} else "png"

    def _get_debug_samples_cfg(self) -> dict[str, Any]:
        companion_cfg = self._config.get("mahjong_companion", {})
        debug_cfg = companion_cfg.get("debug_samples", {})
        return debug_cfg if isinstance(debug_cfg, dict) else {}

    def _get_frame_gate_cfg(self) -> dict[str, Any]:
        companion_cfg = self._config.get("mahjong_companion", {})
        frame_gate_cfg = companion_cfg.get("frame_change_gate", {})
        if not isinstance(frame_gate_cfg, dict):
            frame_gate_cfg = {}
        return frame_gate_cfg

    def _perception_debug_dump_enabled(self) -> bool:
        companion_cfg = self._config.get("mahjong_companion", {})
        perception_cfg = companion_cfg.get("perception", {})
        if not isinstance(perception_cfg, dict):
            return True
        return bool(perception_cfg.get("debug_dump", True))

    def _perception_enabled(self) -> bool:
        companion_cfg = self._config.get("mahjong_companion", {})
        perception_cfg = companion_cfg.get("perception", {})
        if not isinstance(perception_cfg, dict):
            return True
        return bool(perception_cfg.get("enabled", True))

    def _decision_debug_dump_enabled(self) -> bool:
        companion_cfg = self._config.get("mahjong_companion", {})
        decision_cfg = companion_cfg.get("decision", {})
        if not isinstance(decision_cfg, dict):
            return True
        return bool(decision_cfg.get("debug_dump", True))

    def _decision_enabled(self) -> bool:
        companion_cfg = self._config.get("mahjong_companion", {})
        decision_cfg = companion_cfg.get("decision", {})
        if not isinstance(decision_cfg, dict):
            return True
        return bool(decision_cfg.get("enabled", True))

    def _preturn_planning_enabled(self) -> bool:
        companion_cfg = self._config.get("mahjong_companion", {})
        decision_cfg = companion_cfg.get("decision", {})
        if not isinstance(decision_cfg, dict):
            return True
        return bool(decision_cfg.get("preturn_planning_enabled", True))

    def _narration_debug_dump_enabled(self) -> bool:
        companion_cfg = self._config.get("mahjong_companion", {})
        narration_cfg = companion_cfg.get("narration", {})
        if not isinstance(narration_cfg, dict):
            return True
        return bool(narration_cfg.get("debug_dump", True))

    def _narration_enabled(self) -> bool:
        companion_cfg = self._config.get("mahjong_companion", {})
        narration_cfg = companion_cfg.get("narration", {})
        if not isinstance(narration_cfg, dict):
            return True
        return bool(narration_cfg.get("enabled", True))

    def _auto_dispatch_enabled(self) -> bool:
        speech_cfg = self._get_speech_policy_cfg()
        return bool(speech_cfg.get("auto_dispatch_enabled", True))

    def _get_overlay_cfg(self) -> dict[str, Any]:
        companion_cfg = self._config.get("mahjong_companion", {})
        overlay_cfg = companion_cfg.get("overlay", {})
        return overlay_cfg if isinstance(overlay_cfg, dict) else {}

    def _get_speech_policy_cfg(self) -> dict[str, Any]:
        companion_cfg = self._config.get("mahjong_companion", {})
        speech_cfg = companion_cfg.get("speech_policy", {})
        if not isinstance(speech_cfg, dict):
            speech_cfg = {}
        merged = dict(speech_cfg)
        merged["voice_enabled"] = self.state.voice_enabled
        merged["voice_mode"] = self.state.voice_mode
        return merged

    def _get_voice_target_lanlan(self) -> str:
        speech_cfg = self._get_speech_policy_cfg()
        return str(speech_cfg.get("target_lanlan", "")).strip()

    def _get_overlay_max_age_ms(self) -> int:
        cfg = self._get_overlay_cfg()
        return _coerce_int(cfg.get("discard_marker_max_age_ms", 0), default=0, minimum=0)

    def _get_overlay_region_change_threshold(self) -> int:
        cfg = self._get_overlay_cfg()
        return _coerce_int(cfg.get("discard_marker_region_change_threshold", 10), default=10, minimum=1)


def _coerce_int(value: Any, *, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _coerce_float(value: Any, *, default: float, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _coerce_string_set(value: Any, *, default: set[str]) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        parsed = {str(item).strip() for item in value if str(item).strip()}
        return parsed or set(default)
    if isinstance(value, str):
        parsed = {item.strip() for item in value.split(",") if item.strip()}
        return parsed or set(default)
    return set(default)
