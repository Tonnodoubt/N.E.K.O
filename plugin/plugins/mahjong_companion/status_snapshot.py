from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .session_state import SessionState


@dataclass(frozen=True)
class StatusSnapshotContext:
    state: SessionState
    selected_window_title: str = ""
    overlay_visible: bool = False
    preturn_discard_plan: Any | None = None
    last_preturn_plan_meta: dict[str, Any] = field(default_factory=dict)
    last_fast_advice_frame_path: str = ""
    last_fast_button_scan_meta: dict[str, Any] = field(default_factory=dict)


def derive_report_status(state: SessionState) -> str:
    if state.runtime_mode == "standby" and state.running:
        return "standby"
    if state.runtime_mode == "off":
        return "idle"
    runtime_status = state.status
    if runtime_status in {"idle", "starting", "stopping", "warning", "error"}:
        return runtime_status
    if not state.running:
        return "idle"
    if state.scene and state.scene != "unknown":
        return state.scene
    return "scanning"


def current_screen_overlays(decision_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    decision = decision_payload if isinstance(decision_payload, dict) else {}
    direct = decision.get("screen_overlays")
    if isinstance(direct, list):
        return [dict(item) for item in direct if isinstance(item, dict)]
    engine_meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
    overlays = engine_meta.get("screen_overlays")
    if isinstance(overlays, list):
        return [dict(item) for item in overlays if isinstance(item, dict)]
    return []


def build_status_snapshot(ctx: StatusSnapshotContext) -> dict[str, Any]:
    state = ctx.state
    report_status = derive_report_status(state)
    preturn_discard_plan = (
        ctx.preturn_discard_plan.to_dict()
        if ctx.preturn_discard_plan is not None
        else {}
    )
    return {
        "status": report_status,
        "runtime_status": state.status,
        "mode": state.mode,
        "scene": state.scene,
        "session_id": state.session_id,
        "window_bound": state.window_bound,
        "window_title": state.window_title,
        "window_match_keyword": state.window_match_keyword,
        "selected_window_title": ctx.selected_window_title,
        "overlay_visible": ctx.overlay_visible,
        "window_left": state.window_left,
        "window_top": state.window_top,
        "window_width": state.window_width,
        "window_height": state.window_height,
        "last_frame_path": state.last_frame_path,
        "last_frame_at": state.last_frame_at,
        "last_capture_source": state.last_capture_source,
        "last_capture_ok": state.last_capture_ok,
        "last_scene": state.last_scene,
        "last_scene_confidence": state.last_scene_confidence,
        "last_is_user_turn": state.last_is_user_turn,
        "last_buttons": state.last_buttons,
        "preturn_discard_plan": preturn_discard_plan,
        "last_preturn_plan_meta": dict(ctx.last_preturn_plan_meta),
        "last_fast_advice_frame_path": ctx.last_fast_advice_frame_path,
        "last_fast_button_scan_meta": dict(ctx.last_fast_button_scan_meta),
        "last_perception_at": state.last_perception_at,
        "last_perception_ok": state.last_perception_ok,
        "last_perception": state.last_perception,
        "last_decision_at": state.last_decision_at,
        "last_decision_ok": state.last_decision_ok,
        "last_decision_type": state.last_decision_type,
        "last_decision_risk_level": state.last_decision_risk_level,
        "last_tile_analysis_available": state.last_tile_analysis_available,
        "last_shanten_estimate": state.last_shanten_estimate,
        "last_ukeire_estimate": state.last_ukeire_estimate,
        "last_decision": state.last_decision,
        "screen_overlays": current_screen_overlays(state.last_decision),
        "last_narration_at": state.last_narration_at,
        "last_narration_ok": state.last_narration_ok,
        "last_narration_type": state.last_narration_type,
        "last_narration_channel": state.last_narration_channel,
        "last_narration_delivery": state.last_narration_delivery,
        "last_narration_text": state.last_narration_text,
        "last_narration": state.last_narration,
        "last_companion_mood": state.last_companion_mood,
        "last_companion_view": state.last_companion_view,
        "voice_enabled": state.voice_enabled,
        "voice_mode": state.voice_mode,
        "last_notification_at": state.last_notification_at,
        "last_notification_text": state.last_notification_text,
        "last_notification_key": state.last_notification_key,
        "last_notification_channel": state.last_notification_channel,
        "last_notification_delivery": state.last_notification_delivery,
        "last_notification_ok": state.last_notification_ok,
        "last_spoken_at": state.last_spoken_at,
        "last_spoken_text": state.last_spoken_text,
        "last_speak_ok": state.last_speak_ok,
        "runtime_mode": state.runtime_mode,
        "game_runtime_status": state.runtime_status,
        "last_runtime_command_id": state.last_runtime_command_id,
        "last_runtime_command_at": state.last_runtime_command_at,
        "last_runtime_command_source": state.last_runtime_command_source,
        "last_runtime_command_action": state.last_runtime_command_action,
        "last_runtime_command_ok": state.last_runtime_command_ok,
        "last_runtime_command_result": state.last_runtime_command_result,
        "last_runtime_interrupt_reason": state.last_runtime_interrupt_reason,
        "last_error": state.last_error,
    }
