from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .contracts import DecisionResult, PerceivedGameState
from .decision.preturn_planner import apply_preturn_draw_tile
from .decision.generator import build_decision
from .perception.pipeline import analyze_action_buttons_fast
from .perception.drawn_tile_fast_path import detect_drawn_tile_fast_path
from .frame_resources import _path_mtime
from .session_state import now_iso
from .state_transitions import _image_region_signature


class PreturnFastPathMixin:
    _PERCEPTION_FIELDS_FOR_FAST_PATH = (
        "scene",
        "last_scene",
        "last_scene_confidence",
        "last_is_user_turn",
        "last_buttons",
        "last_perception_at",
        "last_perception_ok",
        "last_perception",
        "last_error",
    )
    _DECISION_FIELDS_FOR_FAST_PATH = (
        "last_decision_at",
        "last_decision_ok",
        "last_decision_type",
        "last_decision_risk_level",
        "last_tile_analysis_available",
        "last_shanten_estimate",
        "last_ukeire_estimate",
        "last_decision",
    )
    _NARRATION_FIELDS_FOR_FAST_PATH = (
        "last_narration_at",
        "last_narration_ok",
        "last_narration_type",
        "last_narration_channel",
        "last_narration_delivery",
        "last_narration_text",
        "last_narration",
        "last_companion_mood",
        "last_companion_view",
        "last_speak_ok",
    )

    def _maybe_emit_fast_action_button_overlay_locked(self, frame_path: Path) -> bool:
        if not self._fast_button_scan_enabled():
            return False
        now_mono = time.monotonic()
        min_interval_ms = self._get_fast_button_scan_min_interval_ms()
        if (
            min_interval_ms > 0
            and self._last_fast_button_scan_at > 0.0
            and (now_mono - self._last_fast_button_scan_at) * 1000.0 < min_interval_ms
        ):
            return False
        self._last_fast_button_scan_at = now_mono

        try:
            perceived, debug_payload = analyze_action_buttons_fast(
                frame_path,
                calibration_dir=self.plugin.data_path("calibration"),
            )
        except Exception as exc:
            self._last_fast_button_scan_meta = {
                "ok": False,
                "frame_path": str(frame_path),
                "error": str(exc),
            }
            return False

        visible_buttons = {str(button).strip() for button in perceived.buttons if str(button).strip()}
        if not perceived.buttons or not perceived.button_regions:
            self._last_fast_button_scan_meta = {
                "ok": True,
                "frame_path": str(frame_path),
                "button_count": 0,
                "overlay_count": 0,
            }
            if self._clear_missing_fast_action_button_overlay_locked(
                visible_buttons,
                reason="fast_buttons_missing",
            ):
                self._emit_status()
                return True
            return False

        decision = build_decision(perceived)
        overlays = self._build_fast_action_button_overlays(decision, perceived, frame_path)
        self._last_fast_button_scan_meta = {
            "ok": True,
            "frame_path": str(frame_path),
            "buttons": list(perceived.buttons),
            "button_count": len(perceived.buttons),
            "overlay_count": len(overlays),
            "decision_type": decision.decision_type,
            "recommended_focus": decision.recommended_focus,
            "debug": dict(debug_payload),
        }
        if not overlays:
            if self._clear_missing_fast_action_button_overlay_locked(
                visible_buttons,
                reason="fast_recommended_button_missing",
            ):
                self._emit_status()
                return True
            return False

        existing_overlays = self._current_screen_overlays()
        if existing_overlays and overlays:
            existing_buttons = {
                o.get("button_type") for o in existing_overlays
                if isinstance(o, dict) and o.get("kind") == "action_button_recommendation"
            }
            new_buttons = {o.get("button_type") for o in overlays}
            if existing_buttons == new_buttons:
                return False

        payload = decision.to_dict()
        overlay_updated_at = time.monotonic()
        overlays = [
            {
                **overlay,
                "created_at_monotonic": overlay_updated_at,
                "fast_path": True,
            }
            for overlay in overlays
        ]
        payload["screen_overlays"] = overlays
        payload["engine_meta"] = {
            **dict(payload.get("engine_meta") if isinstance(payload.get("engine_meta"), dict) else {}),
            "screen_overlays": overlays,
            "screen_overlay_count": len(overlays),
            "fast_action_button_overlay": True,
        }

        self.state.last_decision_at = now_iso()
        self.state.last_decision_ok = True
        self.state.last_decision_type = decision.decision_type
        self.state.last_decision_risk_level = decision.risk_level
        self.state.last_tile_analysis_available = False
        self.state.last_shanten_estimate = None
        self.state.last_ukeire_estimate = None
        self.state.last_decision = payload
        self._last_screen_overlay_update_at = overlay_updated_at
        self._arm_fast_poll_if_needed_locked(buttons=list(perceived.buttons), focus=decision.recommended_focus)
        self._emit_status()
        return True

    def _clear_missing_fast_action_button_overlay_locked(self, visible_buttons: set[str], *, reason: str) -> bool:
        overlays = self._current_screen_overlays()
        if not overlays:
            return False

        action_overlays = [
            overlay
            for overlay in overlays
            if isinstance(overlay, dict) and overlay.get("kind") == "action_button_recommendation"
        ]
        if not action_overlays or len(action_overlays) != len(overlays):
            return False

        overlay_buttons = {
            str(overlay.get("button_type", "")).strip()
            for overlay in action_overlays
            if str(overlay.get("button_type", "")).strip()
        }
        if not overlay_buttons or overlay_buttons & visible_buttons:
            return False
        return self._clear_screen_overlays_locked(reason=reason)

    def _build_fast_action_button_overlays(
        self,
        decision: DecisionResult,
        perceived: PerceivedGameState,
        frame_path: Path,
    ) -> list[dict[str, Any]]:
        engine_meta = decision.engine_meta if isinstance(decision.engine_meta, dict) else {}
        recommended_buttons = engine_meta.get("recommended_button_types")
        if not isinstance(recommended_buttons, list):
            return []
        frame_mtime = _path_mtime(frame_path)
        overlays: list[dict[str, Any]] = []
        for button_type in recommended_buttons:
            clean_type = str(button_type or "").strip()
            if not clean_type:
                continue
            region = self._find_button_region(perceived.button_regions, clean_type)
            if region is None:
                continue
            local_box = self._button_region_to_local_box(region)
            if not local_box:
                continue
            screen_box = self._local_box_to_screen_box(local_box)
            if not screen_box:
                continue
            region_signature = _image_region_signature(frame_path, local_box)
            overlays.append({
                "kind": "action_button_recommendation",
                "button_type": clean_type,
                "label": str(region.get("label") or clean_type),
                "box": screen_box,
                "local_box": local_box,
                "frame_path": str(frame_path),
                "frame_mtime": frame_mtime,
                "region_signature": region_signature,
                "confidence": region.get("confidence"),
                "template_id": str(region.get("template_id") or ""),
                "source": "fast_button_scan",
            })
            break
        return overlays

    def _maybe_emit_fast_preturn_advice_locked(self, frame_path: Path) -> bool:
        if not self._preturn_planning_enabled() or self._preturn_discard_plan is None:
            return False
        frame_key = str(frame_path)
        if self._last_fast_advice_frame_path == frame_key:
            return False

        # plan.draw_slot_index is already clamped to [1, 14] by build_preturn_discard_plan,
        # and _resolve_draw_slot defensively clamps again. No need to clamp here a third time
        # (CODE_REVIEW_v1.2 N-L3).
        draw_slot_index = self._preturn_discard_plan.draw_slot_index
        result = detect_drawn_tile_fast_path(
            frame_path,
            calibration_dir=self.plugin.data_path("calibration"),
            draw_slot_index=draw_slot_index,
        )
        self._last_preturn_plan_meta = {
            **dict(self._last_preturn_plan_meta),
            "fast_path": result.to_dict(),
            "fast_path_draw_slot_index": draw_slot_index,
            "fast_path_frame_path": frame_key,
        }
        if not result.ok:
            return False

        base_state = self._fast_path_base_state(result.tile, result.to_dict())
        prepared, meta = apply_preturn_draw_tile(base_state, self._preturn_discard_plan, result.tile)
        self._last_preturn_plan_meta = {
            **dict(meta),
            "fast_path": result.to_dict(),
            "fast_path_draw_slot_index": draw_slot_index,
            "fast_path_frame_path": frame_key,
            "fast_path_applied": bool(meta.get("applied")),
        }
        if not meta.get("applied"):
            return False

        # Snapshot the entire perception/decision/narration slice so a failed
        # fast-path decision rolls back ALL touched fields, not just last_perception.
        # The previous code left scene / last_buttons / cleared decision&narration
        # in a dirty half-applied state for the next tick (CODE_REVIEW_v1.2 N-H3).
        state_snapshot = self._snapshot_fast_path_state()
        perception_payload = self._apply_perception_result(prepared)
        decision = self._generate_decision_locked()
        if not decision.get("ok"):
            self._restore_fast_path_state(state_snapshot)
            return False

        if self._narration_enabled():
            narration = self._generate_narration_locked()
            if narration.get("ok"):
                event = self._current_narration_event()
                if event is not None and self._auto_dispatch_enabled():
                    # Queue for post-cycle dispatch outside the lock. The slow
                    # plugin.push_message (TTS / IPC) used to block every UI
                    # entry point that needed the orchestrator lock; now it
                    # runs in `_drain_pending_fast_dispatch_events` after the
                    # cycle releases the lock (CODE_REVIEW_v1.2 N-H2).
                    self._pending_fast_dispatch_events.append(event)

        self._last_fast_advice_frame_path = frame_key
        self._last_preturn_plan_meta.update(
            {
                "fast_advice_emitted": True,
                "fast_advice_frame_path": frame_key,
                "fast_advice_top_tile": decision.get("mahjong_analysis", {})
                .get("candidate_discards", [{}])[0]
                .get("tile", ""),
                "fast_advice_perception": perception_payload,
            }
        )
        self._emit_status()
        return True

    def _fast_path_base_state(self, drawn_tile: str, fast_path_payload: dict[str, Any]) -> PerceivedGameState:
        plan = self._preturn_discard_plan
        previous = self._current_perceived_state()
        if previous is not None:
            payload = previous.to_dict()
        else:
            payload = {
                "scene": "in_match",
                "confidence": 0.78,
                "is_user_turn": True,
                "buttons": [],
                "notes": [],
            }
        payload.update(
            {
                "scene": "in_match",
                "confidence": max(float(payload.get("confidence", 0.0) or 0.0), 0.78),
                "is_user_turn": True,
                "buttons": [],
                "hand_tiles": [*(plan.hand_tiles if plan is not None else []), drawn_tile],
            }
        )
        # Previously this constructed [[""] for _ in range(plan.meld_count)] as a
        # placeholder. _meld_group_count would count that as N valid groups, but
        # _normalize_group_list filtered the empty-string tiles back out, leaving
        # the two views inconsistent (CODE_REVIEW_v1.2 H1). The meld_count is
        # already carried via analysis_hints["recognized_meld_group_count"]
        # below; downstream `_meld_group_count` helpers fall back to that hint
        # when state.melds is empty, so we do not need a fake melds array here.
        notes = list(payload.get("notes") if isinstance(payload.get("notes"), list) else [])
        notes.append("fast preturn advice from draw slot")
        payload["notes"] = notes
        hints = dict(payload.get("analysis_hints") if isinstance(payload.get("analysis_hints"), dict) else {})
        hints["drawn_tile_fast_path"] = fast_path_payload
        hints["recognized_hand_tile_count"] = len(payload["hand_tiles"])
        if plan is not None and plan.meld_count:
            try:
                hinted_meld_count = int(hints.get("recognized_meld_group_count", 0) or 0)
            except (TypeError, ValueError):
                hinted_meld_count = 0
            hints["recognized_meld_group_count"] = max(
                plan.meld_count,
                hinted_meld_count,
            )
        payload["analysis_hints"] = hints
        if plan is not None:
            payload["dora_indicators"] = list(payload.get("dora_indicators") or [])
        return PerceivedGameState(**payload)

    def _snapshot_fast_path_state(self) -> dict[str, Any]:
        names = (
            *self._PERCEPTION_FIELDS_FOR_FAST_PATH,
            *self._DECISION_FIELDS_FOR_FAST_PATH,
            *self._NARRATION_FIELDS_FOR_FAST_PATH,
        )
        snapshot: dict[str, Any] = {}
        for name in names:
            value = getattr(self.state, name)
            if isinstance(value, dict):
                snapshot[name] = dict(value)
            elif isinstance(value, list):
                snapshot[name] = list(value)
            else:
                snapshot[name] = value
        snapshot["__fast_poll_until"] = self._fast_poll_until
        return snapshot

    def _restore_fast_path_state(self, snapshot: dict[str, Any]) -> None:
        for name, value in snapshot.items():
            if name == "__fast_poll_until":
                self._fast_poll_until = value
                continue
            setattr(self.state, name, value)
