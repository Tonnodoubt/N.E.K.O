from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .contracts import DecisionResult, PerceivedGameState
from .decision.preturn_planner import apply_preturn_discard_plan, build_preturn_discard_plan
from .frame_resources import _path_mtime
from .narration import NarrationEvent, apply_speech_policy
from .session_state import now_iso
from .storage import load_json_payload

_SCREEN_MARKER_KINDS = {
    "discard_recommendation",
    "action_button_recommendation",
    "meld_selection_recommendation",
}


class StateTransitionMixin:
    def _current_perceived_state(self) -> Optional[PerceivedGameState]:
        if not self.state.last_perception_ok or not self.state.last_perception:
            return None
        try:
            return PerceivedGameState(**self.state.last_perception)
        except Exception:
            return None

    def _current_decision_result(self) -> Optional[DecisionResult]:
        if not self.state.last_decision_ok or not self.state.last_decision:
            return None
        try:
            return DecisionResult(**self.state.last_decision)
        except Exception:
            return None

    def _maybe_clear_expired_screen_overlays_locked(self) -> bool:
        overlays = self._current_screen_overlays()
        if not overlays:
            return False
        frame_path = self._resolve_latest_frame_path()
        if frame_path is None:
            return False

        now_mono = time.monotonic()
        perceived = self._current_perceived_state()
        if perceived is not None:
            current_hand_count = len(perceived.hand_tiles)
            current_buttons = set(perceived.buttons)
            for overlay in overlays:
                if not isinstance(overlay, dict):
                    continue
                if overlay.get("kind") == "discard_recommendation":
                    stored_count = overlay.get("hand_tile_count")
                    if isinstance(stored_count, int) and stored_count != current_hand_count:
                        return self._clear_screen_overlays_locked(reason="hand_tile_count_changed")
                elif overlay.get("kind") == "action_button_recommendation":
                    button_type = str(overlay.get("button_type", "")).strip()
                    if button_type and button_type not in current_buttons:
                        return self._clear_screen_overlays_locked(reason="button_no_longer_visible")

        if self._screen_overlay_region_changed(frame_path, overlays):
            return self._clear_screen_overlays_locked(reason="overlay_region_changed")

        max_age_ms = self._get_overlay_max_age_ms()
        if max_age_ms > 0 and self._last_screen_overlay_update_at > 0:
            age_ms = (now_mono - self._last_screen_overlay_update_at) * 1000.0
            if age_ms >= max_age_ms:
                return self._clear_screen_overlays_locked(reason="overlay_expired")
        return False

    def _clear_screen_overlays_locked(self, *, reason: str) -> bool:
        decision = self.state.last_decision if isinstance(self.state.last_decision, dict) else {}
        if not decision:
            return False

        changed = False
        if isinstance(decision.get("screen_overlays"), list):
            decision = dict(decision)
            decision.pop("screen_overlays", None)
            changed = True

        engine_meta = decision.get("engine_meta")
        if isinstance(engine_meta, dict) and isinstance(engine_meta.get("screen_overlays"), list):
            decision = dict(decision)
            updated_meta = dict(engine_meta)
            updated_meta.pop("screen_overlays", None)
            updated_meta["screen_overlay_count"] = 0
            updated_meta["screen_overlay_cleared_reason"] = reason
            decision["engine_meta"] = updated_meta
            changed = True

        if not changed:
            return False
        self.state.last_decision = decision
        self._last_screen_overlay_update_at = 0.0
        return True

    def _screen_overlay_region_changed(self, frame_path: Path, overlays: list[dict[str, Any]]) -> bool:
        threshold = self._get_overlay_region_change_threshold()
        for overlay in overlays:
            if (
                not isinstance(overlay, dict)
                or overlay.get("kind") not in _SCREEN_MARKER_KINDS
            ):
                continue
            source_path = Path(str(overlay.get("frame_path") or ""))
            local_box = overlay.get("local_box") if isinstance(overlay.get("local_box"), dict) else {}
            signature = str(overlay.get("region_signature") or "")
            if not source_path or not local_box or not signature:
                continue
            if source_path == frame_path:
                continue
            current_signature = _image_region_signature(frame_path, local_box)
            if current_signature and _image_region_signature_distance(current_signature, signature) >= threshold:
                return True
        return False

    def _current_narration_event(self) -> Optional[NarrationEvent]:
        if not self.state.last_narration_ok or not self.state.last_narration:
            return None
        try:
            return NarrationEvent(**self.state.last_narration)
        except Exception:
            return None

    def _reapply_current_narration_policy_locked(self) -> Optional[NarrationEvent]:
        event = self._current_narration_event()
        if event is None:
            return None
        updated = apply_speech_policy(
            event,
            self._get_speech_policy_cfg(),
            last_spoken_at=self.state.last_spoken_at,
            last_spoken_text=self.state.last_spoken_text,
            last_notified_at=self.state.last_notification_at,
            last_notified_text=self.state.last_notification_text,
            last_notified_key=self.state.last_notification_key,
        )
        self.state.last_narration = updated.to_dict()
        self.state.last_narration_delivery = updated.delivery
        self.state.last_narration_channel = updated.channel
        self.state.last_narration_text = updated.text
        if self.state.last_companion_view:
            self.state.last_companion_view["delivery"] = updated.delivery
            self.state.last_companion_view["speakable"] = updated.speakable
        self._emit_status()
        return updated

    def _apply_perception_result(self, perceived: PerceivedGameState) -> dict[str, Any]:
        payload = perceived.to_dict()
        previous_decision = dict(self.state.last_decision) if isinstance(self.state.last_decision, dict) else {}
        previous_decision_at = self.state.last_decision_at
        previous_perception = dict(self.state.last_perception) if isinstance(self.state.last_perception, dict) else {}
        self.state.scene = perceived.scene
        self.state.last_scene = perceived.scene
        self.state.last_scene_confidence = perceived.confidence
        self.state.last_is_user_turn = perceived.is_user_turn
        self.state.last_buttons = list(perceived.buttons)
        self.state.last_perception_at = now_iso()
        self.state.last_perception_ok = True
        self.state.last_perception = perceived.to_dict()
        self._arm_fast_poll_if_needed_locked(buttons=list(perceived.buttons))
        self._clear_decision_state()
        if _has_reusable_fast_action_button_overlays(previous_decision, perceived):
            self.state.last_decision = previous_decision
            self.state.last_decision_ok = True
            self.state.last_decision_at = previous_decision_at
            self.state.last_decision_type = str(previous_decision.get("decision_type", "")).strip()
            self.state.last_decision_risk_level = str(previous_decision.get("risk_level", "")).strip()
        self._clear_narration_state()
        self.state.last_error = ""

        # Clear meld selection pending when:
        # - call/kan buttons reappear (user didn't click)
        # - scene left in_match
        # - hand tiles count dropped by meld amount
        if self._meld_selection_pending:
            current_buttons = set(perceived.buttons)
            call_buttons = {"chi", "pon", "kan"}
            if current_buttons & call_buttons:
                self._clear_meld_selection_pending()
            elif perceived.scene not in {"in_match", "dialog", "unknown"}:
                self._clear_meld_selection_pending()
            else:
                prev_hand = previous_perception.get("hand_tiles") if isinstance(previous_perception.get("hand_tiles"), list) else []
                curr_hand = perceived.hand_tiles if isinstance(perceived.hand_tiles, list) else []
                if prev_hand and curr_hand and len(curr_hand) < len(prev_hand):
                    self._clear_meld_selection_pending()

        return payload

    def _prepare_perceived_state_for_decision(self, perceived: PerceivedGameState) -> PerceivedGameState:
        if not self._preturn_planning_enabled():
            self._clear_preturn_discard_plan_locked()
            return perceived

        prepared, meta = apply_preturn_discard_plan(perceived, self._preturn_discard_plan)
        if meta.get("applied"):
            self._last_preturn_plan_meta = dict(meta)
            return prepared

        plan = build_preturn_discard_plan(perceived)
        if plan is not None:
            self._preturn_discard_plan = plan
            self._last_preturn_plan_meta = {
                "applied": False,
                "planned": True,
                "hand_signature": plan.hand_signature,
                "candidate_count": len(plan.candidate_discards),
                "top_tile": plan.candidate_discards[0].get("tile", "") if plan.candidate_discards else "",
            }
            return perceived

        if perceived.scene != "in_match":
            self._clear_preturn_discard_plan_locked()
        elif perceived.is_user_turn:
            self._last_preturn_plan_meta = dict(meta)
        return perceived

    def _apply_decision_result(self, decision: DecisionResult) -> dict[str, Any]:
        # Preserve fast-path meld_selection decision — the slow path's
        # turn_observe/generic decision should not overwrite it.
        previous = self._current_decision_result()
        if previous is not None and previous.decision_type == "meld_selection":
            previous_payload = self.state.last_decision if isinstance(self.state.last_decision, dict) else {}
            return previous_payload

        call_focuses = {"call_decision", "kan_decision"}
        if decision.recommended_focus in call_focuses:
            previous = self._current_decision_result()
            if (
                previous is not None
                and previous.recommended_focus == decision.recommended_focus
                and set(previous.buttons) == set(decision.buttons)
            ):
                prev_meta = previous.engine_meta if isinstance(previous.engine_meta, dict) else {}
                prev_recommended = prev_meta.get("recommended_button_types")
                new_meta = decision.engine_meta if isinstance(decision.engine_meta, dict) else {}
                new_recommended = new_meta.get("recommended_button_types")
                if (
                    isinstance(prev_recommended, list)
                    and isinstance(new_recommended, list)
                    and prev_recommended != new_recommended
                ):
                    decision.engine_meta = {
                        **dict(new_meta),
                        "recommended_button_types": list(prev_recommended),
                    }
                    decision.review_tags = list(previous.review_tags)

        payload = decision.to_dict()
        analysis = decision.mahjong_analysis if isinstance(decision.mahjong_analysis, dict) else {}
        overlays = self._build_decision_overlays(decision)
        overlay_updated_at = time.monotonic() if overlays else 0.0
        if overlays:
            overlays = [
                {
                    **dict(overlay),
                    "created_at_monotonic": overlay_updated_at,
                }
                for overlay in overlays
                if isinstance(overlay, dict)
            ]
            payload["screen_overlays"] = overlays
            decision.engine_meta = {
                **dict(decision.engine_meta),
                "screen_overlays": overlays,
                "screen_overlay_count": len(overlays),
            }
            payload = decision.to_dict()
            payload["screen_overlays"] = overlays
        self._last_screen_overlay_update_at = overlay_updated_at
        self._arm_fast_poll_if_needed_locked(
            buttons=list(decision.buttons),
            focus=decision.recommended_focus,
        )
        self.state.last_decision_at = now_iso()
        self.state.last_decision_ok = True
        self.state.last_decision_type = decision.decision_type
        self.state.last_decision_risk_level = decision.risk_level
        self.state.last_tile_analysis_available = bool(analysis.get("tile_level_available", False))
        self.state.last_shanten_estimate = analysis.get("shanten_estimate")
        self.state.last_ukeire_estimate = analysis.get("ukeire_estimate")
        self.state.last_decision = decision.to_dict()
        self._clear_narration_state()
        self.state.last_error = ""

        # Arm meld selection detection when a call/kan decision is active.
        if decision.recommended_focus in {"call_decision", "kan_decision"}:
            meta = decision.engine_meta if isinstance(decision.engine_meta, dict) else {}
            recommended_buttons = meta.get("recommended_button_types")
            if isinstance(recommended_buttons, list) and recommended_buttons:
                meld_type = str(recommended_buttons[0] or "").strip()
                if meld_type in {"chi", "pon", "kan"}:
                    self._set_meld_selection_pending(meld_type)

        return payload

    def _build_decision_overlays(self, decision: DecisionResult) -> list[dict[str, Any]]:
        button_overlays = self._build_action_button_overlays(decision)
        if button_overlays:
            return button_overlays
        if decision.decision_type != "tile_efficiency_hint":
            return []
        analysis = decision.mahjong_analysis if isinstance(decision.mahjong_analysis, dict) else {}
        candidates = analysis.get("candidate_discards") if isinstance(analysis.get("candidate_discards"), list) else []
        top_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        tile = str(top_candidate.get("tile", "")).strip()
        if not tile:
            return []
        perceived = self._current_perceived_state()
        if perceived is None:
            return []
        local_box = self._find_hand_tile_box(perceived, tile)
        if not local_box:
            return []
        screen_box = self._local_box_to_screen_box(local_box)
        if not screen_box:
            return []
        frame_path = self._resolve_latest_frame_path()
        region_signature = _image_region_signature(frame_path, local_box) if frame_path is not None else ""
        frame_mtime = _path_mtime(frame_path) if frame_path is not None else 0.0
        hand_tile_count = len(perceived.hand_tiles)
        return [
            {
                "kind": "discard_recommendation",
                "tile": tile,
                "label": str(top_candidate.get("label") or ""),
                "box": screen_box,
                "local_box": dict(local_box),
                "frame_path": str(frame_path) if frame_path is not None else "",
                "frame_mtime": frame_mtime,
                "region_signature": region_signature,
                "source": "hand_tile_slots",
                "hand_tile_count": hand_tile_count,
            }
        ]

    def _build_action_button_overlays(self, decision: DecisionResult) -> list[dict[str, Any]]:
        engine_meta = decision.engine_meta if isinstance(decision.engine_meta, dict) else {}
        recommended_buttons = engine_meta.get("recommended_button_types")
        if not isinstance(recommended_buttons, list):
            return []
        perceived = self._current_perceived_state()
        if perceived is None:
            return []
        regions = perceived.button_regions if isinstance(perceived.button_regions, list) else []
        frame_path = self._resolve_latest_frame_path()
        frame_mtime = _path_mtime(frame_path) if frame_path is not None else 0.0
        overlays: list[dict[str, Any]] = []
        for button_type in recommended_buttons:
            clean_type = str(button_type or "").strip()
            if not clean_type:
                continue
            region = self._find_button_region(regions, clean_type)
            if region is None:
                continue
            local_box = self._button_region_to_local_box(region)
            if not local_box:
                continue
            screen_box = self._local_box_to_screen_box(local_box)
            if not screen_box:
                continue
            region_signature = _image_region_signature(frame_path, local_box) if frame_path is not None else ""
            overlays.append({
                "kind": "action_button_recommendation",
                "button_type": clean_type,
                "label": str(region.get("label") or clean_type),
                "box": screen_box,
                "local_box": local_box,
                "frame_path": str(frame_path) if frame_path is not None else "",
                "frame_mtime": frame_mtime,
                "region_signature": region_signature,
                "confidence": region.get("confidence"),
                "template_id": str(region.get("template_id") or ""),
                "source": "button_regions",
            })
            break
        return overlays

    def _find_button_region(self, regions: list[Any], button_type: str) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_confidence = float("-inf")
        for region in regions:
            if not isinstance(region, dict):
                continue
            if str(region.get("button_type", "")).strip() != button_type:
                continue
            try:
                confidence = float(region.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if best is None or confidence > best_confidence:
                best = region
                best_confidence = confidence
        return best

    def _button_region_to_local_box(self, region: dict[str, Any]) -> dict[str, int]:
        bbox = region.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            return {}
        try:
            left, top, right, bottom = [int(value) for value in bbox]
        except (TypeError, ValueError):
            return {}
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return {}
        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

    def _find_hand_tile_box(self, perceived: PerceivedGameState, tile: str) -> dict[str, Any]:
        hints = perceived.analysis_hints if isinstance(perceived.analysis_hints, dict) else {}
        slots = hints.get("hand_tile_slots") if isinstance(hints.get("hand_tile_slots"), list) else []
        best_slot: dict[str, Any] | None = None
        for slot in slots:
            if not isinstance(slot, dict) or str(slot.get("tile", "")).strip() != tile:
                continue
            if best_slot is None or int(slot.get("index", 0) or 0) > int(best_slot.get("index", 0) or 0):
                best_slot = slot
        if best_slot is not None and isinstance(best_slot.get("box"), dict):
            return dict(best_slot["box"])

        for detection in perceived.raw_detections:
            if not isinstance(detection, dict):
                continue
            if detection.get("group") != "hand" or detection.get("accepted", True) is False:
                continue
            if str(detection.get("candidate_tile", "")).strip() != tile:
                continue
            box = detection.get("box")
            if isinstance(box, dict):
                return dict(box)
        return {}

    def _local_box_to_screen_box(self, box: dict[str, Any]) -> dict[str, int]:
        try:
            left = int(box.get("left", 0) or 0)
            top = int(box.get("top", 0) or 0)
            width = int(box.get("width", 0) or 0)
            height = int(box.get("height", 0) or 0)
        except (TypeError, ValueError):
            return {}
        if width <= 0 or height <= 0:
            return {}
        window_left = self.state.window_left if isinstance(self.state.window_left, int) else 0
        window_top = self.state.window_top if isinstance(self.state.window_top, int) else 0
        return {
            "left": window_left + left,
            "top": window_top + top,
            "width": width,
            "height": height,
        }

    def _apply_narration_result(self, event: Any, view_model: Any) -> dict[str, Any]:
        event_payload = event.to_dict()
        view_payload = view_model.to_dict()
        self.state.last_narration_at = now_iso()
        self.state.last_narration_ok = True
        self.state.last_narration_type = event.event_type
        self.state.last_narration_channel = event.channel
        self.state.last_narration_delivery = event.delivery
        self.state.last_narration_text = event.text
        self.state.last_narration = event.to_dict()
        self.state.last_companion_mood = view_model.mood
        self.state.last_companion_view = view_model.to_dict()
        self.state.last_error = ""
        return {
            **event_payload,
            "companion_view": view_payload,
        }

    def load_cached_outputs(self) -> None:
        return None

    def _load_cache_json(self, path: Path) -> dict[str, Any]:
        return load_json_payload(path, default={}, expected_type=dict, logger=self.logger)

    def _dispatch_narration_locked(self, event: NarrationEvent) -> dict[str, Any]:
        if event.delivery not in {"proactive_notification", "voice_candidate"}:
            return {
                "ok": False,
                "skipped": True,
                "reason": "delivery_suppressed",
                "delivery": event.delivery,
            }
        return self._narration_dispatcher.dispatch(
            event,
            state=self.state,
            emit_status=self._emit_status,
            target_lanlan=self._get_voice_target_lanlan(),
            require_running=True,
            require_window_bound=True,
        )

    def _dispatch_debug_narration_locked(self, event: NarrationEvent) -> dict[str, Any]:
        self._narration_dispatcher.apply_debug_reply_event(event, state=self.state)
        return self._narration_dispatcher.dispatch(
            event,
            state=self.state,
            emit_status=self._emit_status,
            target_lanlan=self._get_voice_target_lanlan(),
            require_running=False,
            require_window_bound=False,
        )

    def _build_debug_reply_event(self, event: NarrationEvent) -> NarrationEvent:
        return self._narration_dispatcher.build_debug_reply_event(event)

    def _apply_debug_reply_event(self, event: NarrationEvent) -> None:
        self._narration_dispatcher.apply_debug_reply_event(event, state=self.state)

    def _mark_perception_failure(self, error: str) -> None:
        self.state.scene = "unknown"
        self.state.last_scene = "unknown"
        self.state.last_scene_confidence = 0.0
        self.state.last_is_user_turn = False
        self.state.last_buttons = []
        self.state.last_perception_at = now_iso()
        self.state.last_perception_ok = False
        self.state.last_perception = {}
        self._clear_decision_state()
        self._clear_narration_state()
        self._clear_preturn_discard_plan_locked()
        self.state.last_error = error

    def _mark_decision_failure(self, error: str) -> None:
        self._clear_decision_state()
        self._clear_narration_state()
        self.state.last_error = error

    def _mark_narration_failure(self, error: str) -> None:
        self._clear_narration_state()
        self.state.last_error = error

    def _clear_binding(self) -> None:
        self.state.window_bound = False
        self.state.window_title = ""
        self.state.window_match_keyword = ""
        self.state.window_left = None
        self.state.window_top = None
        self.state.window_width = None
        self.state.window_height = None

    def _clear_perception_state(self) -> None:
        self.state.scene = "unknown"
        self.state.last_scene = "unknown"
        self.state.last_scene_confidence = 0.0
        self.state.last_is_user_turn = False
        self.state.last_buttons = []
        self.state.last_perception_at = ""
        self.state.last_perception_ok = False
        self.state.last_perception = {}

    def _clear_decision_state(self) -> None:
        self.state.last_decision_at = ""
        self.state.last_decision_ok = False
        self.state.last_decision_type = ""
        self.state.last_decision_risk_level = ""
        self.state.last_tile_analysis_available = False
        self.state.last_shanten_estimate = None
        self.state.last_ukeire_estimate = None
        self.state.last_decision = {}

    def _clear_preturn_discard_plan_locked(self) -> None:
        self._preturn_discard_plan = None
        self._last_preturn_plan_meta = {}
        self._last_fast_advice_frame_path = ""

    def _clear_narration_state(self) -> None:
        self.state.last_narration_at = ""
        self.state.last_narration_ok = False
        self.state.last_narration_type = ""
        self.state.last_narration_channel = ""
        self.state.last_narration_delivery = ""
        self.state.last_narration_text = ""
        self.state.last_narration = {}
        self.state.last_companion_mood = "calm"
        self.state.last_companion_view = {}
        self.state.last_speak_ok = False

def _image_region_signature(frame_path: Path, box: dict[str, Any]) -> str:
    try:
        left = int(box.get("left", 0) or 0)
        top = int(box.get("top", 0) or 0)
        width = int(box.get("width", 0) or 0)
        height = int(box.get("height", 0) or 0)
    except (TypeError, ValueError):
        return ""
    if width <= 0 or height <= 0:
        return ""
    try:
        with Image.open(frame_path) as opened:
            image_width, image_height = opened.size
            inset_x = min(max(2, int(width * 0.16)), max(0, (width - 1) // 2))
            inset_y = min(max(2, int(height * 0.12)), max(0, (height - 1) // 2))
            crop_box = (
                max(0, left + inset_x),
                max(0, top + inset_y),
                min(image_width, left + width - inset_x),
                min(image_height, top + height - inset_y),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                return ""
            region = opened.convert("L").crop(crop_box).resize((8, 8))
            pixels = list(region.getdata())
    except Exception:
        return ""
    return bytes(int(pixel) for pixel in pixels).hex()


def _has_fast_action_button_overlays(decision: dict[str, Any]) -> bool:
    overlays = decision.get("screen_overlays")
    if not isinstance(overlays, list):
        engine_meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
        overlays = engine_meta.get("screen_overlays")
    if not isinstance(overlays, list):
        return False
    return any(
        isinstance(overlay, dict)
        and overlay.get("kind") in {"action_button_recommendation", "meld_selection_recommendation"}
        and bool(overlay.get("fast_path") or overlay.get("source") == "meld_selection_scan")
        for overlay in overlays
    )


def _has_reusable_fast_action_button_overlays(decision: dict[str, Any], perceived: PerceivedGameState) -> bool:
    overlays = decision.get("screen_overlays")
    if not isinstance(overlays, list):
        engine_meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
        overlays = engine_meta.get("screen_overlays")
    if not isinstance(overlays, list):
        return False

    visible_buttons = {str(button).strip() for button in perceived.buttons if str(button).strip()}
    reusable = False
    for overlay in overlays:
        if not isinstance(overlay, dict):
            continue
        kind = overlay.get("kind")
        if kind == "meld_selection_recommendation" and overlay.get("source") == "meld_selection_scan":
            reusable = True
            continue
        if kind == "action_button_recommendation" and bool(overlay.get("fast_path")):
            button_type = str(overlay.get("button_type", "")).strip()
            if button_type and button_type in visible_buttons:
                reusable = True
            continue
    return reusable


def _image_region_signature_distance(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    try:
        left_bytes = bytes.fromhex(left)
        right_bytes = bytes.fromhex(right)
    except ValueError:
        return 0.0
    if len(left_bytes) != len(right_bytes):
        return 0.0
    return sum(abs(a - b) for a, b in zip(left_bytes, right_bytes)) / max(1, len(left_bytes))
