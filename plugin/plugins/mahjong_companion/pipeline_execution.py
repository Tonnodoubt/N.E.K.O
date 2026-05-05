from __future__ import annotations

from pathlib import Path
from typing import Any

from .decision.debug_dump import write_debug_artifacts as write_decision_debug_artifacts
from .narration import apply_speech_policy
from .narration.debug_dump import write_debug_artifacts as write_narration_debug_artifacts
from .perception.debug_dump import write_debug_artifacts as write_perception_debug_artifacts
from .review import append_review_candidate, build_memory_summary, build_review_candidate, stage_memory_summary


class PipelineExecutionMixin:
    def _run_companion_pipeline_locked(
        self,
        frame_path: str,
        *,
        capture: bool,
        dispatch: bool,
        force_reply: bool,
        persist_review_artifacts: bool | None = None,
    ) -> dict[str, Any]:
        target_frame = self._resolve_pipeline_frame_path(frame_path, capture)
        if isinstance(target_frame, dict):
            return target_frame

        perception = self._analyze_frame_locked(target_frame)
        if not perception.get("ok"):
            return {
                "ok": False,
                "stage": "perception",
                "frame_path": str(target_frame),
                "perception": perception,
            }

        decision = self._generate_decision_locked(
            persist_review_artifacts=persist_review_artifacts,
            review_frame_path=target_frame,
        )
        if not decision.get("ok"):
            return {
                "ok": False,
                "stage": "decision",
                "frame_path": str(target_frame),
                "perception": perception,
                "decision": decision,
            }

        narration = self._generate_narration_locked()
        if not narration.get("ok"):
            return {
                "ok": False,
                "stage": "narration",
                "frame_path": str(target_frame),
                "perception": perception,
                "decision": decision,
                "narration": narration,
            }

        dispatch_payload = {
            "ok": False,
            "skipped": True,
            "reason": "dispatch_disabled",
        }
        dispatch_event = self._current_narration_event()
        if dispatch and dispatch_event is not None:
            if force_reply and dispatch_event.delivery not in {"proactive_notification", "voice_candidate"}:
                dispatch_event = self._build_debug_reply_event(dispatch_event)
            dispatch_payload = (
                self._dispatch_debug_narration_locked(dispatch_event)
                if force_reply
                else self._dispatch_narration_locked(dispatch_event)
            )

        return {
            "ok": True,
            "frame_path": str(target_frame),
            "perception": perception,
            "decision": decision,
            "narration": narration,
            "dispatch": dispatch_payload,
        }

    def _speak_last_narration_locked(self) -> dict[str, Any]:
        ready, payload = self._ensure_narration_locked()
        if not ready:
            return payload

        if not self.state.last_narration_ok or not self.state.last_narration_text:
            return {
                "ok": False,
                "error": "no narration available",
            }

        if not self.state.running:
            return {
                "ok": False,
                "error": "session is not running",
            }

        binding_result = self._bind_window()
        if not binding_result.bound:
            self._emit_status()
            return {
                "ok": False,
                "error": "mahjong window is not currently bound",
                "window_title": self.state.window_title,
                "binding_error": binding_result.error,
            }

        event = self._reapply_current_narration_policy_locked()
        if event is None:
            return {
                "ok": False,
                "error": "no narration available",
            }

        if event.delivery != "voice_candidate":
            return {
                "ok": False,
                "error": "current narration is not eligible for voice playback",
                "delivery": event.delivery,
                "voice_mode": self.state.voice_mode,
            }

        try:
            dispatch = self._dispatch_narration_locked(event)
            if not dispatch.get("ok"):
                return dispatch
            return {
                "ok": True,
                "spoken": True,
                "text": event.text,
                "voice_mode": self.state.voice_mode,
                "delivery": event.delivery,
            }
        except Exception as exc:
            self.logger.exception("speak_last_narration failed")
            self.state.last_speak_ok = False
            self.state.last_notification_ok = False
            self.state.last_error = str(exc)
            self._emit_status()
            return {
                "ok": False,
                "error": str(exc),
            }

    def _analyze_frame_locked(self, frame_path: Path, *, live: bool = False) -> dict[str, Any]:
        if not frame_path.exists():
            self._mark_perception_failure("image not found: %s" % frame_path)
            self._emit_status()
            return {
                "ok": False,
                "error": "image not found: %s" % frame_path,
                "frame_path": str(frame_path),
            }

        try:
            try:
                perceived, debug_payload = self._perception_adapter.analyze(frame_path, live=live)
            except TypeError:
                perceived, debug_payload = self._perception_adapter.analyze(frame_path)
            perceived = self._prepare_perceived_state_for_decision(perceived)
            artifacts = {}
            if self._perception_debug_dump_enabled() and self._debug_artifacts_allowed(frame_path):
                artifacts = write_perception_debug_artifacts(frame_path, perceived, debug_payload)
            payload = self._apply_perception_result(perceived)
            payload.update(artifacts)
            payload["ok"] = True
            payload["frame_path"] = str(frame_path)
            self._emit_status()
            return payload
        except Exception as exc:
            self.logger.exception("analyze_debug_frame failed")
            self._mark_perception_failure(str(exc))
            self._emit_status()
            return {
                "ok": False,
                "error": str(exc),
                "frame_path": str(frame_path),
            }

    def _generate_decision_locked(
        self,
        *,
        persist_review_artifacts: bool | None = None,
        review_frame_path: Path | None = None,
    ) -> dict[str, Any]:
        perceived = self._current_perceived_state()
        if perceived is None:
            self._mark_decision_failure("no perception available")
            self._emit_status()
            return {
                "ok": False,
                "error": "no perception available",
            }

        try:
            decision = self._decision_adapter.suggest(perceived)
            debug_payload = {
                "source_scene": perceived.scene,
                "source_confidence": perceived.confidence,
                "source_buttons": list(perceived.buttons),
                "source_is_user_turn": perceived.is_user_turn,
                "source_notes": list(perceived.notes),
                "decision_reason_codes": list(decision.reason_codes),
            }
            artifacts = {}
            frame_path = self._resolve_latest_frame_path()
            if frame_path is not None and self._decision_debug_dump_enabled() and self._debug_artifacts_allowed(frame_path):
                artifacts = write_decision_debug_artifacts(frame_path, decision, debug_payload)
            payload = self._apply_decision_result(decision)
            payload.update(artifacts)
            should_persist_review = self.state.running if persist_review_artifacts is None else persist_review_artifacts
            if should_persist_review:
                self._persist_review_artifacts_locked(
                    decision,
                    perceived,
                    payload,
                    frame_path=review_frame_path,
                )
            payload["ok"] = True
            self._emit_status()
            return payload
        except Exception as exc:
            self.logger.exception("generate_decision failed")
            self._mark_decision_failure(str(exc))
            self._emit_status()
            return {
                "ok": False,
                "error": str(exc),
            }

    def _persist_review_artifacts_locked(
        self,
        decision: Any,
        perceived: Any,
        payload: dict[str, Any],
        *,
        frame_path: Path | None = None,
    ) -> None:
        frame_path = frame_path or self._resolve_latest_frame_path()
        candidate = build_review_candidate(frame_path, decision, perceived)
        if candidate is None:
            return

        candidate["session_id"] = self.state.session_id
        cache_dir = self.plugin.data_path("session_cache")
        review_path = append_review_candidate(cache_dir, candidate)
        payload["review_candidates_path"] = str(review_path)
        payload["review_candidate_appended"] = bool(getattr(append_review_candidate, "last_appended", False))

        memory_summary = build_memory_summary(candidate, decision, perceived)
        if memory_summary is None:
            return
        payload["memory_bridge"] = stage_memory_summary(cache_dir, memory_summary)

    def _generate_narration_locked(self) -> dict[str, Any]:
        decision = self._current_decision_result()
        if decision is None:
            self._mark_narration_failure("no decision available")
            self._emit_status()
            return {
                "ok": False,
                "error": "no decision available",
            }

        try:
            event, view_model, debug_payload = self._narration_adapter.generate(decision)
            event = apply_speech_policy(
                event,
                self._get_speech_policy_cfg(),
                last_spoken_at=self.state.last_spoken_at,
                last_spoken_text=self.state.last_spoken_text,
                last_notified_at=self.state.last_notification_at,
                last_notified_text=self.state.last_notification_text,
                last_notified_key=self.state.last_notification_key,
            )
            view_model.delivery = event.delivery
            view_model.speakable = event.speakable
            artifacts = {}
            frame_path = self._resolve_latest_frame_path()
            if frame_path is not None and self._narration_debug_dump_enabled() and self._debug_artifacts_allowed(frame_path):
                artifacts = write_narration_debug_artifacts(frame_path, event, view_model, debug_payload)
            payload = self._apply_narration_result(event, view_model)
            payload.update(artifacts)
            payload["ok"] = True
            self._emit_status()
            return payload
        except Exception as exc:
            self.logger.exception("generate_narration failed")
            self._mark_narration_failure(str(exc))
            self._emit_status()
            return {
                "ok": False,
                "error": str(exc),
            }

    def _run_live_cycle_locked(self) -> None:
        binding_result = self._bind_window()
        if not binding_result.bound:
            self.state.status = "warning"
            self.state.last_error = binding_result.error or "mahjong window is not currently bound"
            self._emit_status()
            return

        try:
            self._capture_debug_frame_locked(binding_result)
        except Exception as exc:
            self.logger.exception("automatic capture failed")
            self._handle_capture_failure_locked(exc)
            self._emit_status()
            return

        frame_path = self._resolve_latest_frame_path()
        if frame_path is None:
            self._mark_perception_failure("no captured frame available")
            self._emit_status()
            return

        self._maybe_emit_fast_action_button_overlay_locked(frame_path)
        self._maybe_emit_fast_meld_selection_locked(frame_path)
        self._maybe_emit_fast_preturn_advice_locked(frame_path)

        if not self._should_process_frame_locked(frame_path):
            if self._maybe_clear_expired_screen_overlays_locked():
                self._emit_status()
            if self.state.running and self.state.status != "warning":
                self.state.status = "scanning"
                self._emit_status()
            return

        if self._perception_enabled():
            perception = self._analyze_frame_locked(frame_path, live=True)
            if not perception.get("ok"):
                return

        if self._decision_enabled():
            decision = self._generate_decision_locked()
            if not decision.get("ok"):
                return

        if self._narration_enabled():
            narration = self._generate_narration_locked()
            if not narration.get("ok"):
                return
            event = self._current_narration_event()
            if event is not None and self._auto_dispatch_enabled():
                self._pending_fast_dispatch_events.append(event)

        if self.state.running and self.state.status != "warning":
            self.state.status = "scanning"
            self._emit_status()

    def _handle_capture_failure_locked(self, exc: Exception) -> bool:
        self.state.last_capture_ok = False
        self.state.last_capture_source = ""
        self.state.last_error = str(exc)
        self._consecutive_capture_failures += 1
        return self._apply_failure_degrade()

    def _ensure_perception_locked(self) -> tuple[bool, dict[str, Any]]:
        if self.state.last_perception_ok and self.state.last_perception:
            return True, self.state.last_perception

        frame_path = self._resolve_latest_frame_path()
        if frame_path is None:
            return False, {
                "ok": False,
                "error": "no perception available and no captured frame available",
            }
        payload = self._analyze_frame_locked(frame_path)
        if not payload.get("ok"):
            return False, payload
        return True, payload

    def _ensure_decision_locked(self) -> tuple[bool, dict[str, Any]]:
        if self.state.last_decision_ok and self.state.last_decision:
            return True, self.state.last_decision
        ready, payload = self._ensure_perception_locked()
        if not ready:
            return False, payload
        payload = self._generate_decision_locked()
        if not payload.get("ok"):
            return False, payload
        return True, payload

    def _ensure_narration_locked(self) -> tuple[bool, dict[str, Any]]:
        if self.state.last_narration_ok and self.state.last_narration:
            return True, self.state.last_narration
        ready, payload = self._ensure_decision_locked()
        if not ready:
            return False, payload
        payload = self._generate_narration_locked()
        if not payload.get("ok"):
            return False, payload
        return True, payload
