from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..narration import NarrationEvent
from ..review import (
    append_review_summary_history,
    generate_review_summary as generate_review_summary_artifact,
    sync_memory_bridge_queue,
)
from ..session_state import now_iso


@dataclass(frozen=True)
class RuntimeCycleOutcome:
    mode: str
    processed_command: bool = False
    processed_overlay_command: bool = False
    consumed_live_tick: bool = False
    flushed_outbound: int = 0
    reason: str = ""


class RuntimeSchedulerMixin:
    def _run_runtime_cycle_locked(self) -> RuntimeCycleOutcome:
        overlay_forced_refresh = self._process_overlay_commands_locked()
        processed_command = self._process_runtime_command_locked()

        runtime_mode = self._game_runtime.set_mode(self.state.runtime_mode)
        self.state.runtime_mode = runtime_mode
        if overlay_forced_refresh:
            flushed = self._flush_runtime_outbox_locked(limit=self._runtime_outbox_flush_per_tick)
            self._emit_status()
            return RuntimeCycleOutcome(
                mode=runtime_mode,
                processed_command=processed_command,
                processed_overlay_command=True,
                consumed_live_tick=True,
                flushed_outbound=len(flushed),
                reason="overlay_refresh",
            )
        if runtime_mode == "off":
            self.state.runtime_status = "off"
            self._game_runtime.set_status("off")
            self.state.status = "idle"
            flushed = self._flush_runtime_outbox_locked(limit=self._runtime_outbox_flush_per_tick)
            self._emit_status()
            return RuntimeCycleOutcome(
                mode=runtime_mode,
                processed_command=processed_command,
                flushed_outbound=len(flushed),
                reason="runtime_off",
            )

        if runtime_mode == "standby":
            self.state.runtime_status = "standby"
            self._game_runtime.set_status("standby")
            if self.state.running and self.state.status not in {"warning", "error"}:
                self.state.status = "standby"
            flushed = self._flush_runtime_outbox_locked(limit=self._runtime_outbox_flush_per_tick)
            self._emit_status()
            return RuntimeCycleOutcome(
                mode=runtime_mode,
                processed_command=processed_command,
                flushed_outbound=len(flushed),
                reason="runtime_standby",
            )

        self.state.runtime_status = "active"
        self._game_runtime.set_status("active")
        self._run_live_cycle_locked()
        flushed = self._flush_runtime_outbox_locked(limit=self._runtime_outbox_flush_per_tick)
        return RuntimeCycleOutcome(
            mode=runtime_mode,
            processed_command=processed_command,
            consumed_live_tick=True,
            flushed_outbound=len(flushed),
            reason="runtime_active",
        )

    def _process_runtime_command_locked(self) -> bool:
        command = self._game_runtime.pop_inbound()
        if command is None:
            self._sync_runtime_mailbox_state_locked()
            return False

        self.state.last_runtime_command_id = command.message_id
        self.state.last_runtime_command_source = command.source
        self.state.last_runtime_command_action = command.action
        self.state.last_runtime_command_at = now_iso()
        self.state.runtime_status = "processing_command"
        result: dict[str, Any]

        try:
            result = self._handle_runtime_command_locked(command.action, command.payload)
            ok = bool(result.get("ok", True))
            self.state.last_runtime_command_ok = ok
            self.state.last_runtime_command_result = dict(result)
            if not ok and result.get("error"):
                self.state.last_error = str(result.get("error"))
        except Exception as exc:
            self.logger.exception("runtime command failed")
            self.state.last_runtime_command_ok = False
            self.state.last_runtime_command_result = {
                "ok": False,
                "error": str(exc),
            }
            self.state.last_error = str(exc)
        finally:
            self._sync_runtime_mailbox_state_locked()
            if self.state.runtime_mode != "off":
                self.state.runtime_status = self.state.runtime_mode
            else:
                self.state.runtime_status = "off"
        return True

    def _handle_runtime_command_locked(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._command_registry.dispatch(action, payload)

    def _build_runtime_command_registry(self) -> None:
        self._command_registry.register("refresh_status", self._runtime_cmd_refresh_status)
        self._command_registry.register("set_mode", self._runtime_cmd_set_mode)
        self._command_registry.register("set_runtime_mode", self._runtime_cmd_set_runtime_mode)
        self._command_registry.register("summarize_review", self._runtime_cmd_summarize_review)
        self._command_registry.register("sync_memory", self._runtime_cmd_sync_memory)
        self._command_registry.register(
            "dispatch_current_narration",
            self._runtime_cmd_dispatch_current_narration,
        )
        self._command_registry.register("explain_current_hand", self._runtime_cmd_explain_current_hand)

    def _runtime_cmd_refresh_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "status": self.get_status()}

    def _runtime_cmd_set_mode(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "")).strip()
        if mode not in {"spectate", "replay", "teaching", "silent"}:
            return {"ok": False, "error": f"invalid mode: {mode}"}
        self.state.mode = mode
        self._emit_status()
        return {"ok": True, "mode": mode}

    def _runtime_cmd_set_runtime_mode(self, payload: dict[str, Any]) -> dict[str, Any]:
        runtime_mode = str(payload.get("runtime_mode", "")).strip().lower()
        if runtime_mode not in {"active", "standby", "off"}:
            return {
                "ok": False,
                "error": f"invalid runtime mode: {runtime_mode}",
            }
        self.state.runtime_mode = self._game_runtime.set_mode(runtime_mode)
        if runtime_mode == "off":
            self.state.status = "idle"
        return {"ok": True, "runtime_mode": self.state.runtime_mode}

    def _runtime_cmd_summarize_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        cache_dir = self.plugin.data_path("session_cache")
        summary, summary_path = generate_review_summary_artifact(
            cache_dir,
            session_id=self.state.session_id,
        )
        history_path = append_review_summary_history(
            cache_dir,
            summary,
            limit=int(self._get_coaching_cfg().get("history_limit", 24)),
        )
        payload_result = self._apply_review_summary_result(summary)
        payload_result.update(self._refresh_coaching_state_locked(cache_dir))
        payload_result["path"] = str(summary_path)
        payload_result["history_path"] = str(history_path)
        self._emit_status()
        return {"ok": True, **payload_result}

    def _runtime_cmd_sync_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        cache_dir = self.plugin.data_path("session_cache")
        bridge_cfg = self._get_memory_bridge_cfg()
        report, report_path = sync_memory_bridge_queue(
            cache_dir,
            writer=self._host_memory_writer,
            bucket_id=str(bridge_cfg.get("host_memory_bucket_id", "mahjong_companion_coaching")),
            batch_size=int(bridge_cfg.get("host_sync_batch_size", 5)),
        )
        self._apply_host_memory_sync_result(report)
        self._emit_status()
        return {"ok": True, **report, "path": str(report_path)}

    def _runtime_cmd_dispatch_current_narration(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = self._current_narration_event()
        if event is None:
            return {"ok": False, "error": "no narration available"}
        dispatch = self._dispatch_narration_locked(event)
        return {"ok": bool(dispatch.get("ok")), "dispatch": dispatch}

    def _runtime_cmd_explain_current_hand(self, payload: dict[str, Any]) -> dict[str, Any]:
        ready, perception_payload = self._ensure_perception_locked()
        if not ready:
            return {"ok": False, "error": perception_payload.get("error", "perception_unavailable")}
        decision_payload = self._generate_decision_locked(persist_review_artifacts=False)
        if not decision_payload.get("ok"):
            return {"ok": False, "error": decision_payload.get("error", "decision_failed")}
        narration_payload = self._generate_narration_locked()
        if not narration_payload.get("ok"):
            return {"ok": False, "error": narration_payload.get("error", "narration_failed")}
        return {
            "ok": True,
            "perception": perception_payload,
            "decision": decision_payload,
            "narration": narration_payload,
        }

    def _queue_runtime_outbound_event_locked(
        self,
        event: NarrationEvent,
        *,
        require_running: bool,
        require_window_bound: bool,
    ) -> None:
        envelope = self._game_runtime.enqueue_outbound(
            event_type="narration_event",
            payload={
                "event": event.to_dict(),
                "require_running": bool(require_running),
                "require_window_bound": bool(require_window_bound),
                "target_lanlan": self._get_voice_target_lanlan(),
            },
            priority=int(event.priority),
            dedupe_key=str(event.dedupe_key or "").strip(),
        )
        if envelope is None:
            self._sync_runtime_mailbox_state_locked()
            return
        self.state.last_runtime_outbound_id = envelope.message_id
        self.state.last_runtime_outbound_at = now_iso()
        self._sync_runtime_mailbox_state_locked()

    def _flush_runtime_outbox_locked(self, *, limit: int = 1) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []
        messages = self._game_runtime.pop_outbound_batch(limit=limit)
        for message in messages:
            if message.event_type != "narration_event":
                responses.append({
                    "ok": False,
                    "skipped": True,
                    "reason": f"unsupported_outbound_event:{message.event_type}",
                    "queued_message_id": message.message_id,
                })
                continue

            payload = message.payload if isinstance(message.payload, dict) else {}
            event_payload = payload.get("event", {})
            if not isinstance(event_payload, dict):
                responses.append({
                    "ok": False,
                    "skipped": True,
                    "reason": "invalid_narration_payload",
                    "queued_message_id": message.message_id,
                })
                continue

            try:
                event = NarrationEvent(**event_payload)
                response = self._narration_dispatcher.dispatch(
                    event,
                    state=self.state,
                    emit_status=self._emit_status,
                    target_lanlan=str(payload.get("target_lanlan", "")).strip(),
                    require_running=bool(payload.get("require_running", True)),
                    require_window_bound=bool(payload.get("require_window_bound", True)),
                )
            except Exception as exc:
                self.logger.exception("runtime outbox dispatch failed")
                response = {
                    "ok": False,
                    "error": str(exc),
                }

            response["queued_message_id"] = message.message_id
            responses.append(response)

        self._sync_runtime_mailbox_state_locked()
        return responses

    def _sync_runtime_mailbox_state_locked(self) -> None:
        snapshot = self._game_runtime.snapshot()
        self.state.runtime_inbound_pending = int(snapshot.get("inbound_pending", 0) or 0)
        self.state.runtime_outbound_pending = int(snapshot.get("outbound_pending", 0) or 0)
        self.state.runtime_dropped_inbound = int(snapshot.get("dropped_inbound", 0) or 0)
        self.state.runtime_dropped_outbound = int(snapshot.get("dropped_outbound", 0) or 0)
        self.state.runtime_deduped_outbound = int(snapshot.get("deduped_outbound", 0) or 0)
