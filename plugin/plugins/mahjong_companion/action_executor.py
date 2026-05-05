from __future__ import annotations

from typing import Any

from .action import LocatedAction
from .action.action_log import ActionLogEntry, append_action_log
from .contracts import ActionExecutionResult
from .session_state import now_iso
from .window_binding import WindowBindingResult


class ActionExecutorMixin:
    def _action_risk_level(self, action_id: str) -> str:
        action = self._action_registry.get_action(action_id)
        if action is None:
            return "safe"
        return action.risk_level or "safe"

    def _confirmation_chain_step(self, step: str, value: Any) -> dict[str, Any]:
        return {
            "step": step,
            "at": now_iso(),
            "value": value,
        }

    def _resolve_located_action_locked(
        self,
        action_id: str,
        *,
        binding: WindowBindingResult,
    ) -> LocatedAction | None:
        perceived = self.state.last_perception if isinstance(self.state.last_perception, dict) else None
        located = self._button_candidate_locator.locate(
            action_id,
            binding=binding,
            perceived=perceived,
        )
        if located is not None:
            return located
        return self._fixed_offset_locator.locate(
            action_id,
            binding=binding,
            perceived=perceived,
        )

    def _execute_assist_action_locked(
        self,
        action_id: str,
        *,
        dry_run: bool = False,
        user_confirmed: bool = False,
    ) -> dict[str, Any]:
        """Core execution logic for assist actions."""
        self._human_override_guard.configure_pointer_provider(self._input_adapter.get_pointer)
        current_scene = self.state.scene or "unknown"
        action_policy_cfg = self._get_action_policy_cfg()
        risk_level = self._action_risk_level(action_id)
        confirmation_chain: list[dict[str, Any]] = [
            self._confirmation_chain_step("user_explicit", bool(user_confirmed)),
        ]
        allowed_contexts = action_policy_cfg.get("allowed_contexts", [])
        allowed_contexts = (
            [str(item).strip() for item in allowed_contexts if str(item).strip()]
            if isinstance(allowed_contexts, list) else []
        )

        if self.state.runtime_mode != "active":
            reason = f"runtime_mode={self.state.runtime_mode} blocks game actions"
            confirmation_chain.append(self._confirmation_chain_step("runtime_mode", "blocked"))
            blocked = ActionExecutionResult(
                ok=False,
                action_id=action_id,
                executed_at=now_iso(),
                blocked_reason=reason,
                guard_aborted=False,
                window_title=self.state.window_title,
                risk_level=risk_level,
            )
            self._record_action_result(
                blocked,
                allow_reason=reason,
                risk_level=risk_level,
                confirmation_chain=confirmation_chain,
            )
            self._emit_status()
            return blocked.to_dict()

        allowed, reason = self._action_registry.validate(
            action_id,
            current_scene=current_scene,
            action_mode=self.state.action_mode,
            session_running=self.state.running,
            user_confirmed=user_confirmed,
        )
        confirmation_chain.append(
            self._confirmation_chain_step("registry_validate", "passed" if allowed else f"rejected: {reason}"),
        )
        if not allowed:
            blocked = ActionExecutionResult(
                ok=False,
                action_id=action_id,
                executed_at=now_iso(),
                blocked_reason=reason,
                guard_aborted=False,
                window_title=self.state.window_title,
                risk_level=risk_level,
            )
            self._record_action_result(
                blocked,
                allow_reason=reason,
                risk_level=risk_level,
                confirmation_chain=confirmation_chain,
            )
            self._emit_status()
            return blocked.to_dict()

        if allowed_contexts and current_scene not in allowed_contexts:
            reason = f"scene '{current_scene}' not in action_policy.allowed_contexts {allowed_contexts}"
            confirmation_chain.append(self._confirmation_chain_step("action_policy_context", "rejected"))
            blocked = ActionExecutionResult(
                ok=False,
                action_id=action_id,
                executed_at=now_iso(),
                blocked_reason=reason,
                guard_aborted=False,
                window_title=self.state.window_title,
                risk_level=risk_level,
            )
            self._record_action_result(
                blocked,
                allow_reason=reason,
                risk_level=risk_level,
                confirmation_chain=confirmation_chain,
            )
            self._emit_status()
            return blocked.to_dict()

        if dry_run:
            confirmation_chain.append(self._confirmation_chain_step("dry_run", True))
            result = ActionExecutionResult(
                ok=True, action_id=action_id,
                executed_at=now_iso(),
                blocked_reason="dry_run",
                window_title=self.state.window_title,
                risk_level=risk_level,
            )
            self._record_action_result(
                result,
                allow_reason="dry_run",
                risk_level=risk_level,
                confirmation_chain=confirmation_chain,
            )
            self._emit_status()
            return result.to_dict()

        if self.state.window_bound:
            binding_result = WindowBindingResult(
                bound=True,
                window_title=self.state.window_title,
                match_keyword=self.state.window_match_keyword,
                left=self.state.window_left,
                top=self.state.window_top,
                width=self.state.window_width,
                height=self.state.window_height,
            )
        else:
            binding_result = self._bind_window()
        if not binding_result.bound:
            reason = binding_result.error or "window not bound"
            confirmation_chain.append(self._confirmation_chain_step("bind_window", "failed"))
            blocked = ActionExecutionResult(
                ok=False,
                action_id=action_id,
                executed_at=now_iso(),
                blocked_reason=reason,
                guard_aborted=False,
                window_title=self.state.window_title,
                risk_level=risk_level,
            )
            self._record_action_result(
                blocked,
                allow_reason=reason,
                risk_level=risk_level,
                confirmation_chain=confirmation_chain,
            )
            self._emit_status()
            return blocked.to_dict()

        if not binding_result.has_bounds():
            reason = "window bounds unavailable"
            confirmation_chain.append(self._confirmation_chain_step("bind_window", "bounds_unavailable"))
            blocked = ActionExecutionResult(
                ok=False,
                action_id=action_id,
                executed_at=now_iso(),
                blocked_reason=reason,
                guard_aborted=False,
                window_title=self.state.window_title,
                risk_level=risk_level,
            )
            self._record_action_result(
                blocked,
                allow_reason=reason,
                risk_level=risk_level,
                confirmation_chain=confirmation_chain,
            )
            self._emit_status()
            return blocked.to_dict()

        located_action = self._resolve_located_action_locked(
            action_id,
            binding=binding_result,
        )
        command = located_action.to_input_command() if located_action is not None else None
        if command is None:
            reason = f"no screen mapping for action_id: {action_id}"
            confirmation_chain.append(self._confirmation_chain_step("locator", "not_found"))
            blocked = ActionExecutionResult(
                ok=False,
                action_id=action_id,
                executed_at=now_iso(),
                blocked_reason=reason,
                guard_aborted=False,
                window_title=self.state.window_title,
                risk_level=risk_level,
            )
            self._record_action_result(
                blocked,
                allow_reason=reason,
                risk_level=risk_level,
                confirmation_chain=confirmation_chain,
            )
            self._emit_status()
            return blocked.to_dict()

        guard_cfg = self._get_human_override_guard_cfg()
        guard_enabled = bool(guard_cfg.get("enabled", True)) and bool(guard_cfg.get("abort_on_human_input", True))
        active_window_sec = float(guard_cfg.get("active_window_sec", 1.5))
        movement_threshold_px = int(guard_cfg.get("movement_threshold_px", 18))
        check_window_focus = bool(guard_cfg.get("check_window_focus", True))

        guard_window = self._human_override_guard.arm(
            enabled=guard_enabled,
            active_window_sec=active_window_sec,
            movement_threshold_px=movement_threshold_px,
            expected_window_title=binding_result.window_title or self.state.window_title,
            check_window_focus=check_window_focus,
        )
        confirmation_chain.append(
            self._confirmation_chain_step(
                "guard_arm",
                "armed" if guard_window.armed else "inactive",
            ),
        )

        def guard_check() -> tuple[bool, str]:
            decision = self._human_override_guard.evaluate()
            if decision.should_abort:
                return True, decision.reason
            return False, ""

        result_payload = self._input_adapter.execute(command, guard_check=guard_check)
        self._human_override_guard.reset()

        guard_aborted = bool(result_payload.get("aborted", False))
        ok = bool(result_payload.get("ok", False))
        result = ActionExecutionResult(
            ok=ok,
            action_id=action_id,
            executed_at=now_iso(),
            blocked_reason="" if ok else result_payload.get("abort_reason", "execution_failed"),
            guard_aborted=guard_aborted,
            window_title=self.state.window_title,
            locator_source=located_action.source,
            button_region=located_action.button_region,
            target_x=command.target_x,
            target_y=command.target_y,
            risk_level=risk_level,
        )

        confirmation_chain.append(self._confirmation_chain_step("execute", "ok" if ok else result.blocked_reason))
        self._record_action_result(
            result,
            allow_reason=reason,
            locator_source=located_action.source,
            button_region=located_action.button_region,
            target_x=command.target_x,
            target_y=command.target_y,
            risk_level=risk_level,
            confirmation_chain=confirmation_chain,
        )
        self._emit_status()
        return result.to_dict()

    def _record_action_result(
        self,
        result: ActionExecutionResult,
        *,
        allow_reason: str = "",
        locator_source: str = "",
        button_region: dict[str, Any] | None = None,
        target_x: int | None = None,
        target_y: int | None = None,
        risk_level: str = "",
        confirmation_chain: list[dict[str, Any]] | None = None,
    ) -> None:
        log_entry = ActionLogEntry(
            action_id=result.action_id,
            executed_at=result.executed_at or now_iso(),
            ok=result.ok,
            blocked_reason=result.blocked_reason,
            guard_aborted=result.guard_aborted,
            window_title=result.window_title,
            trigger_source="manual",
            allow_reason=allow_reason,
            locator_source=locator_source or result.locator_source,
            button_region=button_region if button_region is not None else result.button_region,
            target_x=target_x if target_x is not None else result.target_x,
            target_y=target_y if target_y is not None else result.target_y,
            risk_level=risk_level or result.risk_level,
            confirmation_chain=confirmation_chain or [],
        )
        log_path = append_action_log(self.plugin.data_path("session_cache"), log_entry)
        result.log_path = str(log_path)
        self._apply_action_result(result)

    def _apply_action_result(self, result: ActionExecutionResult) -> None:
        self.state.last_action_id = result.action_id
        self.state.last_action_at = result.executed_at
        self.state.last_action_ok = result.ok
        self.state.last_action_blocked_reason = result.blocked_reason
        self.state.last_action_guard_aborted = result.guard_aborted
        self.state.last_action_locator_source = result.locator_source
        self.state.last_action_risk_level = result.risk_level
        if not result.ok:
            self.state.last_error = result.blocked_reason
        else:
            self.state.last_error = ""
