from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .llm_coach import merge_heuristic_and_llm
from .models import CoachDecision, MahjongCoachConfig, RoundCoachState


LLM_DECISION_TYPES = {"opening_plan", "coach_checkpoint"}


class DecisionCoordinator:
    def should_enhance_with_llm(self, decision: CoachDecision, config: MahjongCoachConfig) -> bool:
        if not config.llm_enabled or decision.action_required or not decision.hand_tiles:
            return False
        if decision.decision_type == "opening_plan":
            return config.llm_opening_enabled
        if decision.decision_type == "coach_checkpoint":
            return config.llm_checkpoint_enabled
        return False

    def build_enhancement_token(
        self,
        decision: CoachDecision,
        state: RoundCoachState,
        hand_signature: str,
    ) -> str:
        return json.dumps(
            {
                "round_id": state.round_id,
                "hand_signature": hand_signature,
                "decision_type": decision.decision_type,
                "update_count": state.update_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def is_stale(
        self,
        token: str,
        *,
        current_hand_signature: str,
        round_id: str,
        update_count: int,
    ) -> bool:
        try:
            payload = json.loads(token)
        except Exception:
            return True
        return (
            str(payload.get("round_id") or "") != round_id
        )

    def token_hand_signature(self, token: str) -> str:
        try:
            payload = json.loads(token)
        except Exception:
            return ""
        return str(payload.get("hand_signature") or "")

    def token_update_count(self, token: str) -> int | None:
        try:
            payload = json.loads(token)
        except Exception:
            return None
        try:
            return int(payload.get("update_count"))
        except Exception:
            return None

    def apply_llm_plan(self, decision: CoachDecision, heuristic_plan: dict[str, Any], llm_plan: dict[str, Any]) -> CoachDecision:
        plan = merge_heuristic_and_llm(heuristic_plan, llm_plan)
        summary = str(plan.get("summary") or decision.suggestion or decision.summary)
        detail = str(plan.get("detail") or decision.detail)
        engine_meta = dict(decision.engine_meta)
        engine_meta["analysis_source"] = "llm"
        engine_meta["llm_enhanced"] = True
        return replace(
            decision,
            summary=decision.summary,
            detail=detail,
            suggestion=summary,
            llm_enhanced=True,
            analysis_source="llm",
            engine_meta=engine_meta,
        )
