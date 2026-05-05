from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..contracts import PerceivedGameState
from .tile_efficiency import build_incremental_draw_candidates, build_mahjong_analysis
from .utils import meld_group_count, recognized_hand_tile_count


@dataclass
class PreturnDiscardPlan:
    hand_tiles: list[str] = field(default_factory=list)
    hand_signature: str = ""
    meld_count: int = 0
    draw_slot_index: int = 14
    candidate_discards: list[dict[str, Any]] = field(default_factory=list)
    shanten_estimate: int | None = None
    ukeire_estimate: int | None = None
    analysis_confidence: float = 0.0
    attack_defense_bias: str = "neutral"
    defense_alerts: list[str] = field(default_factory=list)
    source_analysis_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_preturn_discard_plan(state: PerceivedGameState) -> PreturnDiscardPlan | None:
    """构建预切牌计划，用于在玩家摸牌前预先分析可能的切牌选择。
    
    该函数在非用户回合且手牌看起来处于听牌状态时，预先计算摸牌后的切牌策略，
    以便在玩家实际摸牌后能够快速做出决策。
    
    Args:
        state: PerceivedGameState - 当前游戏状态对象，包含手牌、副露、场景等信息
        
    Returns:
        PreturnDiscardPlan | None - 预切牌计划对象，如果不符合预计算条件则返回None
        
    条件检查：只有在比赛场景、非用户回合且没有按钮显示时才进行预计算
    """
    if state.scene != "in_match" or state.is_user_turn or state.buttons:
        return None
    if not _looks_like_waiting_hand(state):
        return None

    # 计算副露组数量和摸牌位置索引
    meld_count = meld_group_count(state)
    draw_slot_index = max(1, min(14, len(state.hand_tiles) + 1))
    
    # 执行麻将分析，专注于预切牌规划
    analysis = build_mahjong_analysis(
        state,
        recommended_focus="preturn_discard_planning",
        review_tags=["preturn_planning"],
    )
    if not analysis.tile_level_available or not analysis.candidate_discards:
        return None

    # 构建并返回预切牌计划对象
    return PreturnDiscardPlan(
        hand_tiles=list(state.hand_tiles),
        hand_signature=_hand_signature(state.hand_tiles),
        meld_count=meld_count,
        draw_slot_index=draw_slot_index,
        candidate_discards=list(analysis.candidate_discards),
        shanten_estimate=analysis.shanten_estimate,
        ukeire_estimate=analysis.ukeire_estimate,
        analysis_confidence=analysis.analysis_confidence,
        attack_defense_bias=analysis.attack_defense_bias,
        defense_alerts=list(analysis.defense_alerts),
        source_analysis_version=analysis.analysis_version,
    )


def apply_preturn_discard_plan(
    state: PerceivedGameState,
    plan: PreturnDiscardPlan | None,
) -> tuple[PerceivedGameState, dict[str, Any]]:
    if plan is None:
        return state, {"applied": False, "reason": "no_plan"}
    if state.scene != "in_match" or not state.is_user_turn or state.buttons:
        return state, {"applied": False, "reason": "not_user_discard_turn"}
    if not state.hand_tiles:
        return state, {"applied": False, "reason": "missing_hand_tiles"}

    drawn_tile = _find_drawn_tile(plan.hand_tiles, state.hand_tiles)
    if not drawn_tile:
        return state, {
            "applied": False,
            "reason": "hand_delta_not_single_draw",
            "plan_signature": plan.hand_signature,
            "current_signature": _hand_signature(state.hand_tiles),
        }

    hints = dict(state.analysis_hints) if isinstance(state.analysis_hints, dict) else {}
    candidates = build_incremental_draw_candidates(
        plan.hand_tiles,
        drawn_tile,
        plan.candidate_discards,
        dora_indicators=state.dora_indicators,
        hints=hints,
        riichi_players=state.riichi_players,
    )
    if not candidates:
        return state, {"applied": False, "reason": "no_incremental_candidates"}

    updated_hints = dict(hints)
    updated_hints["candidate_discards"] = candidates
    updated_hints["preturn_plan_applied"] = True
    updated_hints["preturn_plan_source"] = "single_draw_delta"
    updated_hints["preturn_plan_waiting_signature"] = plan.hand_signature
    updated_hints["preturn_plan_drawn_tile"] = drawn_tile
    updated_hints["preturn_plan_candidate_count"] = len(candidates)
    updated_hints["recognized_meld_group_count"] = max(meld_group_count(state), plan.meld_count)
    updated_hints["recognized_hand_tile_count"] = max(
        recognized_hand_tile_count(state),
        len(state.hand_tiles),
    )
    if plan.shanten_estimate is not None and updated_hints.get("shanten_estimate") is None:
        updated_hints["shanten_estimate"] = plan.shanten_estimate
    if plan.ukeire_estimate is not None and updated_hints.get("ukeire_estimate") is None:
        updated_hints["ukeire_estimate"] = plan.ukeire_estimate
    if plan.attack_defense_bias and not updated_hints.get("attack_defense_bias"):
        updated_hints["attack_defense_bias"] = plan.attack_defense_bias
    if plan.analysis_confidence and not updated_hints.get("analysis_confidence"):
        updated_hints["analysis_confidence"] = plan.analysis_confidence
    if plan.source_analysis_version and not updated_hints.get("analysis_version"):
        updated_hints["analysis_version"] = f"{plan.source_analysis_version}+preturn"

    # Use dataclasses.replace instead of PerceivedGameState(**state.to_dict())
    # so that adding a derived field to to_dict() (e.g. a timestamp) does not
    # surface as a TypeError inside the locked fast path
    # (CODE_REVIEW_v1.2 N-M5).
    return replace(state, analysis_hints=updated_hints), {
        "applied": True,
        "drawn_tile": drawn_tile,
        "candidate_count": len(candidates),
        "top_tile": candidates[0].get("tile", ""),
        "plan_signature": plan.hand_signature,
        "current_signature": _hand_signature(state.hand_tiles),
    }


def apply_preturn_draw_tile(
    state: PerceivedGameState,
    plan: PreturnDiscardPlan | None,
    drawn_tile: str,
) -> tuple[PerceivedGameState, dict[str, Any]]:
    if plan is None:
        return state, {"applied": False, "reason": "no_plan"}
    drawn_tile = str(drawn_tile).strip()
    if not drawn_tile:
        return state, {"applied": False, "reason": "missing_drawn_tile"}
    prepared = replace(
        state,
        hand_tiles=[*plan.hand_tiles, drawn_tile],
        is_user_turn=True,
        buttons=[],
        analysis_hints=_with_drawn_tile_slot_hint(state.analysis_hints, plan, drawn_tile),
    )
    return apply_preturn_discard_plan(prepared, plan)


def _with_drawn_tile_slot_hint(
    raw_hints: Any,
    plan: PreturnDiscardPlan,
    drawn_tile: str,
) -> dict[str, Any]:
    hints = dict(raw_hints) if isinstance(raw_hints, dict) else {}
    raw_slots = hints.get("hand_tile_slots")
    slots = list(raw_slots) if isinstance(raw_slots, list) else []
    if not slots:
        return hints

    draw_slot = None
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        if str(slot.get("slot_id", "")) == f"hand_{plan.draw_slot_index}":
            draw_slot = dict(slot)
            break
    if draw_slot is None:
        return hints

    draw_slot["tile"] = drawn_tile
    draw_slot["index"] = len(plan.hand_tiles)
    draw_slot["source"] = "drawn_tile_fast_path"
    slots.append(draw_slot)
    hints["hand_tile_slots"] = slots
    return hints


def _looks_like_waiting_hand(state: PerceivedGameState) -> bool:
    count = recognized_hand_tile_count(state)
    meld_count = meld_group_count(state)
    expected_waiting_count = max(1, 13 - meld_count * 3)
    return count == expected_waiting_count and len(state.hand_tiles) == expected_waiting_count


def _find_drawn_tile(previous_hand: list[str], current_hand: list[str]) -> str:
    """Return the single newly-drawn tile if `current_hand` is exactly
    `previous_hand` plus one extra tile.

    Strong invariant (CODE_REVIEW_v1.2 N-H4): for every tile that existed in
    the previous hand, the current hand must have AT LEAST the same count.
    This rules out recognition drift (e.g. a previous 5p getting re-classified
    as 5s while a real new tile is drawn) — the disappearing tile would make
    `current[tile] < previous[tile]` and we bail. Drawing a duplicate of an
    existing tile is now allowed (`current[tile] == previous[tile] + 1`),
    where the previous `!=` check incorrectly returned "" for that case.
    """
    previous_counts = Counter(previous_hand)
    current_counts = Counter(current_hand)
    deltas: list[str] = []
    for tile, count in current_counts.items():
        extra = count - previous_counts.get(tile, 0)
        if extra > 0:
            deltas.extend([tile] * extra)
    if len(deltas) != 1 or sum(current_counts.values()) != sum(previous_counts.values()) + 1:
        return ""
    for tile, count in previous_counts.items():
        if current_counts.get(tile, 0) < count:
            return ""
    return deltas[0]


def _hand_signature(hand_tiles: list[str]) -> str:
    counts = Counter(str(tile).strip() for tile in hand_tiles if str(tile).strip())
    return "|".join(f"{tile}:{counts[tile]}" for tile in sorted(counts))
