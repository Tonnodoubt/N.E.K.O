from __future__ import annotations

from typing import Any

from ..contracts import DecisionResult, PerceivedGameState
from .utils import coerce_int, meld_group_count, recognized_hand_tile_count
from ..tile_labels import (
    dedupe as _dedupe,
    format_tile_label,
    normalize_tile as _normalize_tile,
    replace_tile_codes_in_text,
)
from .tile_efficiency import build_mahjong_analysis

WIN_BUTTONS = {"ron", "tsumo"}
DECLARATION_BUTTONS = {"riichi"}
CALL_BUTTONS = {"chi", "pon", "kan"}
PASSIVE_BUTTONS = {"skip", "confirm", "cancel"}
ACTIONABLE_BUTTONS = WIN_BUTTONS | DECLARATION_BUTTONS | CALL_BUTTONS | PASSIVE_BUTTONS


def build_decision(state: PerceivedGameState) -> DecisionResult:
    buttons = list(state.buttons)
    source_scene = state.scene
    effective_scene = _resolve_effective_scene(state, buttons)
    reason_codes: list[str] = []
    review_tags: list[str] = []
    decision_type = "scene_update"
    priority = 20
    risk_level = "low"
    action_required = False
    speakable = False
    summary = "当前局面没有新的关键提醒。"
    detail = "可以继续观察画面变化，暂时不用主动提醒。"
    suggestion = "先继续观察牌桌变化。"
    recommended_focus = "observe"
    recommended_button_types: list[str] = []

    win_buttons = [button for button in buttons if button in WIN_BUTTONS]
    declaration_buttons = [button for button in buttons if button in DECLARATION_BUTTONS]
    call_buttons = [button for button in buttons if button in CALL_BUTTONS]
    passive_buttons = [button for button in buttons if button in PASSIVE_BUTTONS]

    if effective_scene != source_scene:
        reason_codes.append(f"scene.promoted_from_{source_scene}")
        reason_codes.append(f"scene.promoted_to_{effective_scene}")

    if effective_scene == "unknown" and not buttons:
        decision_type = "uncertain_state"
        summary = "当前局面还不够清晰。"
        detail = "这一帧暂时没看清楚，可以继续抓下一张图确认。"
        suggestion = "先等下一张清晰帧，再决定是否提醒。"
        recommended_focus = "need_clearer_frame"
        reason_codes.append("scene.unknown")
    elif effective_scene == "replay":
        decision_type = "waiting_state"
        summary = "当前更像是回放或等待状态。"
        detail = "现在不像是需要立刻操作的局面，更适合静默陪看。"
        suggestion = "更适合把这一刻当作复盘节点，而不是即时提醒。"
        recommended_focus = "replay_observe"
        reason_codes.append("scene.replay")
        review_tags.append("replay_segment")
    elif effective_scene == "dialog" and passive_buttons:
        decision_type = "action_available"
        priority = 58
        risk_level = "medium"
        action_required = True
        speakable = False
        summary = "当前弹出了确认类操作。"
        detail = "这更像是托管、继续或确认一类弹窗，建议先看清内容再点。"
        suggestion = "先确认弹窗语义，再决定是确认、取消还是等待。"
        recommended_focus = "dialog_confirmation"
        reason_codes.append("scene.dialog")
        reason_codes.extend(f"button.{button}_visible" for button in passive_buttons)
        review_tags.append("dialog_confirm")
        recommended_button_types = _preferred_passive_buttons(passive_buttons)
    elif effective_scene == "in_match" and win_buttons:
        decision_type = "danger_action"
        priority = 96
        risk_level = "high"
        action_required = True
        speakable = True
        summary = "当前像是出现了和牌窗口。"
        detail = "检测到 ron 或 tsumo 一类高价值按钮，这通常值得立刻确认。"
        suggestion = "先确认和牌条件与按钮语义，优先别错过这一手。"
        recommended_focus = "win_confirmation"
        reason_codes.extend(f"button.{button}_visible" for button in win_buttons)
        review_tags.extend(["win_window", "high_value_timing"])
        recommended_button_types = list(win_buttons)
    elif effective_scene == "in_match" and declaration_buttons:
        decision_type = "danger_action"
        priority = 88
        risk_level = "high"
        action_required = True
        speakable = True
        summary = "当前出现了立直决策点。"
        detail = "检测到 riichi 按钮，这是需要明确路线和风险判断的时刻。"
        suggestion = "先看牌河和当前手型，再决定要不要立直。"
        recommended_focus = "riichi_decision"
        reason_codes.extend(f"button.{button}_visible" for button in declaration_buttons)
        review_tags.extend(["riichi_window", "decision_point"])
        recommended_button_types = list(declaration_buttons)
    elif effective_scene == "in_match" and "kan" in call_buttons:
        decision_type = "danger_action"
        priority = 82
        risk_level = "high"
        action_required = True
        speakable = False
        summary = "当前出现了杠牌决策点。"
        detail = "杠会改变场上信息和后续进张，建议先确认这一手是不是值得开杠。"
        suggestion = "先确认开杠会不会打乱当前路线，再决定是否点下去。"
        recommended_focus = "kan_decision"
        reason_codes.append("button.kan_visible")
        review_tags.extend(["kan_choice", "decision_point"])
    elif effective_scene == "in_match" and call_buttons:
        decision_type = "action_available"
        priority = 72
        risk_level = "medium"
        action_required = True
        speakable = bool(state.is_user_turn)
        summary = "当前出现了吃碰一类操作机会。"
        detail = "吃碰会直接改变手牌结构，这一类按钮更适合结合当前路线来判断。"
        suggestion = "先确认你现在是进攻路线还是防守路线，再决定要不要吃碰。"
        recommended_focus = "call_decision"
        reason_codes.extend(f"button.{button}_visible" for button in call_buttons)
        review_tags.extend(["call_window", "route_choice"])
    elif effective_scene == "in_match" and passive_buttons:
        decision_type = "action_available"
        priority = 56
        risk_level = "medium"
        action_required = True
        speakable = bool(state.is_user_turn and ("confirm" in passive_buttons or "cancel" in passive_buttons))
        summary = "当前存在确认或略过类操作。"
        detail = "画面更像是在等待用户确认、取消或跳过，这类按钮值得看一眼但通常不必高声打断。"
        suggestion = "先看清这是不是过牌、确认还是取消，再决定是否操作。"
        recommended_focus = "confirm_or_skip"
        reason_codes.extend(f"button.{button}_visible" for button in passive_buttons)
        review_tags.append("ui_confirmation")
        recommended_button_types = _preferred_passive_buttons(passive_buttons)
    elif effective_scene == "in_match" and state.is_user_turn:
        decision_type = "scene_update"
        priority = 44
        risk_level = "medium"
        action_required = True
        speakable = False
        summary = "当前像是轮到用户关注的阶段。"
        detail = "虽然没有明显按钮，但画面像是进入了需要观察摸牌和牌河的时刻。"
        suggestion = "先看最新摸牌、牌河和副露，再决定下一步。"
        recommended_focus = "turn_observe"
        reason_codes.append("turn.user_likely")
        review_tags.append("turn_checkpoint")
    elif effective_scene in {"menu", "lobby", "matching", "result", "dialog"}:
        decision_type = "waiting_state"
        priority = 25
        summary = "当前更像是非对局操作阶段。"
        detail = "这是菜单、大厅、对话框或结算一类状态，通常不需要高频播报。"
        suggestion = "更适合静默展示状态，不需要持续提醒。"
        recommended_focus = f"scene_{effective_scene}"
        reason_codes.append(f"scene.{effective_scene}")
        if effective_scene == "result":
            review_tags.append("round_result")
    else:
        reason_codes.append(f"scene.{effective_scene}")

    if state.is_user_turn and "turn.user_likely" not in reason_codes:
        reason_codes.append("turn.user_likely")

    if state.scene == "in_match" and state.is_user_turn and not buttons:
        expected_hand_count = _expected_discard_turn_hand_count(state)
        if expected_hand_count < 14:
            reason_codes.append("turn.post_meld_hand_shape")

    if state.confidence < 0.45:
        reason_codes.append("perception.low_confidence")
        detail = f"{detail} 当前识别置信度还不高，最好再看一眼确认。"
        review_tags.append("low_confidence")
        if not win_buttons:
            speakable = False

    if not reason_codes:
        reason_codes.append("state.default")

    review_tags = _dedupe(review_tags)
    mahjong_analysis = build_mahjong_analysis(
        state,
        recommended_focus=recommended_focus,
        review_tags=review_tags,
    )
    hand_tile_count = recognized_hand_tile_count(state)

    if effective_scene == "in_match" and declaration_buttons:
        summary, detail, suggestion, risk_level, recommended_button = _build_riichi_to_action_copy(
            state=state,
            mahjong_analysis=mahjong_analysis,
            buttons=buttons,
        )
        recommended_button_types = _button_priority_list(recommended_button, buttons)
        speakable = bool(state.confidence >= 0.45)
    elif effective_scene == "in_match" and "kan" in call_buttons:
        summary, detail, suggestion, risk_level, recommended_button = _build_call_to_action_copy(
            action="kan",
            state=state,
            mahjong_analysis=mahjong_analysis,
        )
        recommended_button_types = _button_priority_list(recommended_button, buttons)
        speakable = bool(state.confidence >= 0.45)
    elif effective_scene == "in_match" and call_buttons:
        action = "pon" if "pon" in call_buttons else "chi"
        summary, detail, suggestion, risk_level, recommended_button = _build_call_to_action_copy(
            action=action,
            state=state,
            mahjong_analysis=mahjong_analysis,
        )
        recommended_button_types = _button_priority_list(recommended_button, buttons)
        speakable = bool(state.confidence >= 0.45)

    if (
        mahjong_analysis.tile_level_available
        and effective_scene == "in_match"
        and state.is_user_turn
        and decision_type == "scene_update"
        and not buttons
        and _looks_like_discard_turn(state, recognized_hand_tile_count=hand_tile_count)
    ):
        top_candidate = mahjong_analysis.candidate_discards[0] if mahjong_analysis.candidate_discards else {}
        discard_tile = str(top_candidate.get("tile", "")).strip()
        discard_label = format_tile_label(discard_tile)
        discard_reason = replace_tile_codes_in_text(top_candidate.get("reason", ""))
        bias = mahjong_analysis.attack_defense_bias
        defense_alert = (
            replace_tile_codes_in_text(mahjong_analysis.defense_alerts[0])
            if mahjong_analysis.defense_alerts else ""
        )
        ukeire_text = (
            f"，进张估计约 {mahjong_analysis.ukeire_estimate}"
            if mahjong_analysis.ukeire_estimate is not None else ""
        )

        decision_type = "tile_efficiency_hint"
        priority = max(priority, 62)
        risk_level = "medium" if bias == "slightly_defensive" else "low"
        action_required = True
        speakable = bool(state.confidence >= 0.72 and mahjong_analysis.shanten_estimate in {0, 1})
        summary = (
            "这一巡更适合先走稳一点的牌效率路线。"
            if bias == "slightly_defensive"
            else "这一巡可以开始给出轻量牌理建议了。"
        )
        detail = "当前已经有结构化牌理输入。"
        if discard_label:
            detail = f"当前更像适合先考虑处理 {discard_label} 这类改善较弱的牌{ukeire_text}。"
        if discard_reason:
            detail = f"{detail} {discard_reason}。"
        if defense_alert:
            detail = f"{detail} {defense_alert}"
        suggestion = (
            f"优先考虑处理 {discard_label} 这类改善较弱的牌。"
            if discard_label else "优先保留更连贯的块，再处理孤张或边张。"
        )
        recommended_focus = "tile_efficiency"
        reason_codes.extend(["analysis.tile_level_available", "analysis.tile_efficiency_hint"])
        review_tags.extend(["tile_efficiency", "mid_round_choice"])

    review_tags = _dedupe(review_tags)
    review_summary_snippet = _build_review_summary_snippet(
        decision_type=decision_type,
        recommended_focus=recommended_focus,
        review_tags=review_tags,
    )

    return DecisionResult(
        decision_type=decision_type,
        priority=priority,
        risk_level=risk_level,
        action_required=action_required,
        speakable=speakable,
        summary=summary,
        detail=detail,
        suggestion=suggestion,
        recommended_focus=recommended_focus,
        scene=effective_scene,
        buttons=buttons,
        reason_codes=reason_codes,
        review_tags=review_tags,
        review_summary_snippet=review_summary_snippet,
        mahjong_analysis=mahjong_analysis.to_dict(),
        engine_meta={
            "engine": "rule_based_v2",
            "confidence": state.confidence,
            "source_scene": source_scene,
            "scene_promoted": effective_scene != source_scene,
            "focus_area": recommended_focus,
            "analysis_version": mahjong_analysis.analysis_version,
            "analysis_confidence": mahjong_analysis.analysis_confidence,
            "tile_level_state": mahjong_analysis.tile_level_state,
            "defense_alert_count": len(mahjong_analysis.defense_alerts),
            "button_groups": {
                "win": win_buttons,
                "declaration": declaration_buttons,
                "call": call_buttons,
                "passive": passive_buttons,
            },
            "recommended_button_types": recommended_button_types,
            "tile_level_available": mahjong_analysis.tile_level_available,
            "review_candidate": bool(review_tags and priority >= 44),
        },
    )


def decide_perception(state: PerceivedGameState) -> tuple[DecisionResult, dict[str, Any]]:
    decision = build_decision(state)
    debug_payload = {
        "source_scene": state.scene,
        "source_confidence": state.confidence,
        "source_buttons": list(state.buttons),
        "source_is_user_turn": state.is_user_turn,
        "source_notes": list(state.notes),
        "decision_reason_codes": list(decision.reason_codes),
        "decision_focus_area": decision.recommended_focus,
        "decision_review_tags": list(decision.review_tags),
    }
    return decision, debug_payload


def _resolve_effective_scene(state: PerceivedGameState, buttons: list[str]) -> str:
    scene = str(state.scene or "unknown")
    if scene in {"in_match", "dialog", "replay"}:
        return scene

    if scene == "unknown" and _has_match_table_evidence(state):
        return "in_match"

    template_buttons = {
        str(region.get("button_type", "")).strip()
        for region in state.button_regions
        if isinstance(region, dict)
    }
    if template_buttons & ACTIONABLE_BUTTONS:
        return "in_match"

    if state.is_user_turn and any(button in buttons for button in (WIN_BUTTONS | DECLARATION_BUTTONS | CALL_BUTTONS)):
        return "in_match"

    return scene


def _has_match_table_evidence(state: PerceivedGameState) -> bool:
    if state.hand_tiles or state.discard_piles or state.visible_tiles:
        return True
    hints = state.analysis_hints if isinstance(state.analysis_hints, dict) else {}
    try:
        recognized_count = int(hints.get("recognized_hand_tile_count", 0) or 0)
    except (TypeError, ValueError):
        recognized_count = 0
    return recognized_count > 0 or bool(hints.get("tile_level_available"))


def _looks_like_discard_turn(
    state: PerceivedGameState,
    *,
    recognized_hand_tile_count: int,
) -> bool:
    expected_hand_count = _expected_discard_turn_hand_count(state)
    if recognized_hand_tile_count >= expected_hand_count:
        return True

    hints = state.analysis_hints if isinstance(state.analysis_hints, dict) else {}
    if "recognized_hand_tile_count" in hints:
        return False

    if state.hand_tiles and recognized_hand_tile_count >= max(1, expected_hand_count - 1):
        return True
    return isinstance(hints.get("candidate_discards"), list) and bool(hints.get("candidate_discards"))


def _expected_discard_turn_hand_count(state: PerceivedGameState) -> int:
    mc = meld_group_count(state)
    return max(1, 14 - mc * 3)


def _preferred_passive_buttons(buttons: list[str]) -> list[str]:
    return [button for button in ("skip", "confirm", "cancel") if button in buttons]


def _button_priority_list(primary: str, buttons: list[str]) -> list[str]:
    return [primary] if primary and primary in buttons else []


def _decline_button(buttons: list[str]) -> str:
    for button in ("skip", "cancel", "confirm"):
        if button in buttons:
            return button
    return ""


def _build_riichi_to_action_copy(
    *,
    state: PerceivedGameState,
    mahjong_analysis: Any,
    buttons: list[str],
) -> tuple[str, str, str, str, str]:
    defense_alerts = list(getattr(mahjong_analysis, "defense_alerts", []) or [])
    candidates = list(getattr(mahjong_analysis, "candidate_discards", []) or [])
    top_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    ukeire = coerce_int(top_candidate.get("ukeire_estimate"))
    wait_quality_bonus = coerce_int(top_candidate.get("wait_quality_bonus")) or 0
    has_pressure = bool(state.riichi_players or defense_alerts)
    decline_button = _decline_button(buttons)
    wait_text = f"当前有效牌约 {ukeire} 张，" if ukeire is not None else ""
    defense_text = replace_tile_codes_in_text(defense_alerts[0]) if defense_alerts else ""

    if has_pressure and ukeire is not None and ukeire < 6:
        summary = "当前出现了立直选择。"
        detail = f"{wait_text}但场上已经有压力，这种追立直更容易把手锁死。"
        if defense_text:
            detail = f"{detail} {defense_text}"
        suggestion = "建议先跳过立直，保留换牌和防守余地。"
        return summary, detail, suggestion, "high", decline_button

    if ukeire is not None and ukeire <= 2 and wait_quality_bonus < 0 and decline_button:
        summary = "当前出现了立直选择。"
        detail = f"{wait_text}听牌形状偏窄，立直后不方便改良。"
        suggestion = "可以先跳过立直，等更好的听牌或安全窗口。"
        return summary, detail, suggestion, "medium", decline_button

    summary = "当前出现了立直选择。"
    detail = f"{wait_text}没有明显需要立刻收手的压力，可以用立直提高和牌推进力。"
    if ukeire is not None and ukeire >= 6:
        detail = f"{detail} 这个待牌数量已经值得主动压上去。"
    suggestion = "建议立直，抢先把这手牌推到和牌窗口。"
    return summary, detail, suggestion, "medium", "riichi"


def _open_hand_yaku_signal(state: PerceivedGameState) -> tuple[bool, str]:
    hints = state.analysis_hints if isinstance(state.analysis_hints, dict) else {}
    if bool(hints.get("open_yaku_available")):
        return True, "当前已有开放手役线索，鸣牌后不容易变成无役。"
    yaku_candidates = hints.get("yaku_candidates")
    if isinstance(yaku_candidates, list) and yaku_candidates:
        return True, "当前已有候选役种线索，可以把鸣牌纳入进攻路线。"

    tiles = [_normalize_tile(tile) for tile in state.hand_tiles]
    tiles = [tile for tile in tiles if tile]
    if not tiles:
        return False, ""

    counts = {tile: tiles.count(tile) for tile in set(tiles)}
    if any(tile.endswith("z") and count >= 2 for tile, count in counts.items()):
        return True, "手里有字牌对子/刻子，副露后仍有役牌方向。"
    if _is_tanyao_candidate(tiles):
        return True, "手牌接近断幺路线，鸣牌后仍有开放手役。"
    if _is_honitsu_candidate(tiles):
        return True, "手牌有混一色方向，鸣牌后仍有价值路线。"
    if meld_group_count(state) > 0:
        return True, "当前已经副露过，继续按已有开放路线评估。"
    return False, ""


def _is_tanyao_candidate(tiles: list[str]) -> bool:
    for tile in tiles:
        if tile.endswith("z"):
            return False
        if tile[0] in {"1", "9"}:
            return False
    return True


def _is_honitsu_candidate(tiles: list[str]) -> bool:
    suits = {tile[1] for tile in tiles if len(tile) == 2 and tile[1] in {"m", "p", "s"}}
    honor_count = sum(1 for tile in tiles if tile.endswith("z"))
    suited_count = sum(1 for tile in tiles if len(tile) == 2 and tile[1] in {"m", "p", "s"})
    return len(suits) == 1 and suited_count >= 6 and honor_count >= 2


def _build_call_to_action_copy(
    *,
    action: str,
    state: PerceivedGameState,
    mahjong_analysis: Any,
) -> tuple[str, str, str, str, str]:
    bias = str(getattr(mahjong_analysis, "attack_defense_bias", "") or "").strip()
    shanten_estimate = getattr(mahjong_analysis, "shanten_estimate", None)
    defense_alerts = list(getattr(mahjong_analysis, "defense_alerts", []) or [])
    has_pressure = bool(defense_alerts)
    tile_level_available = bool(getattr(mahjong_analysis, "tile_level_available", False))
    has_open_yaku, open_yaku_reason = _open_hand_yaku_signal(state)
    is_defensive = (
        bias == "slightly_defensive"
        or has_pressure
        or (shanten_estimate is not None and shanten_estimate >= 3)
    )
    is_aggressive = bias == "slightly_attack" or (shanten_estimate is not None and shanten_estimate <= 1)
    is_live_call_candidate = (
        action in {"chi", "pon"}
        and tile_level_available
        and not has_pressure
        and shanten_estimate is not None
        and shanten_estimate <= 2
        and (has_open_yaku or shanten_estimate <= 1)
    )
    action_label = {"chi": "吃", "pon": "碰", "kan": "杠"}.get(action, action)
    defense_text = replace_tile_codes_in_text(defense_alerts[0]) if defense_alerts else ""

    if action == "kan":
        if is_defensive:
            summary = "当前出现了杠牌决策点。"
            detail = "这手更倾向先别杠，开杠会把节奏和风险一起放大。"
            suggestion = "建议先跳过这次杠牌，继续按现有手型走。"
            if defense_text:
                detail = f"{detail} {defense_text}"
            return summary, detail, suggestion, "high", "skip"
        if is_aggressive:
            summary = "当前出现了杠牌决策点。"
            detail = "这手已经比较接近成型，如果杠后仍不破坏牌型，可以认真考虑这次杠。"
            suggestion = "如果这是不影响听牌或明显增值的杠，可以考虑点杠。"
            return summary, detail, suggestion, "medium", "kan"
        summary = "当前出现了杠牌决策点。"
        detail = "这手没有明显需要立刻开杠的信号，先保守一点通常更稳。"
        suggestion = "建议先不杠，继续看下一巡。"
        return summary, detail, suggestion, "medium", "skip"

    if is_defensive:
        summary = f"当前出现了{action_label}牌选择。"
        detail = f"这手更倾向先跳过，不建议现在{action_label}。"
        suggestion = f"建议按跳过处理，这口先别{action_label}。"
        if defense_text:
            detail = f"{detail} {defense_text}"
        return summary, detail, suggestion, "medium", "skip"

    if is_aggressive:
        summary = f"当前出现了{action_label}牌选择。"
        detail = f"这手已经比较接近成型，如果{action_label}后手牌更顺，可以考虑这次{action_label}。"
        suggestion = f"如果{action_label}完能明显缩进向听或直接进攻，可以考虑{action_label}。"
        return summary, detail, suggestion, "medium", action

    if is_live_call_candidate:
        summary = f"当前出现了{action_label}牌选择。"
        detail = f"这手还有进攻空间，当前手牌约 {shanten_estimate} 向听，没有明显立直压力时可以考虑这次{action_label}。"
        if open_yaku_reason:
            detail = f"{detail} {open_yaku_reason}"
        suggestion = f"可以考虑{action_label}，随后选择能让打后向听更低的组合。"
        return summary, detail, suggestion, "medium", action

    summary = f"当前出现了{action_label}牌选择。"
    detail = f"这手没有特别强的副露信号，默认更倾向先跳过这次{action_label}。"
    if (
        action in {"chi", "pon"}
        and tile_level_available
        and not has_open_yaku
        and shanten_estimate is not None
        and shanten_estimate >= 2
    ):
        detail = f"{detail} 副露后暂时有无役风险，过早鸣牌可能把和牌路线变窄。"
    suggestion = f"建议优先跳过；只有在{action_label}后牌型明显变顺时再考虑点。"
    return summary, detail, suggestion, "medium", "skip"


def _build_review_summary_snippet(
    *,
    decision_type: str,
    recommended_focus: str,
    review_tags: list[str],
) -> str:
    if not review_tags:
        return ""
    if "win_window" in review_tags:
        return "这类和牌确认窗口值得作为本局高光节点回看。"
    if "riichi_window" in review_tags:
        return "这次立直窗口更适合在赛后回看当时的路线判断。"
    if {"kan_choice", "call_window", "route_choice"} & set(review_tags):
        return "这类路线选择点适合放进复盘里看节奏是否过急。"
    if "tile_efficiency" in review_tags:
        return "这类中盘牌效率选择适合赛后回看当时为什么会偏进攻或偏保守。"
    if decision_type == "action_available" and recommended_focus == "dialog_confirmation":
        return "确认类弹窗也值得复盘，避免把关键确认误当成普通过渡。"
    return "这个节点已经被标记为适合赛后继续回看。"
