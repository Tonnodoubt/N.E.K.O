from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tile_labels import format_tile_label, replace_tile_codes_in_text

@dataclass
class _DragState:
    value: bool = False
    dx: int = 0
    dy: int = 0


def _state_text(status: dict[str, Any]) -> str:
    bound = "已绑定" if status.get("window_bound") else "未绑定"
    scene = _scene_label(str(status.get("last_scene") or status.get("scene") or "-"))
    turn = "轮到你" if status.get("last_is_user_turn") else "观察中"
    runtime_mode = str(status.get("runtime_mode") or "")
    runtime = "暂停" if runtime_mode == "standby" else str(status.get("runtime_status") or status.get("status") or "-")
    return f"{bound} · {scene} · {turn} · {runtime}"


def _advice_view(status: dict[str, Any]) -> dict[str, str]:
    fallback = _advice_text(status)
    if status.get("runtime_mode") == "standby":
        return {"primary": "已暂停", "reason": fallback}

    if status.get("last_error"):
        return {"primary": "先处理", "reason": fallback}

    decision = status.get("last_decision") if isinstance(status.get("last_decision"), dict) else {}
    decision_type = str(decision.get("decision_type") or status.get("last_decision_type") or "").strip()
    recommended_focus = str(decision.get("recommended_focus") or "").strip()
    if _should_prioritize_decision_text(decision_type, recommended_focus):
        action_label = _action_primary_label(decision, recommended_focus)
        reason = _decision_primary_text(decision, status) or fallback
        return {"primary": action_label, "reason": _reason_line(reason)}

    single = _single_recommendation(decision)
    if single.get("kind") in {"discard", "preturn_discard"}:
        tile = str(single.get("tile") or "").strip()
        candidate = single.get("candidate") if isinstance(single.get("candidate"), dict) else {}
        reason = replace_tile_codes_in_text(str(candidate.get("reason", "")).strip())
        if not reason:
            reason = str(decision.get("detail") or decision.get("suggestion") or "").strip()
        if tile:
            return {
                "primary": format_tile_label(tile) or tile,
                "reason": _reason_line(reason or "保留更连贯的块，只给这一张出牌建议。"),
            }

    analysis = decision.get("mahjong_analysis") if isinstance(decision.get("mahjong_analysis"), dict) else {}
    candidates = analysis.get("candidate_discards") if isinstance(analysis.get("candidate_discards"), list) else []
    top_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    tile = str(top_candidate.get("tile") or top_candidate.get("tile_id") or "").strip()
    if tile:
        reason = replace_tile_codes_in_text(str(top_candidate.get("reason", "")).strip())
        if not reason:
            reason = str(decision.get("detail") or decision.get("suggestion") or "").strip()
        return {
            "primary": format_tile_label(tile) or tile,
            "reason": _reason_line(reason or "保留更连贯的块，先处理改善较弱的牌。"),
        }

    return _fallback_advice_view(status, fallback)


def _fallback_advice_view(status: dict[str, Any], fallback: str) -> dict[str, str]:
    if status.get("window_bound"):
        perception = status.get("last_perception") if isinstance(status.get("last_perception"), dict) else {}
        hints = perception.get("analysis_hints") if isinstance(perception.get("analysis_hints"), dict) else {}
        recognized_count = hints.get("recognized_hand_tile_count")
        if recognized_count:
            return {
                "primary": "看牌中",
                "reason": f"已识别到 {recognized_count} 张手牌，等轮到你时给推荐。",
            }
        return {"primary": "看牌中", "reason": "等识别到你的手牌后会给出推荐。"}
    return {"primary": "未绑定", "reason": fallback}


def _action_primary_label(decision: dict[str, Any], recommended_focus: str) -> str:
    button_labels = {
        "ron": "荣和",
        "tsumo": "自摸",
        "riichi": "立直",
        "kan": "杠",
        "pon": "碰",
        "chi": "吃",
        "skip": "跳过",
        "confirm": "确认",
        "cancel": "取消",
    }
    engine_meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
    single = _single_recommendation(decision)
    if single.get("kind") == "button":
        label = button_labels.get(str(single.get("button_type") or single.get("action") or ""))
        if label:
            return label
    buttons = engine_meta.get("recommended_button_types")
    if not isinstance(buttons, list) or not buttons:
        buttons = decision.get("buttons")
    if isinstance(buttons, list):
        for button in buttons:
            label = button_labels.get(str(button))
            if label:
                return label

    focus_labels = {
        "win_confirmation": "和牌",
        "riichi_decision": "立直",
        "kan_decision": "杠",
        "call_decision": "鸣牌",
        "meld_selection": "选牌",
        "dialog_confirmation": "确认",
        "confirm_or_skip": "跳过",
    }
    return focus_labels.get(recommended_focus, "提醒")


def _single_recommendation(decision: dict[str, Any]) -> dict[str, Any]:
    engine_meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
    single = engine_meta.get("single_recommendation")
    return single if isinstance(single, dict) else {}


def _reason_line(text: str, *, limit: int = 54) -> str:
    value = " ".join(str(text or "").split())
    value = replace_tile_codes_in_text(value)
    if len(value) <= limit:
        return value
    return f"{value[:limit - 1]}..."


def _primary_font_size(text: str) -> int:
    length = len(str(text or ""))
    if length <= 3:
        return 38
    if length <= 5:
        return 34
    if length <= 8:
        return 28
    return 22


def _advice_text(status: dict[str, Any]) -> str:
    if status.get("runtime_mode") == "standby":
        return "已暂停看牌。切回教学模式后会继续自动给建议。"

    if status.get("last_error"):
        return f"先处理：{status['last_error']}"

    decision = status.get("last_decision") if isinstance(status.get("last_decision"), dict) else {}
    decision_type = str(decision.get("decision_type") or status.get("last_decision_type") or "").strip()
    recommended_focus = str(decision.get("recommended_focus") or "").strip()
    if _should_prioritize_decision_text(decision_type, recommended_focus):
        primary_text = _decision_primary_text(decision, status)
        if primary_text:
            return primary_text
    if decision_type != "tile_efficiency_hint":
        suggestion = str(decision.get("suggestion") or status.get("last_narration_text") or "").strip()
        if suggestion:
            return suggestion
    analysis = decision.get("mahjong_analysis") if isinstance(decision.get("mahjong_analysis"), dict) else {}
    candidates = analysis.get("candidate_discards") if isinstance(analysis.get("candidate_discards"), list) else []
    top_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    tile = str(top_candidate.get("tile", "")).strip()
    reason = str(top_candidate.get("reason", "")).strip()
    ukeire = top_candidate.get("ukeire_estimate")
    if tile:
        tile_label = format_tile_label(tile)
        ukeire_text = f" · 进张约 {ukeire}" if ukeire is not None else ""
        reason_text = replace_tile_codes_in_text(reason)
        return f"建议优先考虑：{tile_label}{ukeire_text}" + (f"\n{reason_text}" if reason_text else "")

    suggestion = str(decision.get("suggestion") or status.get("last_narration_text") or "").strip()
    if suggestion:
        return suggestion

    perception = status.get("last_perception") if isinstance(status.get("last_perception"), dict) else {}
    hints = perception.get("analysis_hints") if isinstance(perception.get("analysis_hints"), dict) else {}
    recognized_count = hints.get("recognized_hand_tile_count")
    if status.get("window_bound") and recognized_count:
        return f"已识别到 {recognized_count} 张手牌，正在等轮到你或更清晰的出牌形态。"
    if status.get("window_bound"):
        return "正在看牌局。等识别到你的手牌后会给出建议。"
    return "先选择雀魂窗口并绑定。"


def _scene_label(scene: str) -> str:
    labels = {
        "in_match": "牌局中",
        "dialog": "弹窗/确认",
        "replay": "回放",
        "lobby": "大厅",
        "menu": "菜单",
        "matching": "匹配中",
        "result": "结算",
        "unknown": "未识别",
    }
    return labels.get(scene, scene or "-")


def _should_prioritize_decision_text(decision_type: str, recommended_focus: str) -> bool:
    actionable_focuses = {
        "call_decision",
        "kan_decision",
        "riichi_decision",
        "win_confirmation",
        "dialog_confirmation",
        "confirm_or_skip",
        "meld_selection",
    }
    return recommended_focus in actionable_focuses or decision_type == "danger_action"


def _decision_primary_text(decision: dict[str, Any], status: dict[str, Any]) -> str:
    suggestion = str(decision.get("suggestion") or status.get("last_narration_text") or "").strip()
    if suggestion:
        return suggestion

    summary = str(decision.get("summary") or "").strip()
    detail = str(decision.get("detail") or "").strip()
    if summary and detail:
        return f"{summary}\n{detail}"
    return summary or detail


def _meta_text(status: dict[str, Any]) -> str:
    shanten = status.get("last_shanten_estimate")
    ukeire = status.get("last_ukeire_estimate")
    voice = str(status.get("voice_mode") or "-")
    delivery = str(status.get("last_narration_delivery") or "-")
    parts = [f"语音 {voice}", f"输出 {delivery}"]
    if shanten is not None:
        parts.append(f"向听 {shanten}")
    if ukeire is not None:
        parts.append(f"进张 {ukeire}")
    return " · ".join(parts)


def _render_screen_markers(marker: Any, canvas: Any, status: dict[str, Any]) -> None:
    _hide_marker(marker, canvas)


def _hide_marker(marker: Any, canvas: Any) -> None:
    try:
        canvas.delete("all")
        marker.withdraw()
    except Exception:
        pass


def _parse_marker_box(box: dict[str, Any]) -> dict[str, int]:
    try:
        left = int(box.get("left", 0) or 0)
        top = int(box.get("top", 0) or 0)
        width = int(box.get("width", 0) or 0)
        height = int(box.get("height", 0) or 0)
    except (TypeError, ValueError):
        return {}
    if width <= 0 or height <= 0:
        return {}
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


def _position_overlay(root: Any, status: dict[str, Any]) -> None:
    try:
        left = int(status.get("window_left") or 0)
        top = int(status.get("window_top") or 0)
        width = int(status.get("window_width") or 0)
        overlay_width = 340
        overlay_height = 172
        screen_width = int(root.winfo_screenwidth() or 0)
        outside_x = left + width + 12
        if width and outside_x + overlay_width + 12 <= screen_width:
            x = outside_x
        elif left - overlay_width - 12 >= 0:
            x = left - overlay_width - 12
        else:
            x = left + max(16, width - overlay_width - 28) if width else 80
        y = top + 58 if status.get("window_bound") else 80
        root.geometry(f"{overlay_width}x{overlay_height}+{x}+{y}")
    except Exception:
        root.geometry("340x172+80+80")
