from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session_state import now_iso
from ..storage import load_json_payload, locked_json_path, write_json_atomic

logger = logging.getLogger(__name__)

FOCUS_BY_MEMORY_TAG = {
    "mahjong_high_value_timing": "high_value_confirmation",
    "mahjong_riichi_preference": "riichi_decision",
    "mahjong_route_choice": "call_decision",
    "mahjong_tile_efficiency": "tile_efficiency",
    "mahjong_risk_focus": "defense_judgement",
    "mahjong_needs_review": "perception_stability",
}

FOCUS_LABELS = {
    "high_value_confirmation": "关键和牌确认",
    "riichi_decision": "立直判断",
    "call_decision": "副露路线选择",
    "tile_efficiency": "中盘牌效率",
    "defense_judgement": "风险停一拍",
    "perception_stability": "识别稳定度",
}

COACH_FOCUS_TEXT = {
    "high_value_confirmation": "先保证关键和牌窗口不漏确认",
    "riichi_decision": "先把立直前的确认顺序固定下来",
    "call_decision": "先减少中盘路线摇摆",
    "tile_efficiency": "先固定中盘的弃牌优先级",
    "defense_judgement": "高风险窗口先停一拍再操作",
    "perception_stability": "先提高识别稳定度，再追更细牌理",
}


def generate_coaching_trend(
    cache_dir: Path,
    *,
    session_window: int = 3,
) -> tuple[dict[str, Any], Path]:
    history = load_review_summary_history(cache_dir / "review_summary_history.json")
    queue_payload = _load_json_payload(cache_dir / "memory_bridge_queue.json", default={"items": []})
    pending_memories = queue_payload.get("items", [])
    if not isinstance(pending_memories, list):
        pending_memories = []

    trend = build_trend_summary(
        review_summaries=history,
        pending_memories=[item for item in pending_memories if isinstance(item, dict)],
        session_window=session_window,
    )
    trend_path = cache_dir / "coaching_trend.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with locked_json_path(trend_path):
        write_json_atomic(trend_path, trend)
    return trend, trend_path


def append_review_summary_history(
    cache_dir: Path,
    summary: dict[str, Any],
    *,
    limit: int = 24,
) -> Path:
    history_path = cache_dir / "review_summary_history.json"
    with locked_json_path(history_path):
        payload = _load_json_payload(history_path, default={"updated_at": "", "items": []})
        items = payload.get("items", [])
        if not isinstance(items, list):
            items = []

        session_id = str(summary.get("session_id", "")).strip()
        replaced = False
        if session_id:
            for index, existing in enumerate(items):
                if not isinstance(existing, dict):
                    continue
                if str(existing.get("session_id", "")).strip() != session_id:
                    continue
                items[index] = dict(summary)
                replaced = True
                break

        if not replaced:
            items.append(dict(summary))

        items = sorted(
            [item for item in items if isinstance(item, dict)],
            key=_summary_sort_key,
        )[-max(1, int(limit)):]
        payload["items"] = items
        payload["updated_at"] = now_iso()
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(history_path, payload)
    return history_path


def load_review_summary_history(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_payload(path, default={"items": []})
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    normalized = [item for item in items if isinstance(item, dict)]
    return sorted(normalized, key=_summary_sort_key)


def build_trend_summary(
    *,
    review_summaries: list[dict[str, Any]],
    pending_memories: list[dict[str, Any]] | None = None,
    session_window: int = 3,
) -> dict[str, Any]:
    normalized_summaries = [item for item in review_summaries if isinstance(item, dict)]
    normalized_summaries = sorted(normalized_summaries, key=_summary_sort_key)
    window_size = max(1, int(session_window))
    recent_summaries = normalized_summaries[-window_size:]
    queue_items = [item for item in (pending_memories or []) if isinstance(item, dict)]

    if not recent_summaries and not queue_items:
        return {
            "generated_at": now_iso(),
            "window": f"last_{window_size}_sessions",
            "source_session_count": 0,
            "source_session_ids": [],
            "pending_memory_count": 0,
            "style_bias": "neutral",
            "common_hesitations": [],
            "repeated_mistake_patterns": [],
            "focus_counts": {},
            "tag_counts": {},
            "coach_focus": "observe",
            "coach_focus_text": "先继续积累可解释的复盘样本",
            "summary_text": "当前还没有足够的复盘样本，先继续积累关键节点。",
        }

    tag_counts: dict[str, int] = {}
    focus_counts: dict[str, int] = {}
    session_ids: list[str] = []

    for summary in recent_summaries:
        session_id = str(summary.get("session_id", "")).strip()
        if session_id:
            session_ids.append(session_id)
        for tag in _normalize_tags(summary.get("memory_bridge_candidates")):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            focus_id = FOCUS_BY_MEMORY_TAG.get(tag)
            if focus_id:
                focus_counts[focus_id] = focus_counts.get(focus_id, 0) + 1

    for item in queue_items:
        for tag in _normalize_tags(item.get("summary_tags")):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            focus_id = FOCUS_BY_MEMORY_TAG.get(tag)
            if focus_id:
                focus_counts[focus_id] = focus_counts.get(focus_id, 0) + 1

    sorted_focuses = sorted(focus_counts.items(), key=lambda item: (-item[1], item[0]))
    common_hesitations = [focus_id for focus_id, _ in sorted_focuses[:3]]
    repeated_mistake_patterns = _detect_repeated_mistake_patterns(recent_summaries)
    coach_focus = (
        str(repeated_mistake_patterns[0]["pattern_id"])
        if repeated_mistake_patterns
        else common_hesitations[0] if common_hesitations else "observe"
    )

    aggressive_score = (
        tag_counts.get("mahjong_high_value_timing", 0)
        + tag_counts.get("mahjong_riichi_preference", 0)
        + tag_counts.get("mahjong_route_choice", 0)
    )
    defensive_score = (
        tag_counts.get("mahjong_risk_focus", 0)
        + tag_counts.get("mahjong_needs_review", 0)
    )
    style_bias = _derive_style_bias(aggressive_score, defensive_score)
    summary_text = _build_trend_summary_text(
        source_count=len(recent_summaries),
        queue_count=len(queue_items),
        common_hesitations=common_hesitations,
        style_bias=style_bias,
    )

    return {
        "generated_at": now_iso(),
        "window": f"last_{window_size}_sessions",
        "source_session_count": len(recent_summaries),
        "source_session_ids": session_ids,
        "pending_memory_count": len(queue_items),
        "style_bias": style_bias,
        "common_hesitations": common_hesitations,
        "repeated_mistake_patterns": repeated_mistake_patterns,
        "focus_counts": focus_counts,
        "tag_counts": tag_counts,
        "coach_focus": coach_focus,
        "coach_focus_text": COACH_FOCUS_TEXT.get(coach_focus, "先继续积累可解释的复盘样本"),
        "summary_text": summary_text,
    }


def _detect_repeated_mistake_patterns(recent_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sessions_by_pattern: dict[str, list[str]] = {}
    evidence_by_pattern: dict[str, list[str]] = {}
    for index, summary in enumerate(recent_summaries, start=1):
        session_id = str(summary.get("session_id", "")).strip() or f"session_{index}"
        pattern_ids = _pattern_ids_from_summary(summary)
        for pattern_id in pattern_ids:
            sessions = sessions_by_pattern.setdefault(pattern_id, [])
            if session_id not in sessions:
                sessions.append(session_id)
            for evidence in _pattern_evidence(summary, pattern_id):
                evidence_bucket = evidence_by_pattern.setdefault(pattern_id, [])
                if evidence not in evidence_bucket:
                    evidence_bucket.append(evidence)

    repeated: list[dict[str, Any]] = []
    for pattern_id, session_ids in sessions_by_pattern.items():
        count = len(session_ids)
        if count < 2:
            continue
        label = FOCUS_LABELS.get(pattern_id, pattern_id)
        severity = "high" if count >= 3 or pattern_id == "defense_judgement" else "medium"
        repeated.append({
            "pattern_id": pattern_id,
            "label": label,
            "session_count": count,
            "source_session_ids": session_ids,
            "severity": severity,
            "evidence": evidence_by_pattern.get(pattern_id, [])[:3],
            "summary": f"最近 {count} 局重复出现 {label} 相关问题。",
            "recommendation": COACH_FOCUS_TEXT.get(pattern_id, "先继续积累可解释的复盘样本"),
        })

    return sorted(
        repeated,
        key=lambda item: (-int(item["session_count"]), str(item["pattern_id"])),
    )[:5]


def _pattern_ids_from_summary(summary: dict[str, Any]) -> set[str]:
    pattern_ids: set[str] = set()
    for tag in _normalize_tags(summary.get("memory_bridge_candidates")):
        focus_id = FOCUS_BY_MEMORY_TAG.get(tag)
        if focus_id:
            pattern_ids.add(focus_id)

    for training_point in _normalize_tags(summary.get("training_points")):
        pattern_id = _pattern_id_from_text(training_point)
        if pattern_id:
            pattern_ids.add(pattern_id)

    for pattern in _normalize_tags(summary.get("mistake_patterns")):
        pattern_id = _pattern_id_from_text(pattern)
        if pattern_id:
            pattern_ids.add(pattern_id)
    return pattern_ids


def _pattern_evidence(summary: dict[str, Any], pattern_id: str) -> list[str]:
    evidence: list[str] = []
    for value in [
        *(_normalize_tags(summary.get("training_points"))),
        *(_normalize_tags(summary.get("mistake_patterns"))),
        *(_normalize_tags(summary.get("risk_points"))),
    ]:
        if _pattern_id_from_text(value) == pattern_id or FOCUS_LABELS.get(pattern_id, "") in value:
            evidence.append(value)
    return evidence[:3]


def _pattern_id_from_text(value: str) -> str:
    text = str(value)
    if any(token in text for token in ["和牌", "高价值", "关键窗口"]):
        return "high_value_confirmation"
    if "立直" in text:
        return "riichi_decision"
    if any(token in text for token in ["副露", "吃碰", "开杠", "路线"]):
        return "call_decision"
    if any(token in text for token in ["牌效率", "弃牌", "孤张", "中盘"]):
        return "tile_efficiency"
    if any(token in text for token in ["高风险", "防守", "现物", "风险"]):
        return "defense_judgement"
    if any(token in text for token in ["低置信度", "识别", "截图"]):
        return "perception_stability"
    return ""


def _build_trend_summary_text(
    *,
    source_count: int,
    queue_count: int,
    common_hesitations: list[str],
    style_bias: str,
) -> str:
    labels = [FOCUS_LABELS.get(item, item) for item in common_hesitations[:2]]
    joined = "、".join(labels) if labels else "关键节点确认"
    if source_count > 0:
        return f"最近 {source_count} 局里，{joined} 是反复出现的训练主题，整体风格 {style_bias}。"
    return f"最近 {queue_count} 条高价值节点里，{joined} 最值得先练，整体风格 {style_bias}。"


def _derive_style_bias(aggressive_score: int, defensive_score: int) -> str:
    delta = aggressive_score - defensive_score
    if aggressive_score >= 3 and defensive_score >= 3 and abs(delta) <= 1:
        return "swingy"
    if delta >= 2:
        return "slightly_aggressive"
    if delta <= -2:
        return "slightly_defensive"
    return "neutral"


def _summary_sort_key(summary: dict[str, Any]) -> tuple[datetime, str]:
    timestamp = _parse_iso(summary.get("generated_at")) or _parse_iso(summary.get("captured_at")) or datetime.min.replace(tzinfo=timezone.utc)
    return timestamp, str(summary.get("session_id", ""))


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        tag = str(item).strip()
        if not tag or tag in tags:
            continue
        tags.append(tag)
    return tags


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _load_json_payload(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    return load_json_payload(path, default=default, expected_type=dict, logger=logger)
