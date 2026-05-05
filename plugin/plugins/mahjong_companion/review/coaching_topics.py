from __future__ import annotations

from pathlib import Path
from typing import Any

from ..session_state import now_iso
from ..storage import locked_json_path, write_json_atomic

TOPIC_DEFINITIONS = {
    "high_value_confirmation": {
        "title": "关键窗口先确认",
        "summary": "最近几局的高价值按钮确认值得继续稳住。",
        "recommendation": "先确认和牌与高价值操作，再处理次要按钮。",
    },
    "riichi_decision": {
        "title": "立直前停一拍",
        "summary": "立直窗口已经反复出现，值得固定确认顺序。",
        "recommendation": "先看牌河、打点与退路，再决定要不要立直。",
    },
    "call_decision": {
        "title": "减少副露摇摆",
        "summary": "中盘吃碰与路线选择是近期最常见的犹豫点。",
        "recommendation": "先定推进目标，减少中盘路线摇摆，再判断这口副露是不是在帮当前路线。",
    },
    "tile_efficiency": {
        "title": "固定弃牌优先级",
        "summary": "中盘牌效率已经能持续沉淀，适合开始练稳定取舍。",
        "recommendation": "优先保留更连贯的块，再处理孤张和改善弱的牌。",
    },
    "defense_judgement": {
        "title": "高风险窗口先刹车",
        "summary": "近期高风险节点出现较多，说明防守确认还需要加强。",
        "recommendation": "在高风险按钮出现时先停一拍，确认牌河和场况再操作。",
    },
    "perception_stability": {
        "title": "先稳识别再加深牌理",
        "summary": "当前还有一部分样本更像识别信息不足时的犹豫。",
        "recommendation": "先提升截图与识别稳定度，再继续追求更细的牌理建议。",
    },
}


def generate_coaching_topics(
    cache_dir: Path,
    trend: dict[str, Any],
    *,
    topic_limit: int = 3,
) -> tuple[dict[str, Any], Path]:
    payload = build_coaching_topics(trend, topic_limit=topic_limit)
    topics_path = cache_dir / "coaching_topics.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with locked_json_path(topics_path):
        write_json_atomic(topics_path, payload)
    return payload, topics_path


def build_coaching_topics(
    trend: dict[str, Any],
    *,
    topic_limit: int = 3,
) -> dict[str, Any]:
    focus_counts = trend.get("focus_counts")
    normalized_counts = focus_counts if isinstance(focus_counts, dict) else {}
    ranked = sorted(
        normalized_counts.items(),
        key=lambda item: (-int(item[1] or 0), str(item[0])),
    )
    total = max(1, sum(int(value or 0) for _, value in ranked))

    topics: list[dict[str, Any]] = []
    for focus_id, count in ranked[: max(1, int(topic_limit))]:
        definition = TOPIC_DEFINITIONS.get(str(focus_id))
        if definition is None:
            continue
        topics.append({
            "topic_id": focus_id,
            "title": definition["title"],
            "summary": definition["summary"],
            "recommendation": definition["recommendation"],
            "count": int(count or 0),
            "confidence": round(min(0.95, max(0.3, int(count or 0) / total)), 2),
        })

    if not topics:
        topics.append({
            "topic_id": "observe",
            "title": "继续积累样本",
            "summary": "当前跨局样本还不够多，先继续沉淀可解释的关键节点。",
            "recommendation": "优先保留高价值窗口和牌效率样本，再开始更细的训练总结。",
            "count": 0,
            "confidence": 0.3,
        })

    headline = str(trend.get("coach_focus_text", "")).strip() or topics[0]["recommendation"]
    return {
        "generated_at": now_iso(),
        "coach_focus": str(trend.get("coach_focus", "")).strip(),
        "headline": headline,
        "topics": topics,
        "summary_text": str(trend.get("summary_text", "")).strip(),
    }
