from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..session_state import now_iso
from ..storage import locked_json_path, write_json_atomic
from ..tile_labels import dedupe as _dedupe

logger = logging.getLogger(__name__)


@dataclass
class ReviewSummary:
    session_id: str
    generated_at: str
    source_candidate_count: int
    highlights: list[str] = field(default_factory=list)
    risk_points: list[str] = field(default_factory=list)
    mistake_patterns: list[str] = field(default_factory=list)
    coach_note: str = ""
    memory_bridge_candidates: list[str] = field(default_factory=list)
    summary_text: str = ""
    facts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    training_points: list[str] = field(default_factory=list)
    schema_version: str = "review-summary-v0.3"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_review_summary(
    cache_dir: Path,
    *,
    session_id: str,
) -> tuple[dict[str, Any], Path]:
    candidates_path = cache_dir / "review_candidates.json"
    candidates = load_review_candidates(candidates_path)
    if not candidates:
        raise ValueError("no review candidates available")

    summary = build_review_summary(
        session_id=session_id,
        candidates=candidates,
    )
    summary_path = cache_dir / "review_summary.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with locked_json_path(summary_path):
        write_json_atomic(summary_path, summary)
    return summary, summary_path


def load_review_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError("review candidates file not found: %s" % path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("corrupted review candidates JSON at %s: %s", path, exc)
        raise ValueError("failed to parse review candidates: %s" % exc) from exc
    except OSError as exc:
        logger.warning("failed to read review candidates at %s: %s", path, exc)
        raise ValueError("failed to parse review candidates: %s" % exc) from exc
    if not isinstance(payload, dict):
        raise ValueError("review candidates payload is not a JSON object")
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("review candidates payload has invalid items field")
    normalized = [item for item in items if isinstance(item, dict)]
    return sorted(normalized, key=_sort_key)


def build_review_summary(
    *,
    session_id: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no review candidates available")

    highlights = _collect_highlights(candidates)
    risk_points = _collect_risk_points(candidates)
    mistake_patterns = _collect_mistake_patterns(candidates)
    coach_note = _build_coach_note(candidates)
    memory_bridge_candidates = _derive_memory_bridge_candidates(candidates)
    summary_text = " ".join(part for part in [highlights[0], coach_note] if part).strip()
    summary = ReviewSummary(
        session_id=session_id,
        generated_at=now_iso(),
        source_candidate_count=len(candidates),
        highlights=highlights,
        risk_points=risk_points,
        mistake_patterns=mistake_patterns,
        coach_note=coach_note,
        memory_bridge_candidates=memory_bridge_candidates,
        summary_text=summary_text,
        facts=_collect_facts(candidates),
        risks=_collect_structured_risks(candidates, fallback=risk_points),
        suggestions=_collect_suggestions(candidates, coach_note=coach_note),
        training_points=_collect_training_points(candidates, fallback=mistake_patterns),
    )
    return summary.to_dict()


def _collect_facts(candidates: list[dict[str, Any]]) -> list[str]:
    risk_counts = _count_by(candidates, "risk_level")
    focus_counts = _count_by(candidates, "recommended_focus")
    decision_counts = _count_by(candidates, "decision_type")
    high_risk_count = risk_counts.get("high", 0)
    top_focus = _top_count(focus_counts)
    top_decision = _top_count(decision_counts)

    facts = [
        f"本局沉淀了 {len(candidates)} 个可复盘候选，其中高风险节点 {high_risk_count} 个。",
    ]
    if top_decision:
        facts.append(f"出现最多的决策类型是 {top_decision[0]}，共 {top_decision[1]} 次。")
    if top_focus:
        facts.append(f"本局最集中的复盘焦点是 {top_focus[0]}，共 {top_focus[1]} 次。")
    return facts[:4]


def _collect_structured_risks(
    candidates: list[dict[str, Any]],
    *,
    fallback: list[str],
) -> list[str]:
    risks: list[str] = []
    high_risk = [c for c in candidates if str(c.get("risk_level", "")).strip() == "high"]
    if high_risk:
        risks.append("高风险窗口需要优先复核，尤其是和牌、立直、开杠这类会立即改变局面的节点。")
    if any({"kan_choice", "call_window", "route_choice"} & set(_list(c.get("review_tags"))) for c in candidates):
        risks.append("副露或开杠相关节点容易把路线提前固定，复盘时要确认当时是否真的需要加速。")
    if any("low_confidence" in set(_list(c.get("review_tags"))) for c in candidates):
        risks.append("低置信度节点不能直接当作打法结论，必须和截图或回放一起核验。")
    return _dedupe(risks or list(fallback))[:4]


def _collect_suggestions(candidates: list[dict[str, Any]], *, coach_note: str) -> list[str]:
    tags = _flatten_tags(candidates)
    suggestions: list[str] = []
    if "win_window" in tags:
        suggestions.append("把和牌确认窗口放在最高优先级，先确认按钮语义，再考虑其它动作。")
    if "riichi_window" in tags:
        suggestions.append("立直前固定检查顺序：手牌价值、牌河压力、退路，再决定是否宣言。")
    if {"kan_choice", "call_window", "route_choice"} & tags:
        suggestions.append("吃碰杠前先说清当前路线是推进、防守还是保留弹性，减少临场摇摆。")
    if "tile_efficiency" in tags:
        suggestions.append("中盘先稳定比较孤张、边张和搭子质量，再决定处理哪一张。")
    if coach_note:
        suggestions.append(coach_note)
    return _dedupe(suggestions)[:4]


def _collect_training_points(
    candidates: list[dict[str, Any]],
    *,
    fallback: list[str],
) -> list[str]:
    tags = _flatten_tags(candidates)
    training_points: list[str] = []
    if {"win_window", "high_value_timing"} & tags:
        training_points.append("关键窗口确认速度")
    if "riichi_window" in tags:
        training_points.append("立直前攻守判断")
    if {"kan_choice", "call_window", "route_choice"} & tags:
        training_points.append("副露/开杠路线选择")
    if "tile_efficiency" in tags:
        training_points.append("中盘牌效率弃牌优先级")
    if "low_confidence" in tags:
        training_points.append("低置信度截图复核")
    if training_points:
        return _dedupe(training_points)[:5]
    return _dedupe(list(fallback))[:3]


def _collect_highlights(candidates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for candidate in sorted(candidates, key=_highlight_priority, reverse=True):
        tags = set(_list(candidate.get("review_tags")))
        focus = str(candidate.get("recommended_focus", "")).strip()
        if "win_window" in tags:
            lines.append("你这局出现过高价值和牌确认窗口，最值得保留这种关键时刻的确认节奏。")
        elif "riichi_window" in tags:
            lines.append("这局有明确的立直决策点，说明你已经碰到了值得细想路线的时刻。")
        elif {"kan_choice", "call_window", "route_choice"} & tags:
            lines.append("这局出现过副露或开杠路线选择点，说明节奏取舍本身就是复盘重点。")
        elif "tile_efficiency" in tags:
            lines.append("这局已经出现过可读的中盘牌效率节点，说明你可以开始回看取舍是不是够稳。")
        elif focus == "dialog_confirmation":
            lines.append("这局还有确认类窗口值得回看，别让按钮语义判断拖慢关键节奏。")
        elif focus == "turn_observe":
            lines.append("有几次轮到你关注的阶段，适合回看当时为什么会犹豫。")
        else:
            summary = str(candidate.get("summary", "")).strip()
            if summary:
                lines.append(summary)
        if len(_dedupe(lines)) >= 3:
            break
    deduped = _dedupe(lines)
    return deduped[:3] if deduped else ["当前已有关键节点沉淀，但高光还不算足够集中。"]


def _collect_risk_points(candidates: list[dict[str, Any]]) -> list[str]:
    high_risk = [c for c in candidates if str(c.get("risk_level", "")).strip() == "high"]
    low_confidence = [c for c in candidates if "low_confidence" in set(_list(c.get("review_tags")))]
    route_choices = [
        c for c in candidates
        if {"kan_choice", "call_window", "route_choice"} & set(_list(c.get("review_tags")))
    ]

    lines: list[str] = []
    if high_risk:
        lines.append(f"这局一共出现了 {len(high_risk)} 次高风险决策点，关键窗口并不少。")
    if route_choices:
        lines.append(f"其中有 {len(route_choices)} 次更像路线选择题，适合回看当时是不是太急着推进。")
    tile_efficiency = [c for c in candidates if "tile_efficiency" in set(_list(c.get("review_tags")))]
    if tile_efficiency:
        lines.append(f"另有 {len(tile_efficiency)} 次已经带牌理语义的中盘选择，适合回看节奏是不是过于保守或激进。")
    if low_confidence:
        lines.append(f"还有 {len(low_confidence)} 次节点识别置信度偏低，复盘时最好结合截图二次确认。")
    return lines or ["这一局的风险点还不算密集，但关键节点确认节奏仍值得继续练。"]


def _collect_mistake_patterns(candidates: list[dict[str, Any]]) -> list[str]:
    tags = _flatten_tags(candidates)
    lines: list[str] = []
    if "low_confidence" in tags:
        lines.append("当前样本里有一部分更像信息不够清晰时的犹豫，而不是明确的牌效率失误。")
    if {"kan_choice", "call_window", "route_choice"} & tags:
        lines.append("这局更容易出现的是路线取舍问题，尤其是吃碰或开杠时机的判断。")
    if "tile_efficiency" in tags:
        lines.append("当前已经能沉淀轻量牌效率节点，下一步适合回看哪些弃牌方向总是拖慢节奏。")
    if "dialog_confirm" in tags:
        lines.append("确认类窗口也值得回看，避免把关键确认拖成额外犹豫。")
    if not lines:
        lines.append("当前样本更像是按钮决策节奏问题，而不是完整牌理层面的明确错误。")
    return lines[:3]


def _build_coach_note(candidates: list[dict[str, Any]]) -> str:
    tags = _flatten_tags(candidates)
    if "win_window" in tags:
        return "这局最值得继续练的是关键窗口出现时的确认速度，先把高价值时刻稳稳抓住。"
    if {"kan_choice", "call_window", "route_choice"} & tags:
        return "后面可以重点练路线取舍：先确认节奏，再决定要不要副露或开杠。"
    if "tile_efficiency" in tags:
        return "后面可以开始练轻量牌效率判断，尤其是中盘该先处理哪类孤张。"
    if "low_confidence" in tags:
        return "下一步更适合先提升截图和识别稳定度，再去追更细的牌理建议。"
    return "这局已经有可读的关键节点了，下一步适合把这些节点继续串成更完整的复盘。"


def _derive_memory_bridge_candidates(candidates: list[dict[str, Any]]) -> list[str]:
    mapped: list[str] = []
    for candidate in candidates:
        tags = set(_list(candidate.get("review_tags")))
        risk_level = str(candidate.get("risk_level", "")).strip()
        if {"win_window", "high_value_timing"} & tags:
            mapped.append("mahjong_high_value_timing")
        if "riichi_window" in tags:
            mapped.append("mahjong_riichi_preference")
        if {"kan_choice", "call_window", "route_choice"} & tags:
            mapped.append("mahjong_route_choice")
        if "tile_efficiency" in tags:
            mapped.append("mahjong_tile_efficiency")
        if risk_level == "high":
            mapped.append("mahjong_risk_focus")
        if "low_confidence" in tags:
            mapped.append("mahjong_needs_review")
    return _dedupe(mapped)


def _flatten_tags(candidates: list[dict[str, Any]]) -> set[str]:
    flattened: set[str] = set()
    for candidate in candidates:
        flattened.update(_list(candidate.get("review_tags")))
    return flattened


def _count_by(candidates: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        value = str(candidate.get(field_name, "")).strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _top_count(counts: dict[str, int]) -> tuple[str, int] | None:
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _highlight_priority(candidate: dict[str, Any]) -> tuple[int, int]:
    tags = set(_list(candidate.get("review_tags")))
    priority = int(candidate.get("priority", 0) or 0)
    bonus = 0
    if "win_window" in tags:
        bonus += 30
    if "riichi_window" in tags:
        bonus += 20
    if {"kan_choice", "call_window", "route_choice"} & tags:
        bonus += 10
    return priority + bonus, priority


def _sort_key(candidate: dict[str, Any]) -> tuple[str, int]:
    return str(candidate.get("captured_at", "")), int(candidate.get("priority", 0) or 0)


def _list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
