from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts import DecisionResult, PerceivedGameState
from ..session_state import now_iso
from ..storage import load_json_payload, locked_json_path, write_json_atomic
from ..tile_labels import dedupe as _dedupe

logger = logging.getLogger(__name__)


def build_memory_summary(
    candidate: dict[str, Any] | None,
    decision: DecisionResult,
    perceived: PerceivedGameState,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    if decision.priority < 75:
        return None

    tags = _derive_summary_tags(decision)
    if not tags:
        return None

    summary_text = _build_summary_text(decision, perceived, tags)
    coach_note = _build_coach_note(decision)
    return {
        "captured_at": candidate.get("captured_at") or now_iso(),
        "session_id": candidate.get("session_id", ""),
        "summary_text": summary_text,
        "summary_tags": tags,
        "coach_note": coach_note,
        "scene": decision.scene,
        "decision_type": decision.decision_type,
        "priority": decision.priority,
        "risk_level": decision.risk_level,
        "recommended_focus": decision.recommended_focus,
        "review_tags": list(decision.review_tags),
        "reason_codes": list(decision.reason_codes),
        "source_frame_path": candidate.get("frame_path", ""),
        "source_review_dedupe_key": candidate.get("dedupe_key", ""),
        "host_sync_status": "pending",
        "host_sync_note": "sdk_memory_write_unavailable",
        "dedupe_key": _build_memory_dedupe_key(decision, tags),
    }


def stage_memory_summary(
    cache_dir: Path,
    summary: dict[str, Any],
    *,
    enabled: bool = True,
    limit: int = 60,
    dedupe_window_sec: int = 6 * 60 * 60,
    max_memories_per_day: int = 3,
) -> dict[str, Any]:
    queue_path = cache_dir / "memory_bridge_queue.json"
    if not enabled:
        return {
            "staged": False,
            "reason": "memory_bridge_disabled",
            "path": str(queue_path),
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    with locked_json_path(queue_path):
        payload = _load_existing(queue_path)
        items = payload.setdefault("items", [])

        if _daily_limit_reached(items, summary, max_memories_per_day):
            return {
                "staged": False,
                "reason": "daily_limit_reached",
                "path": str(queue_path),
            }

        dedupe_key = str(summary.get("dedupe_key", "")).strip()
        if dedupe_key:
            for existing in reversed(items):
                if str(existing.get("dedupe_key", "")).strip() != dedupe_key:
                    continue
                if _should_skip_duplicate(existing, summary, dedupe_window_sec):
                    return {
                        "staged": False,
                        "reason": "duplicate_summary",
                        "path": str(queue_path),
                    }
                break

        items.append(summary)
        if len(items) > limit:
            payload["items"] = items[-limit:]
        payload["updated_at"] = now_iso()
        write_json_atomic(queue_path, payload)
    return {
        "staged": True,
        "reason": "staged_locally",
        "path": str(queue_path),
        "summary": summary,
    }


def _derive_summary_tags(decision: DecisionResult) -> list[str]:
    tags: list[str] = []
    review_tags = set(decision.review_tags)
    if {"win_window", "high_value_timing"} & review_tags:
        tags.append("mahjong_high_value_timing")
    if "riichi_window" in review_tags:
        tags.append("mahjong_riichi_preference")
    if {"kan_choice", "call_window", "route_choice"} & review_tags:
        tags.append("mahjong_route_choice")
    if decision.risk_level == "high":
        tags.append("mahjong_risk_focus")
    if "low_confidence" in review_tags:
        tags.append("mahjong_needs_review")
    return _dedupe(tags)


def _build_summary_text(
    decision: DecisionResult,
    perceived: PerceivedGameState,
    tags: list[str],
) -> str:
    leading = "、".join(tags[:2]) if tags else "mahjong_session"
    return (
        f"雀魂陪伴记录到一个高价值节点：{decision.summary}"
        f" 焦点是 {decision.recommended_focus or 'observe'}，"
        f"当时识别置信度约为 {perceived.confidence:.2f}，"
        f"可归纳为 {leading}。"
    )


def _build_memory_dedupe_key(decision: DecisionResult, tags: list[str]) -> str:
    return "%s|%s|%s|%s" % (
        decision.decision_type,
        decision.scene,
        decision.recommended_focus,
        ",".join(sorted(tags)),
    )


def _build_coach_note(decision: DecisionResult) -> str:
    suggestion = str(decision.suggestion or "").strip()
    if suggestion:
        return suggestion
    focus = str(decision.recommended_focus or "").strip() or "observe"
    return f"下一轮优先关注 {focus}，避免在低价值分支反复犹豫。"


def _load_existing(path: Path) -> dict[str, Any]:
    data = load_json_payload(
        path,
        default={"updated_at": "", "items": []},
        expected_type=dict,
        logger=logger,
    )
    if not isinstance(data.get("items"), list):
        data["items"] = []
    return data


def _daily_limit_reached(items: list[dict[str, Any]], summary: dict[str, Any], max_memories_per_day: int) -> bool:
    if max_memories_per_day <= 0:
        return False
    current_day = _parse_iso(summary.get("captured_at"))
    if current_day is None:
        return False
    same_day_count = 0
    for item in items:
        item_time = _parse_iso(item.get("captured_at"))
        if item_time is None:
            continue
        if item_time.date() == current_day.date():
            same_day_count += 1
    return same_day_count >= int(max_memories_per_day)


def _should_skip_duplicate(
    existing: dict[str, Any],
    summary: dict[str, Any],
    dedupe_window_sec: int,
) -> bool:
    existing_time = _parse_iso(existing.get("captured_at"))
    current_time = _parse_iso(summary.get("captured_at"))
    if existing_time is None or current_time is None:
        return False
    return abs((current_time - existing_time).total_seconds()) < max(0, int(dedupe_window_sec))


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
