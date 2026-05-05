from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts import DecisionResult, PerceivedGameState
from ..session_state import now_iso
from ..storage import load_json_payload, locked_json_path, write_json_atomic

logger = logging.getLogger(__name__)


def build_review_candidate(
    frame_path: Path | None,
    decision: DecisionResult,
    perceived: PerceivedGameState,
) -> dict[str, Any] | None:
    if not decision.review_tags and decision.priority < 60:
        return None

    dedupe_key = "%s|%s|%s|%s" % (
        decision.decision_type,
        decision.scene,
        ",".join(sorted(decision.buttons)),
        ",".join(sorted(decision.review_tags)),
    )
    return {
        "captured_at": now_iso(),
        "frame_path": str(frame_path) if frame_path is not None else "",
        "scene": decision.scene,
        "decision_type": decision.decision_type,
        "priority": decision.priority,
        "risk_level": decision.risk_level,
        "summary": decision.summary,
        "detail": decision.detail,
        "suggestion": decision.suggestion,
        "recommended_focus": decision.recommended_focus,
        "buttons": list(decision.buttons),
        "reason_codes": list(decision.reason_codes),
        "review_tags": list(decision.review_tags),
        "perception_confidence": perceived.confidence,
        "perception_notes": list(perceived.notes),
        "dedupe_key": dedupe_key,
    }


def append_review_candidate(
    cache_dir: Path,
    candidate: dict[str, Any],
    *,
    limit: int = 80,
    dedupe_window_sec: int = 20,
) -> Path:
    append_review_candidate.last_appended = False  # type: ignore[attr-defined]
    cache_dir.mkdir(parents=True, exist_ok=True)
    review_path = cache_dir / "review_candidates.json"
    with locked_json_path(review_path):
        payload = _load_existing(review_path)

        candidates = payload.setdefault("items", [])
        dedupe_key = str(candidate.get("dedupe_key", "")).strip()
        if dedupe_key:
            for existing in reversed(candidates):
                if str(existing.get("dedupe_key", "")).strip() != dedupe_key:
                    continue
                if _should_skip_duplicate(existing, candidate, dedupe_window_sec):
                    return review_path
                break

        candidates.append(candidate)
        append_review_candidate.last_appended = True  # type: ignore[attr-defined]
        if len(candidates) > limit:
            payload["items"] = candidates[-limit:]
        payload["updated_at"] = now_iso()
        write_json_atomic(review_path, payload)
    return review_path


def _load_existing(review_path: Path) -> dict[str, Any]:
    data = load_json_payload(
        review_path,
        default={"updated_at": "", "items": []},
        expected_type=dict,
        logger=logger,
    )
    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    return data


def _should_skip_duplicate(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    dedupe_window_sec: int,
) -> bool:
    existing_frame = str(existing.get("frame_path", "")).strip()
    candidate_frame = str(candidate.get("frame_path", "")).strip()
    if existing_frame and candidate_frame and existing_frame == candidate_frame:
        return True

    existing_time = _parse_iso(existing.get("captured_at"))
    candidate_time = _parse_iso(candidate.get("captured_at"))
    if existing_time is None or candidate_time is None:
        return False
    return abs((candidate_time - existing_time).total_seconds()) < max(0, int(dedupe_window_sec))


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
