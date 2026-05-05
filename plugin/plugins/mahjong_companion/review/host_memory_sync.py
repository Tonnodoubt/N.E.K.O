from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .game_private_memory import build_public_memory_projection
from .host_memory_writer import HostMemoryWriter, UnavailableHostMemoryWriter
from ..session_state import now_iso
from ..storage import load_json_payload, locked_json_path, write_json_atomic

logger = logging.getLogger(__name__)


def sync_memory_bridge_queue(
    cache_dir: Path,
    *,
    writer: HostMemoryWriter | None = None,
    bucket_id: str = "mahjong_companion_coaching",
    batch_size: int = 5,
) -> tuple[dict[str, Any], Path]:
    queue_path = cache_dir / "memory_bridge_queue.json"
    report_path = cache_dir / "host_memory_sync_report.json"
    batch_limit = max(1, int(batch_size))

    with locked_json_path(queue_path):
        payload = _load_json_payload(queue_path, default={"updated_at": "", "items": []})
        items = payload.get("items", [])
        if not isinstance(items, list):
            items = []
            payload["items"] = items

        attempted_at = now_iso()
        prepared: list[dict[str, Any]] = []
        synced = 0
        skipped = 0
        seen_keys: set[str] = set()

        for item in items:
            if not isinstance(item, dict):
                skipped += 1
                continue
            if str(item.get("host_sync_status", "")).strip() == "synced":
                continue

            coaching_memory, skip_reason = build_coaching_memory(item)
            if coaching_memory is None:
                item["host_sync_status"] = "skipped"
                item["host_sync_note"] = skip_reason or "invalid_summary"
                item["host_sync_attempted_at"] = attempted_at
                skipped += 1
                continue

            dedupe_key = str(coaching_memory.get("dedupe_key", "")).strip()
            if dedupe_key and dedupe_key in seen_keys:
                item["host_sync_status"] = "skipped"
                item["host_sync_note"] = "duplicate_coaching_memory"
                item["host_sync_attempted_at"] = attempted_at
                skipped += 1
                continue

            if dedupe_key:
                seen_keys.add(dedupe_key)
            prepared.append(coaching_memory)

        host_writer = writer or UnavailableHostMemoryWriter()
        writer_name = host_writer.writer_name
        if not host_writer.available:
            for item in items:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("host_sync_status", "")).strip()
                if status in {"synced", "skipped"}:
                    continue
                item["host_sync_status"] = "pending"
                item["host_sync_note"] = "sdk_memory_write_unavailable"
                item["host_sync_attempted_at"] = attempted_at

            payload["updated_at"] = attempted_at
            write_json_atomic(queue_path, payload)

            report = {
                "ok": True,
                "status": "host_memory_write_unavailable",
                "note": "sdk_memory_write_unavailable",
                "bucket_id": bucket_id,
                "writer": "",
                "attempted_at": attempted_at,
                "attempted_count": len(prepared),
                "pending_count": len(prepared),
                "synced_count": 0,
                "skipped_count": skipped,
                "queue_path": str(queue_path),
                "prepared_memories": prepared[:batch_limit],
            }
            write_json_atomic(report_path, report)
            return report, report_path

        for coaching_memory in prepared[:batch_limit]:
            result = host_writer.write(bucket_id, coaching_memory)
            if not result.ok:
                if result.error:
                    logger.warning("host memory writer failed: %s", result.error)
                continue

            synced += 1
            writer_name = result.writer or writer_name
            dedupe_key = str(coaching_memory.get("dedupe_key", "")).strip()
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("dedupe_key", "")).strip() != dedupe_key:
                    continue
                item["host_sync_status"] = "synced"
                item["host_sync_note"] = f"synced_via_{writer_name}"
                item["host_sync_attempted_at"] = attempted_at
                item["host_synced_at"] = attempted_at

        for item in items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("host_sync_status", "")).strip()
            if status:
                continue
            item["host_sync_status"] = "pending"
            item["host_sync_note"] = "awaiting_next_sync"
            item["host_sync_attempted_at"] = attempted_at

        payload["updated_at"] = attempted_at
        write_json_atomic(queue_path, payload)

        pending_count = len([
            item for item in items
            if isinstance(item, dict) and str(item.get("host_sync_status", "")).strip() == "pending"
        ])
        report = {
            "ok": True,
            "status": "host_memory_sync_partial" if pending_count else "host_memory_sync_complete",
            "note": f"writer={writer_name}",
            "bucket_id": bucket_id,
            "writer": writer_name,
            "attempted_at": attempted_at,
            "attempted_count": min(len(prepared), batch_limit),
            "pending_count": pending_count,
            "synced_count": synced,
            "skipped_count": skipped,
            "queue_path": str(queue_path),
            "prepared_memories": prepared[:batch_limit],
        }
        write_json_atomic(report_path, report)
        return report, report_path


def build_coaching_memory(summary: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    summary_text = str(summary.get("summary_text", "")).strip()
    if len(summary_text) < 12:
        return None, "summary_too_short"

    public_projection = build_public_memory_projection(summary)
    tags = _normalize_tags(public_projection.get("summary_tags"))
    if not tags:
        return None, "missing_summary_tags"

    coach_note = str(public_projection.get("coach_note", "")).strip()
    if len(coach_note) < 8:
        return None, "coach_note_too_short"

    confidence = _estimate_memory_confidence(summary)
    if confidence < 0.45:
        return None, "memory_confidence_too_low"

    dedupe_key = str(summary.get("dedupe_key", "")).strip()
    if not dedupe_key:
        dedupe_key = "%s|%s" % (coach_note, ",".join(tags))

    coaching_memory = {
        "memory_type": "mahjong_style_summary",
        "generated_at": str(summary.get("captured_at") or now_iso()),
        "session_id": str(summary.get("session_id", "")).strip(),
        "summary": coach_note,
        "coach_note": coach_note,
        "summary_tags": tags,
        "tags": tags,
        "evidence_count": _estimate_evidence_count(summary),
        "confidence": confidence,
        "scene": str(summary.get("scene", "")).strip(),
        "decision_type": str(summary.get("decision_type", "")).strip(),
        "dedupe_key": dedupe_key,
    }
    return coaching_memory, ""


def _estimate_memory_confidence(summary: dict[str, Any]) -> float:
    priority = int(summary.get("priority", 0) or 0)
    base = min(1.0, max(0.0, priority / 100.0))
    if str(summary.get("risk_level", "")).strip() == "high":
        base += 0.08
    if "mahjong_needs_review" in _normalize_tags(summary.get("summary_tags")):
        base -= 0.18
    return max(0.0, min(0.95, round(base, 2)))


def _estimate_evidence_count(summary: dict[str, Any]) -> int:
    reason_codes = summary.get("reason_codes")
    review_tags = summary.get("review_tags")
    return max(
        1,
        len(reason_codes) if isinstance(reason_codes, list) else 0,
        len(review_tags) if isinstance(review_tags, list) else 0,
    )


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


def _load_json_payload(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    return load_json_payload(path, default=default, expected_type=dict, logger=logger)
