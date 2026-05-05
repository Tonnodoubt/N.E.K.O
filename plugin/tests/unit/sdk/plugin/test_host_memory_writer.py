from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plugin.plugins.mahjong_companion.review.host_memory_sync import sync_memory_bridge_queue
from plugin.plugins.mahjong_companion.review.host_memory_writer import (
    SdkHostMemoryWriter,
    UnavailableHostMemoryWriter,
)


class _MemoryWithPut:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def put(self, *, bucket_id: str, payload: dict[str, Any]) -> bool:
        self.calls.append({"bucket_id": bucket_id, "payload": payload})
        return True


class _FailingMemory:
    def write(self, *, bucket_id: str, payload: dict[str, Any]) -> bool:
        raise RuntimeError("memory offline")


def test_unavailable_host_memory_writer_reports_existing_status_shape() -> None:
    writer = UnavailableHostMemoryWriter()

    result = writer.write("mahjong_companion_coaching", {"summary": "x"})

    assert writer.available is False
    assert result.ok is False
    assert result.status == "host_memory_write_unavailable"
    assert result.writer == ""
    assert result.error == "sdk_memory_write_unavailable"


def test_sdk_host_memory_writer_wraps_supported_memory_method() -> None:
    memory = _MemoryWithPut()
    writer = SdkHostMemoryWriter(memory)

    result = writer.write("mahjong_companion_coaching", {"summary": "keep shape"})

    assert writer.available is True
    assert writer.writer_name == "put"
    assert result.ok is True
    assert result.status == "host_memory_write_complete"
    assert result.writer == "put"
    assert memory.calls == [
        {
            "bucket_id": "mahjong_companion_coaching",
            "payload": {"summary": "keep shape"},
        }
    ]


def test_sdk_host_memory_writer_returns_failure_result_on_exception() -> None:
    writer = SdkHostMemoryWriter(_FailingMemory())

    result = writer.write("mahjong_companion_coaching", {"summary": "x"})

    assert writer.available is True
    assert writer.writer_name == "write"
    assert result.ok is False
    assert result.status == "host_memory_write_failed"
    assert result.writer == "write"
    assert result.error == "memory offline"


def test_sync_memory_bridge_queue_uses_writer_without_changing_report_shape(tmp_path: Path) -> None:
    cache_dir = tmp_path / "session_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    queue_path = cache_dir / "memory_bridge_queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "captured_at": "2026-05-01T00:00:00+00:00",
                        "summary_text": "最近这一手更像是中盘副露路线的犹豫点，需要继续观察。",
                        "summary_tags": ["mahjong_route_choice"],
                        "coach_note": "先确认副露后还能不能维持好形。",
                        "priority": 82,
                        "risk_level": "medium",
                        "review_tags": ["route_choice"],
                        "reason_codes": ["button.chi_visible"],
                        "dedupe_key": "route-choice-sync",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    memory = _MemoryWithPut()

    report, report_path = sync_memory_bridge_queue(
        cache_dir,
        writer=SdkHostMemoryWriter(memory),
        batch_size=5,
    )

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "host_memory_sync_complete"
    assert report["writer"] == "put"
    assert report["attempted_count"] == 1
    assert report["pending_count"] == 0
    assert report["synced_count"] == 1
    assert set(report) == set(report_payload)
    assert payload["items"][0]["host_sync_status"] == "synced"
    assert payload["items"][0]["host_sync_note"] == "synced_via_put"
    assert memory.calls[0]["bucket_id"] == "mahjong_companion_coaching"
