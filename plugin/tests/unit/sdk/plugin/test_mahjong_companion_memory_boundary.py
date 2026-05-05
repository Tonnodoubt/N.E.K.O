from __future__ import annotations

import json
from pathlib import Path

from plugin.plugins.mahjong_companion.review.game_private_memory import (
    append_game_private_memory,
    build_game_private_record,
    build_public_memory_projection,
)
from plugin.plugins.mahjong_companion.review.host_memory_sync import build_coaching_memory, sync_memory_bridge_queue
from plugin.plugins.mahjong_companion.review.host_memory_writer import UnavailableHostMemoryWriter
from plugin.plugins.mahjong_companion.contracts import DecisionResult, PerceivedGameState


def test_public_projection_exposes_only_summary_tags_and_coach_note() -> None:
    raw_summary = {
        "summary_text": "这一手更适合稳住两面搭子的进张效率。",
        "summary_tags": ["mahjong_tile_efficiency", "mahjong_route_choice"],
        "coach_note": "先稳两面，再决定是否副露。",
        "raw_detections": [{"class": "tile", "score": 0.9}],
        "analysis_hints": {"private": True},
        "hand_tiles": ["1m", "2m", "3m"],
    }

    projection = build_public_memory_projection(raw_summary)

    assert set(projection.keys()) == {"summary_tags", "coach_note"}
    assert projection["summary_tags"] == ["mahjong_tile_efficiency", "mahjong_route_choice"]
    assert projection["coach_note"] == "先稳两面，再决定是否副露。"


def test_build_coaching_memory_does_not_leak_private_fields() -> None:
    summary = {
        "captured_at": "2026-04-20T00:00:00+00:00",
        "summary_text": "最近已经开始出现中盘牌效率样本，适合固定弃牌优先级。",
        "summary_tags": ["mahjong_tile_efficiency"],
        "coach_note": "先稳住两面搭子的弃牌顺序。",
        "priority": 82,
        "risk_level": "low",
        "dedupe_key": "memory-boundary-1",
        "raw_detections": [{"class": "tile", "score": 0.97}],
        "analysis_hints": {"engine": "private"},
    }

    coaching_memory, reason = build_coaching_memory(summary)

    assert reason == ""
    assert coaching_memory is not None
    assert coaching_memory["summary_tags"] == ["mahjong_tile_efficiency"]
    assert coaching_memory["coach_note"] == "先稳住两面搭子的弃牌顺序。"
    assert "raw_detections" not in coaching_memory
    assert "analysis_hints" not in coaching_memory


def test_game_private_memory_stays_local_and_host_sync_uses_summary_boundary(tmp_path: Path) -> None:
    cache_dir = tmp_path / "session_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    decision = DecisionResult(
        decision_type="tile_efficiency_hint",
        priority=84,
        risk_level="medium",
        summary="更适合先稳住牌效率。",
        detail="当前中张搭子可优先保留。",
        suggestion="先打边张，保留两面。",
        recommended_focus="tile_efficiency",
        scene="in_match",
        reason_codes=["analysis.tile_level_available"],
        review_tags=["tile_efficiency"],
    )
    perceived = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        hand_tiles=["1m", "2m", "3m", "4p", "5p", "6p"],
        raw_detections=[{"class": "tile_1m", "score": 0.99}],
        analysis_hints={"private_trace": "abc"},
    )
    candidate = {
        "captured_at": "2026-04-20T00:00:00+00:00",
        "session_id": "mahjong-memory-boundary",
        "frame_path": "/tmp/frame.png",
        "dedupe_key": "boundary|tile_efficiency",
    }

    record = build_game_private_record(candidate, decision, perceived)
    private_path = append_game_private_memory(cache_dir, record)
    private_payload = json.loads(private_path.read_text(encoding="utf-8"))

    assert private_path.name == "game_private_memory.json"
    assert private_payload["items"]
    assert "raw_detections" in private_payload["items"][0]

    queue_path = cache_dir / "memory_bridge_queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "captured_at": "2026-04-20T00:00:00+00:00",
                        "summary_text": "最近已经开始出现中盘牌效率样本，适合固定弃牌优先级。",
                        "summary_tags": ["mahjong_tile_efficiency"],
                        "coach_note": "先稳住两面搭子的弃牌顺序。",
                        "priority": 84,
                        "risk_level": "medium",
                        "dedupe_key": "boundary|tile_efficiency",
                        "raw_detections": [{"class": "tile", "score": 0.99}],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report, _ = sync_memory_bridge_queue(cache_dir, writer=UnavailableHostMemoryWriter(), batch_size=5)

    assert report["status"] == "host_memory_write_unavailable"
    assert report["prepared_memories"]
    prepared = report["prepared_memories"][0]
    assert prepared["summary_tags"] == ["mahjong_tile_efficiency"]
    assert prepared["coach_note"] == "先稳住两面搭子的弃牌顺序。"
    assert "raw_detections" not in prepared
