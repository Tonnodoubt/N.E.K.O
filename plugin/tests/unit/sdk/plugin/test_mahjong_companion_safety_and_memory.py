from __future__ import annotations

import json
from pathlib import Path

from plugin.plugins.mahjong_companion.action.human_override_guard import HumanOverrideGuard
from plugin.plugins.mahjong_companion.contracts import DecisionResult, PerceivedGameState
from plugin.plugins.mahjong_companion.review.memory_bridge import build_memory_summary, stage_memory_summary


def test_human_override_guard_aborts_on_large_pointer_move() -> None:
    guard = HumanOverrideGuard()
    guard.arm(
        enabled=True,
        active_window_sec=1.5,
        movement_threshold_px=12,
        pointer=(100, 200),
        now_monotonic=10.0,
    )

    decision = guard.evaluate(pointer=(120, 200), now_monotonic=10.3)

    assert decision.should_abort is True
    assert decision.reason == "human_override_detected"
    assert decision.armed is False


def test_human_override_guard_expires_without_abort() -> None:
    guard = HumanOverrideGuard()
    guard.arm(
        enabled=True,
        active_window_sec=0.5,
        movement_threshold_px=12,
        pointer=(100, 200),
        now_monotonic=10.0,
    )

    decision = guard.evaluate(pointer=(102, 201), now_monotonic=10.7)

    assert decision.should_abort is False
    assert decision.reason == "guard_expired"
    assert decision.armed is False


def test_memory_bridge_stages_summary_and_applies_dedupe_and_daily_limit(tmp_path: Path) -> None:
    decision = DecisionResult(
        decision_type="danger_action",
        priority=96,
        risk_level="high",
        action_required=True,
        speakable=True,
        summary="当前像是出现了和牌窗口。",
        detail="检测到 ron 按钮。",
        suggestion="先确认和牌条件与按钮语义。",
        recommended_focus="win_confirmation",
        scene="in_match",
        buttons=["ron"],
        reason_codes=["button.ron_visible"],
        review_tags=["win_window", "high_value_timing"],
        engine_meta={"engine": "rule_based_v2"},
    )
    perceived = PerceivedGameState(
        scene="in_match",
        confidence=0.88,
        is_user_turn=True,
        buttons=["ron"],
        notes=["bottom action bar detected"],
    )
    candidate = {
        "captured_at": "2026-04-15T00:00:00+00:00",
        "frame_path": "/tmp/frame-1.png",
        "dedupe_key": "danger_action|in_match|ron|win_window",
    }

    summary = build_memory_summary(candidate, decision, perceived)

    assert summary is not None
    assert "mahjong_high_value_timing" in summary["summary_tags"]

    cache_dir = tmp_path / "session_cache"
    first = stage_memory_summary(
        cache_dir,
        summary,
        enabled=True,
        dedupe_window_sec=3600,
        max_memories_per_day=2,
    )
    second = stage_memory_summary(
        cache_dir,
        dict(summary),
        enabled=True,
        dedupe_window_sec=3600,
        max_memories_per_day=2,
    )

    third_summary = dict(summary)
    third_summary["captured_at"] = "2026-04-15T08:00:00+00:00"
    third_summary["source_frame_path"] = "/tmp/frame-2.png"
    third = stage_memory_summary(
        cache_dir,
        third_summary,
        enabled=True,
        dedupe_window_sec=3600,
        max_memories_per_day=2,
    )

    fourth_summary = dict(summary)
    fourth_summary["captured_at"] = "2026-04-15T16:00:00+00:00"
    fourth_summary["source_frame_path"] = "/tmp/frame-3.png"
    fourth = stage_memory_summary(
        cache_dir,
        fourth_summary,
        enabled=True,
        dedupe_window_sec=3600,
        max_memories_per_day=2,
    )

    payload = json.loads((cache_dir / "memory_bridge_queue.json").read_text(encoding="utf-8"))

    assert first["staged"] is True
    assert second["staged"] is False
    assert second["reason"] == "duplicate_summary"
    assert third["staged"] is True
    assert fourth["staged"] is False
    assert fourth["reason"] == "daily_limit_reached"
    assert len(payload["items"]) == 2
