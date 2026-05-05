from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_defaults import DEFAULT_CONFIG, merge_runtime_config
from .narration.events import NarrationEvent
from .orchestrator import SessionOrchestrator


@dataclass
class SmokeResult:
    name: str
    ok: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "details": self.details,
        }


class _FakePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logger = logging.getLogger("mahjong-companion-smoke")
        self.statuses: list[dict[str, object]] = []
        self.messages: list[dict[str, object]] = []

    def data_path(self, *parts: str) -> Path:
        path = self.root / "data"
        if parts:
            path = path.joinpath(*parts)
        return path

    def report_status(self, payload: dict[str, object]) -> None:
        self.statuses.append(dict(payload))

    def push_message(self, **kwargs: object) -> dict[str, object]:
        self.messages.append(dict(kwargs))
        return {"ok": True}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sample_frame(name: str) -> Path:
    return _repo_root() / "plugin" / "plugins" / "mahjong_companion" / "data" / "debug_samples" / name


async def run_v1_to_v9_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mahjong-companion-smoke-") as temp_dir:
        root = Path(temp_dir)
        plugin = _FakePlugin(root)
        orchestrator = SessionOrchestrator(plugin)
        orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

        results = [
            await _run_runtime_mode_case(orchestrator, plugin),
            await _run_real_sample_sanity(orchestrator, plugin),
            await _run_tile_efficiency_case(orchestrator, plugin),
        ]

        ok = all(result.ok for result in results)
        session_cache = plugin.data_path("session_cache")
        cache_files = sorted(path.name for path in session_cache.glob("*")) if session_cache.exists() else []
        return {
            "ok": ok,
            "results": [result.to_dict() for result in results],
            "message_count": len(plugin.messages),
            "status_count": len(plugin.statuses),
            "session_cache_files": cache_files,
            "last_status": plugin.statuses[-1] if plugin.statuses else {},
        }


async def _run_real_sample_sanity(orchestrator: SessionOrchestrator, plugin: _FakePlugin) -> SmokeResult:
    source_frame = _sample_frame("20260415-071314-863534-frame.png")
    debug_dir = plugin.data_path("debug_samples")
    debug_dir.mkdir(parents=True, exist_ok=True)
    copied_frame = debug_dir / source_frame.name
    shutil.copy2(source_frame, copied_frame)

    perception = await orchestrator.analyze_frame_path(str(copied_frame))
    decision = await orchestrator.generate_decision()
    narration = await orchestrator.generate_narration()
    pipeline = await orchestrator.run_companion_pipeline(
        frame_path=str(copied_frame),
        dispatch=True,
        force_reply=True,
    )

    perception_value = perception.value
    decision_value = decision.value
    narration_value = narration.value
    pipeline_value = pipeline.value
    ok = all([
        perception_value.get("ok"),
        decision_value.get("ok"),
        narration_value.get("ok"),
        pipeline_value.get("ok"),
        pipeline_value.get("dispatch", {}).get("ok"),
    ])
    return SmokeResult(
        name="v1_to_v4_real_sample",
        ok=ok,
        details={
            "frame": copied_frame.name,
            "scene": perception_value.get("scene"),
            "decision_type": decision_value.get("decision_type"),
            "narration_type": narration_value.get("event_type"),
            "dispatch_delivery": pipeline_value.get("dispatch", {}).get("delivery"),
        },
    )


async def _run_runtime_mode_case(orchestrator: SessionOrchestrator, plugin: _FakePlugin) -> SmokeResult:
    orchestrator.state.running = True
    orchestrator.state.window_bound = True
    orchestrator.state.window_title = "Mahjong Soul"
    orchestrator.state.runtime_mode = "active"
    orchestrator.state.scene = "replay"

    event = NarrationEvent(
        event_type="action_available",
        channel="nudge",
        delivery="proactive_notification",
        priority=75,
        summary="可行动作",
        detail="这是运行时队列 smoke 检查。",
        risk_level="medium",
        scene="replay",
        buttons=["next"],
        text="我把高价值提醒先放队列，再发给猫娘。",
        speakable=False,
        dedupe_key="smoke-runtime-contract",
    )
    dispatch_payload = orchestrator._dispatch_narration_locked(event)

    standby = await orchestrator.set_runtime_mode("standby")
    orchestrator._run_runtime_cycle_locked()
    standby_status = orchestrator.get_status()
    active = await orchestrator.set_runtime_mode("active")
    off = await orchestrator.set_runtime_mode("off")

    standby_value = standby.value
    active_value = active.value
    off_value = off.value
    ok = all([
        dispatch_payload.get("ok"),
        len(plugin.messages) >= 1,
        standby_value.get("ok"),
        standby_status.get("runtime_status") == "standby",
        active_value.get("ok"),
        active_value.get("runtime_mode") == "active",
        off_value.get("ok"),
        off_value.get("runtime_mode") == "off",
    ])
    return SmokeResult(
        name="runtime_modes_and_dispatch",
        ok=ok,
        details={
            "dispatch_delivery": dispatch_payload.get("delivery", ""),
            "standby_status": standby_status.get("runtime_status"),
            "message_count": len(plugin.messages),
        },
    )


async def _run_tile_efficiency_case(orchestrator: SessionOrchestrator, _plugin: _FakePlugin) -> SmokeResult:
    orchestrator.state.running = True
    orchestrator.state.status = "scanning"
    orchestrator.state.scene = "in_match"
    orchestrator.state.last_perception_ok = True
    orchestrator.state.last_perception = {
        "scene": "in_match",
        "confidence": 0.84,
        "is_user_turn": True,
        "buttons": [],
        "notes": ["structured hand sample injected"],
        "roi_hits": {"bottom_hand_area": True},
        "hand_tiles": ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "9m", "5z"],
        "melds": [],
        "dora_indicators": ["4p"],
        "riichi_players": ["right"],
        "raw_detections": [],
        "analysis_hints": {},
    }

    decision = await orchestrator.generate_decision()

    decision_value = decision.value
    analysis = decision_value.get("mahjong_analysis", {})
    ok = all([
        decision_value.get("ok"),
        decision_value.get("decision_type") == "tile_efficiency_hint",
        decision_value.get("recommended_focus") == "tile_efficiency",
        analysis.get("tile_level_available") is True,
        analysis.get("shanten_estimate") is not None,
        isinstance(analysis.get("candidate_discards"), list) and bool(analysis.get("candidate_discards")),
    ])
    return SmokeResult(
        name="tile_efficiency_hint",
        ok=ok,
        details={
            "decision_type": decision_value.get("decision_type"),
            "analysis_confidence": analysis.get("analysis_confidence"),
            "shanten_estimate": analysis.get("shanten_estimate"),
            "ukeire_estimate": analysis.get("ukeire_estimate"),
            "defense_alerts": analysis.get("defense_alerts", []),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mahjong Companion V1-V9 smoke validation.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    payload = asyncio.run(run_v1_to_v9_smoke())
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
