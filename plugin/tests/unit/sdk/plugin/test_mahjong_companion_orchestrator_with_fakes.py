from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from plugin.plugins.mahjong_companion.config_defaults import DEFAULT_CONFIG, merge_runtime_config
from plugin.plugins.mahjong_companion.contracts import DecisionResult, PerceivedGameState
from plugin.plugins.mahjong_companion.decision.preturn_planner import build_preturn_discard_plan
from plugin.plugins.mahjong_companion.narration.events import NarrationEvent
from plugin.plugins.mahjong_companion.narration.view_model import CompanionViewModel
from plugin.plugins.mahjong_companion.orchestrator import SessionOrchestrator


class _FakePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logger = logging.getLogger("mahjong-companion-fake-adapter-test")
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


class _FakePerceptionAdapter:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def analyze(self, image_path: Path) -> tuple[PerceivedGameState, dict[str, Any]]:
        self.calls.append(image_path)
        return (
            PerceivedGameState(
                scene="in_match",
                confidence=0.91,
                is_user_turn=True,
                buttons=["skip"],
                notes=["fake perception adapter"],
                roi_hits={"bottom_action_bar": True},
            ),
            {"adapter": "fake_perception"},
        )


class _SequencePerceptionAdapter:
    def __init__(self, states: list[PerceivedGameState]) -> None:
        self.states = list(states)
        self.calls: list[Path] = []

    def analyze(self, image_path: Path) -> tuple[PerceivedGameState, dict[str, Any]]:
        self.calls.append(image_path)
        if not self.states:
            raise AssertionError("no fake perception states left")
        return self.states.pop(0), {"adapter": "sequence_perception"}


class _FakeNarrationAdapter:
    def __init__(self) -> None:
        self.decisions: list[DecisionResult] = []

    def generate(self, decision: DecisionResult) -> tuple[NarrationEvent, CompanionViewModel, dict[str, Any]]:
        self.decisions.append(decision)
        event = NarrationEvent(
            event_type="fake_narration",
            channel="silent_ui",
            delivery="silent_ui",
            priority=decision.priority,
            summary="fake summary",
            detail="fake detail",
            risk_level=decision.risk_level,
            scene=decision.scene,
            buttons=list(decision.buttons),
            text="fake narration text",
            speakable=False,
            dedupe_key="fake-narration",
        )
        view_model = CompanionViewModel(
            headline="fake headline",
            subline="fake subline",
            mood="calm",
            suggestion_level="nudge",
            speakable=False,
            delivery="silent_ui",
            text=event.text,
        )
        return event, view_model, {"adapter": "fake_narration"}


@pytest.mark.asyncio
async def test_run_companion_pipeline_uses_injected_perception_and_narration_adapters(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    perception_adapter = _FakePerceptionAdapter()
    narration_adapter = _FakeNarrationAdapter()
    orchestrator = SessionOrchestrator(
        plugin,
        perception_adapter=perception_adapter,
        narration_adapter=narration_adapter,
    )
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

    fake_frame = plugin.data_path("debug_samples") / "not-a-real-image.png"
    fake_frame.parent.mkdir(parents=True, exist_ok=True)
    fake_frame.write_bytes(b"this is not a valid image")

    result = await orchestrator.run_companion_pipeline(
        frame_path=str(fake_frame),
        dispatch=False,
        force_reply=False,
    )

    assert result.value["ok"] is True
    assert perception_adapter.calls == [fake_frame]
    assert narration_adapter.decisions
    assert result.value["perception"]["notes"] == ["fake perception adapter"]
    assert result.value["narration"]["event_type"] == "fake_narration"
    assert result.value["narration"]["text"] == "fake narration text"


@pytest.mark.asyncio
async def test_preturn_discard_plan_is_reused_after_single_draw(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    waiting_hand = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7p",
        "8p", "9p", "2s", "3s", "4s", "5z",
    ]
    drawn_hand = [*waiting_hand, "9m"]
    perception_adapter = _SequencePerceptionAdapter(
        [
            PerceivedGameState(
                scene="in_match",
                confidence=0.91,
                is_user_turn=False,
                hand_tiles=waiting_hand,
                analysis_hints={
                    "recognized_hand_tile_count": 13,
                    "analysis_confidence": 0.86,
                    "tile_level_state": "tile_level_reliable",
                },
            ),
            PerceivedGameState(
                scene="in_match",
                confidence=0.92,
                is_user_turn=True,
                hand_tiles=drawn_hand,
                analysis_hints={
                    "recognized_hand_tile_count": 14,
                    "analysis_confidence": 0.86,
                    "tile_level_state": "tile_level_reliable",
                },
            ),
        ]
    )
    orchestrator = SessionOrchestrator(plugin, perception_adapter=perception_adapter)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))

    frame_one = plugin.data_path("debug_samples") / "preturn-1.png"
    frame_two = plugin.data_path("debug_samples") / "preturn-2.png"
    frame_one.parent.mkdir(parents=True, exist_ok=True)
    frame_one.write_bytes(b"fake frame one")
    frame_two.write_bytes(b"fake frame two")

    first = await orchestrator.analyze_frame_path(str(frame_one))
    assert first.value["ok"] is True
    assert orchestrator.get_status()["preturn_discard_plan"]["candidate_discards"]

    second = await orchestrator.run_companion_pipeline(
        frame_path=str(frame_two),
        dispatch=False,
        force_reply=False,
    )

    decision = second.value["decision"]
    assert second.value["ok"] is True
    assert decision["decision_type"] == "tile_efficiency_hint"
    assert decision["mahjong_analysis"]["candidate_discards"][0]["source"] in {"preturn_cached", "drawn_tile"}
    assert second.value["perception"]["analysis_hints"]["preturn_plan_applied"] is True
    assert orchestrator.get_status()["last_preturn_plan_meta"]["applied"] is True


@pytest.mark.parametrize(
    ("waiting_hand", "melds", "expected_draw_slot_index", "expected_slot_id"),
    [
        (
            [
                "1m", "2m", "3m", "4m", "5m", "6m", "7p",
                "8p", "9p", "2s", "3s", "4s", "5z",
            ],
            [],
            14,
            "hand_14",
        ),
        (
            ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "2s", "3s"],
            [["7z", "7z", "7z"]],
            11,
            "hand_11",
        ),
    ],
)
def test_fast_preturn_advice_uses_dynamic_draw_slot_before_full_perception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    waiting_hand: list[str],
    melds: list[list[str]],
    expected_draw_slot_index: int,
    expected_slot_id: str,
) -> None:
    plugin = _FakePlugin(tmp_path)
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    plan = build_preturn_discard_plan(
        PerceivedGameState(
            scene="in_match",
            confidence=0.91,
            is_user_turn=False,
            hand_tiles=waiting_hand,
            melds=melds,
            analysis_hints={
                "recognized_hand_tile_count": len(waiting_hand),
                "recognized_meld_group_count": len(melds),
                "analysis_confidence": 0.86,
                "tile_level_state": "tile_level_reliable",
            },
        )
    )
    assert plan is not None
    orchestrator._preturn_discard_plan = plan

    frame_path = plugin.data_path("debug_samples") / "fast-draw.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"fake frame")

    class _FakeFastResult:
        ok = True
        tile = "9m"
        confidence = 0.9
        reason = "matched"
        slot_id = expected_slot_id
        raw_detection = {"slot_id": expected_slot_id, "candidate_tile": "9m"}

        def to_dict(self) -> dict[str, object]:
            return {
                "ok": self.ok,
                "tile": self.tile,
                "confidence": self.confidence,
                "reason": self.reason,
                "slot_id": self.slot_id,
                "raw_detection": self.raw_detection,
            }

    fast_path_calls: list[dict[str, object]] = []

    def _fake_detect_drawn_tile_fast_path(*args: object, **kwargs: object) -> _FakeFastResult:
        fast_path_calls.append({"args": args, "kwargs": kwargs})
        return _FakeFastResult()

    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.fast_path.detect_drawn_tile_fast_path",
        _fake_detect_drawn_tile_fast_path,
    )

    emitted = orchestrator._maybe_emit_fast_preturn_advice_locked(frame_path)

    assert emitted is True
    assert fast_path_calls[0]["kwargs"]["draw_slot_index"] == expected_draw_slot_index
    assert orchestrator.state.last_decision_type == "tile_efficiency_hint"
    assert orchestrator.state.last_decision["mahjong_analysis"]["candidate_discards"]
    assert orchestrator.state.last_perception["analysis_hints"]["preturn_plan_applied"] is True
    assert (
        orchestrator.state.last_perception["analysis_hints"].get("recognized_meld_group_count", 0)
        == len(melds)
    )
    assert orchestrator.get_status()["last_preturn_plan_meta"]["fast_path_draw_slot_index"] == expected_draw_slot_index
    assert orchestrator.get_status()["last_preturn_plan_meta"]["fast_advice_emitted"] is True
