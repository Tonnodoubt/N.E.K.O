from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugin.plugins.mahjong_coach import MahjongCoachPlugin
from plugin.plugins.mahjong_coach import coach as coach_module
from plugin.plugins.mahjong_coach.coach import RoundCoachEngine, build_round_plan
from plugin.plugins.mahjong_coach.models import LiveSessionState, MahjongCoachConfig
from plugin.plugins.mahjong_coach.overlay import _overlay_geometry, overlay_text_from_payload
from plugin.plugins.mahjong_coach.perception.fast_hand_path import FastHandResult
from plugin.plugins.mahjong_coach.perception.meld_state import MeldStateResult
from plugin.plugins.mahjong_coach.perception.river_state import RiverStateResult
from plugin.plugins.mahjong_coach.tile_labels import hand_signature


HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]


def test_opening_scan_ignores_impossible_buttons_and_skips_river(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_resolve_buttons",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("opening scan should not inspect buttons")),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: (_ for _ in ()).throw(AssertionError("opening scan should not run river recognition")),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["ron", "riichi"])

    assert decision.decision_type == "opening_plan"
    assert decision.perception["action"]["source"] == "opening_hand_scan"
    assert decision.perception["river"]["reason"] == "opening_skips_river_scan"
    assert decision.coach_state["round_phase"] == "opening_strategy"


def test_fast_style_after_opening_uses_open_hand_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    engine.state.opening_emitted = True
    seen: dict[str, int] = {}

    def fake_detect(
        _path: Path,
        *,
        calibration_dir=None,
        min_hand_tiles: int = 12,
        max_hand_tiles: int = 14,
        use_onnx_hand: bool | None = None,
    ) -> FastHandResult:
        seen["min_hand_tiles"] = min_hand_tiles
        seen["max_hand_tiles"] = max_hand_tiles
        return FastHandResult(ok=True, hand_tiles=["1m", "2m", "3m", "4p"], confidence=0.77, reason="test_open_hand")

    monkeypatch.setattr("plugin.plugins.mahjong_coach.coach.detect_fast_hand_path", fake_detect)

    result = engine._detect_hand(Path("frame.png"))

    assert result.ok is True
    assert seen["min_hand_tiles"] == 4
    assert seen["max_hand_tiles"] == 14


def test_fast_style_late_open_hand_can_track_two_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    engine.state.opening_emitted = True
    engine.state.last_hand_tiles = ["1m", "2m", "3m", "4p", "5p"]
    seen: dict[str, int] = {}

    def fake_detect(
        _path: Path,
        *,
        calibration_dir=None,
        min_hand_tiles: int = 12,
        max_hand_tiles: int = 14,
        use_onnx_hand: bool | None = None,
    ) -> FastHandResult:
        seen["min_hand_tiles"] = min_hand_tiles
        return FastHandResult(ok=True, hand_tiles=["1m", "2m"], confidence=0.64, reason="test_late_open_hand")

    monkeypatch.setattr("plugin.plugins.mahjong_coach.coach.detect_fast_hand_path", fake_detect)

    result = engine._detect_hand(Path("frame.png"))

    assert result.ok is True
    assert seen["min_hand_tiles"] == 2


def test_fast_style_opening_still_requires_full_starting_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    seen: dict[str, int] = {}

    def fake_detect(
        _path: Path,
        *,
        calibration_dir=None,
        min_hand_tiles: int = 12,
        max_hand_tiles: int = 14,
        use_onnx_hand: bool | None = None,
    ) -> FastHandResult:
        seen["min_hand_tiles"] = min_hand_tiles
        return FastHandResult(reason="unstable_hand_count")

    monkeypatch.setattr("plugin.plugins.mahjong_coach.coach.detect_fast_hand_path", fake_detect)

    result = engine._detect_hand(Path("frame.png"))

    assert result.ok is False
    assert seen["min_hand_tiles"] == 12


def test_win_window_interrupts_before_hand_scan_after_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True

    def fail_if_called(_path: Path | None) -> FastHandResult:
        raise AssertionError("hand scan should not run for win windows")

    monkeypatch.setattr(engine, "_detect_hand", fail_if_called)

    decision = engine.analyze_frame(observed_buttons=["ron"])

    assert decision.decision_type == "win_window"
    assert decision.action_required is True
    assert decision.buttons == ["ron"]
    assert decision.reason_codes == ["critical_action_interrupt"]


def test_call_window_uses_hand_plan_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    engine.state.current_plan = "主线：断幺速度"
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: (_ for _ in ()).throw(AssertionError("call windows should not wait for river scan")),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["pon"])

    assert decision.decision_type == "call_window"
    assert decision.action_required is True
    assert decision.buttons == ["pon"]
    assert "默认跳过" in decision.suggestion
    assert decision.hand_tiles == HAND


def test_opening_plan_once_then_checkpoint_every_three_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(coach_checkpoint_self_turns=3))
    river_calls: list[Path | None] = []

    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda path: river_calls.append(path) or RiverStateResult(reason="test_river"),
    )

    first = engine.analyze_frame("frame.png", self_turn_index=1)
    second = engine.analyze_frame("frame.png", self_turn_index=2)
    third = engine.analyze_frame("frame.png", self_turn_index=3)

    assert first.decision_type == "opening_plan"
    assert second.decision_type == "observe"
    assert "discard" not in second.suggestion.lower()
    assert third.decision_type == "coach_checkpoint"
    assert third.reason_codes == ["scheduled_checkpoint"]
    assert len(river_calls) == 1


def test_force_checkpoint_works_without_per_turn_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(per_turn_discard_prompt=False))
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )

    decision = engine.analyze_frame("frame.png", force_checkpoint=True)

    assert decision.decision_type == "coach_checkpoint"
    assert decision.reason_codes == ["forced_checkpoint"]
    assert decision.engine_meta["per_turn_discard_prompt"] is False


def test_riichi_window_uses_local_fast_path_without_river(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: (_ for _ in ()).throw(AssertionError("riichi window should not wait for river scan")),
    )

    decision = engine.analyze_frame("frame.png", observed_buttons=["riichi"])

    assert decision.decision_type == "riichi_window"
    assert decision.action_required is True
    assert "本地暂未算出" in decision.suggestion or "推荐立直" in decision.suggestion


def test_riichi_window_recommends_good_wait() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]

    decision = engine._critical_decision(
        ["riichi"],
        {"source": "test"},
        0.0,
        hand_result=FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )

    assert decision.decision_type == "riichi_window"
    assert "推荐立直" in decision.suggestion
    assert "听5索、8索" in decision.suggestion
    assert "好形" in decision.suggestion


def test_riichi_window_warns_on_poor_wait() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    hand = ["1m", "1m", "1m", "2p", "3p", "4p", "5p", "6p", "7p", "2s", "3s", "4s", "7s", "9s"]

    decision = engine._critical_decision(
        ["riichi"],
        {"source": "test"},
        0.0,
        hand_result=FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )

    assert decision.decision_type == "riichi_window"
    assert "谨慎立直" in decision.suggestion
    assert "听8索" in decision.suggestion
    assert "愚形" in decision.suggestion


def test_round_plan_gives_concrete_keep_and_cleanup_guidance() -> None:
    plan = build_round_plan(["2m", "7m", "0p", "8p", "1s", "2s", "2s", "3s", "3s", "4s", "3z", "4z", "5z"])

    assert "索" in plan["summary"]
    assert any("保留" in item and "2索" in item for item in plan["targets"])
    assert any("路线选择" in item and "打" in item for item in plan["cautions"])
    assert any("优先清理" in item and "西" in item for item in plan["cautions"])
    assert "鸣牌" in plan["cautions"][-1]


def test_round_plan_names_honor_cleanup_route() -> None:
    plan = build_round_plan(["1m", "4m", "8m", "2p", "5p", "7p", "3s", "6s", "1z", "2z", "5z", "6z", "7z"])

    assert any("路线选择" in item and "孤字先打" in item for item in plan["cautions"])
    assert any(name in " ".join(plan["cautions"]) for name in ["东", "南", "白", "发", "中"])


def test_round_plan_keeps_dora_out_of_discard_priority() -> None:
    plan = build_round_plan(
        ["2m", "7m", "0p", "8p", "1s", "2s", "2s", "3s", "3s", "4s", "3z", "4z", "5z"],
        MahjongCoachConfig(dora_tiles=["7m"]),
    )

    assert "7m" not in plan["discard_priority"]
    assert any("宝牌/红5" in item and "7万" in item for item in plan["targets"])


def test_call_window_opens_value_honor_pair() -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "5z", "5z", "7m"]

    decision = engine._critical_decision(
        ["pon"],
        {"source": "test"},
        0.0,
        hand_result=FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )

    assert decision.decision_type == "call_window"
    assert "役牌对子白" in decision.suggestion
    assert "可以开" in decision.suggestion


def test_round_plan_includes_local_shanten_and_ukeire() -> None:
    plan = build_round_plan(["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "8s", "1z", "5z"])

    efficiency = plan["efficiency"]
    best_discard = efficiency["discard_options"][0]
    assert efficiency["best_path"] == "standard"
    assert efficiency["current_shanten"] == 1
    assert best_discard["tile"] in {"1z", "5z"}
    assert best_discard["effective_types"] > 0
    assert best_discard["effective_count"] > 0
    assert any("牌效" in item and "有效" in item for item in plan["cautions"])
    assert plan["discard_priority"][0] == best_discard["tile"]


def test_round_plan_subtracts_visible_river_from_ukeire() -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "8s", "1z", "5z"]
    plain = build_round_plan(hand)
    adjusted = build_round_plan(hand, visible_tiles=["1s", "2s", "4s", "5s", "7s", "8s"])

    plain_best = plain["efficiency"]["discard_options"][0]
    adjusted_best = adjusted["efficiency"]["discard_options"][0]
    assert plain_best["tile"] == adjusted_best["tile"]
    assert adjusted_best["effective_count"] < plain_best["effective_count"]
    assert adjusted["efficiency"]["visible_tile_count"] == 6
    assert any("已扣可见牌" in item for item in adjusted["cautions"])


def test_fast_open_hand_plan_focuses_on_closing_not_starting_route() -> None:
    plan = build_round_plan(
        ["4s", "5s", "6s", "2p", "3p", "7p", "5z", "5z"],
        MahjongCoachConfig(play_style="fast"),
    )

    assert plan["efficiency"]["open_melds"] == 2
    assert plan["direction"].startswith("副露")
    assert plan["direction"] != "役牌速攻"
    assert any("已副露2组" in item for item in plan["targets"])
    assert any("副露牌效" in item for item in plan["cautions"])


def test_fast_late_open_hand_shanten_counts_existing_melds() -> None:
    plan = build_round_plan(
        ["2m", "3m", "5p", "5p", "7s"],
        MahjongCoachConfig(play_style="fast"),
    )

    assert plan["efficiency"]["open_melds"] == 3
    assert plan["efficiency"]["current_shanten"] <= 1
    assert plan["direction"].startswith("副露")


def test_round_plan_uses_detected_meld_tiles_for_open_yaku() -> None:
    plan = build_round_plan(
        ["2m", "3m", "4p", "5p", "6s", "7s", "8s", "9s"],
        MahjongCoachConfig(play_style="fast"),
        open_melds=2,
        meld_tiles=["5z", "5z", "5z", "2s", "3s", "4s"],
    )

    assert plan["efficiency"]["open_melds"] == 2
    assert plan["direction"].startswith("副露")
    assert any("副露识别" in item and "白" in item for item in plan["targets"])
    assert any("役牌副露" in item and "白" in item for item in plan["targets"])
    assert not any("可能没役" in item for item in plan["cautions"])


def test_checkpoint_plan_uses_onnx_meld_state(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(play_style="fast"))
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(
            ok=True,
            hand_tiles=["2m", "3m", "4p", "5p", "6s", "7s", "8s", "9s"],
            confidence=0.86,
            reason="test_open_hand",
        ),
    )
    monkeypatch.setattr(
        engine,
        "_detect_melds",
        lambda _path, **_kwargs: MeldStateResult(
            ok=True,
            open_meld_count=2,
            melds=[{"player": "self", "meld_index": 1, "tiles": [{"tile": "5z"}]}],
            tiles=["5z", "5z", "5z", "2s", "3s", "4s"],
            confidence=0.93,
            reason="recognized_self_melds",
        ),
    )
    monkeypatch.setattr(engine, "_resolve_buttons", lambda *_args, **_kwargs: ([], {"source": "test"}))
    monkeypatch.setattr(engine, "_detect_river", lambda _path: RiverStateResult(ok=True, reason="test_river"))

    decision = engine.analyze_frame("frame.png", force_checkpoint=True)

    assert decision.decision_type == "coach_checkpoint"
    assert decision.perception["meld"]["open_meld_count"] == 2
    assert decision.coach_state["last_open_meld_count"] == 2
    assert any("役牌副露" in item for item in decision.coach_state["target_shapes"])


def test_round_plan_prefers_seven_pairs_when_pairs_are_dense() -> None:
    plan = build_round_plan(["1m", "1m", "2p", "2p", "3s", "3s", "4m", "4m", "5p", "5p", "6s", "7s", "8s", "9s"])

    assert plan["direction"] == "七对子"
    assert plan["efficiency"]["best_path"] == "seven_pairs"
    assert any("七对子胚子" in item for item in plan["targets"])
    assert any(item == "保留：1万、4万、2筒、5筒、3索" for item in plan["targets"])


def test_observe_explains_missing_stable_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(
            ok=False,
            reason="unstable_hand_count",
            raw_detections=[{"occupied": True}, {"occupied": True}],
        ),
    )

    decision = engine.analyze_frame()

    assert decision.decision_type == "observe"
    assert "No stable hand tiles" in decision.detail
    assert "hand_unstable_hand_count" in decision.reason_codes


def test_riichi_players_trigger_defense_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )

    decision = engine.analyze_frame("frame.png", riichi_players=["shimocha"])

    assert decision.decision_type == "defense_alert"
    assert decision.action_required is True
    assert decision.coach_state["attack_defense_bias"] == "defense"


def test_opponent_riichi_detection_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig(opponent_riichi_recognition_enabled=False))
    engine.state.opening_emitted = True
    engine.state.update_count = 2
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        coach_module,
        "detect_riichi_sticks",
        lambda _path: (_ for _ in ()).throw(AssertionError("opponent riichi detector should be opt-in")),
    )

    decision = engine.analyze_frame("frame.png")

    assert decision.decision_type != "defense_alert"
    assert decision.engine_meta["opponent_riichi_recognition_enabled"] is False


def test_opponent_riichi_detection_ignores_existing_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    monkeypatch.setattr(
        coach_module,
        "detect_riichi_sticks",
        lambda _path: SimpleNamespace(riichi_players=["unknown"], stick_count=1),
    )

    assert engine._detect_riichi_players(Path("frame.png")) == []
    assert engine.state.riichi_stick_baseline == 1


def test_opponent_riichi_detection_uses_counter_increase(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.riichi_stick_baseline = 0
    engine.state.riichi_pending = {"unknown": 1}
    monkeypatch.setattr(
        coach_module,
        "detect_riichi_sticks",
        lambda _path: SimpleNamespace(riichi_players=["unknown"], stick_count=1),
    )

    assert engine._detect_riichi_players(Path("frame.png")) == ["unknown"]


def test_riichi_defense_uses_recognized_riichi_player_river(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=list(HAND), confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            discard_piles={
                "right_opponent": [
                    {"tile": "7p", "confidence": 0.96},
                    {"tile": "2s", "confidence": 0.95},
                ]
            },
            visible_tiles=["7p", "2s"],
            confidence=0.955,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", riichi_players=["shimocha"])

    assert "现物" in decision.suggestion
    assert "2索" in decision.suggestion
    assert "7筒" not in decision.suggestion
    assert decision.coach_state["last_visible_discards"] == ["7p", "2s"]


def test_riichi_defense_only_recommends_safe_tiles_player_actually_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "7s", "1z", "1z"]
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            discard_piles={
                "right_opponent": [
                    {"tile": "3z", "confidence": 0.96},
                    {"tile": "4z", "confidence": 0.95},
                ]
            },
            visible_tiles=["3z", "4z"],
            confidence=0.955,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", riichi_players=["shimocha"])

    assert decision.decision_type == "defense_alert"
    assert "现物" in decision.suggestion
    assert "手里没有" in decision.suggestion


def test_checkpoint_plan_uses_recognized_river_for_ukeire(monkeypatch: pytest.MonkeyPatch) -> None:
    hand = ["1m", "2m", "3m", "4p", "5p", "6p", "2s", "3s", "4s", "5s", "6s", "8s", "1z", "5z"]
    engine = RoundCoachEngine(MahjongCoachConfig())
    engine.state.opening_emitted = True
    monkeypatch.setattr(
        engine,
        "_detect_hand",
        lambda _path: FastHandResult(ok=True, hand_tiles=hand, confidence=0.91, reason="test_hand"),
    )
    monkeypatch.setattr(
        engine,
        "_detect_river",
        lambda _path: RiverStateResult(
            ok=True,
            visible_tiles=["1s", "2s", "4s", "5s", "7s", "8s"],
            confidence=0.95,
            reason="test_river",
        ),
    )

    decision = engine.analyze_frame("frame.png", self_turn_index=3)

    assert decision.decision_type == "coach_checkpoint"
    assert "已扣可见牌" in decision.detail
    assert any("已扣可见牌" in item for item in decision.coach_state["caution_points"])


def test_live_config_from_payload() -> None:
    cfg = MahjongCoachConfig.from_payload(
        {
            "live": {
                "window_keywords": ["Mahjong Soul"],
                "interval_ms": 900,
                "fast_interval_ms": 180,
                "keep_frames": 12,
                "checkpoint_interval_seconds": 14,
                "overlay_enabled": False,
                "save_format": "jpg",
            }
        }
    )

    assert cfg.live_window_keywords == ["Mahjong Soul"]
    assert cfg.live_interval_ms == 900
    assert cfg.live_fast_interval_ms == 180
    assert cfg.live_keep_frames == 12
    assert cfg.live_checkpoint_interval_seconds == 14
    assert cfg.live_overlay_enabled is False
    assert cfg.live_save_format == "jpg"


def test_live_config_defaults_keep_training_material() -> None:
    cfg = MahjongCoachConfig.from_payload({})

    assert cfg.live_interval_ms == 400
    assert cfg.live_keep_frames == 1000
    assert cfg.live_save_format == "jpg"


def test_show_overlay_restarts_config_window() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay

    assert plugin._show_overlay(strategy=False) is True
    assert overlay.calls == ["start", "config"]


@pytest.mark.asyncio
async def test_show_overlay_entry_uses_strategy_when_live_running() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay
    plugin._live_task = SimpleNamespace(done=lambda: False)
    plugin._live_state = LiveSessionState(running=True)

    result = await plugin.mahjong_coach_show_overlay()

    assert result.unwrap()["running"] is True
    assert overlay.calls == ["start", "strategy"]


@pytest.mark.asyncio
async def test_start_live_reopens_overlay_when_already_running() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    overlay = _FakeOverlay()
    plugin._overlay = overlay
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._cfg = MahjongCoachConfig()
    plugin._live_task = SimpleNamespace(done=lambda: False)
    plugin._live_state = LiveSessionState(running=True)
    plugin.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)

    result = await plugin._overlay_start_live(overlay=True)

    assert result.unwrap()["status"] == "already_running"
    assert overlay.calls == ["start", "strategy"]


def test_river_config_from_payload() -> None:
    cfg = MahjongCoachConfig.from_payload(
        {
            "perception": {
                "river_recognition_enabled": False,
                "river_min_confidence": 0.72,
            }
        }
    )

    assert cfg.river_recognition_enabled is False
    assert cfg.river_min_confidence == 0.72


def test_overlay_text_prioritizes_action_required() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {
                "decision_type": "call_window",
                "action_required": True,
                "summary": "Call window detected",
                "suggestion": "Evaluate chi/pon/kan quickly.",
            },
            "round_state": {"current_plan": "Play inside hand"},
        }
    )

    assert "本地鸣牌" in text
    assert "Evaluate chi/pon/kan quickly" in text


def test_overlay_text_shows_riichi_fast_judgement() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {
                "decision_type": "riichi_window",
                "action_required": True,
                "summary": "立直窗口",
                "suggestion": "推荐立直：打7索听5索、8索，有效2种8枚；好形/枚数够；本地快判。",
            },
            "round_state": {"current_plan": "Play inside hand"},
        }
    )

    assert "本地立直" in text
    assert "推荐立直：打7索听5索、8索" in text
    assert "好形/枚数够" in text


def test_overlay_geometry_defaults_near_self_hand() -> None:
    x, y = _overlay_geometry(1920, 1080, 630, 138)

    assert 600 <= x <= 700
    assert 680 <= y <= 780


def test_overlay_text_uses_strategy_when_no_action() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "current_plan": "主线：围绕索子 122334 推进",
                "attack_defense_bias": "attack",
                "last_update_reason": "opening_plan",
                "target_shapes": ["主线：围绕索子 122334 推进"],
                "caution_points": ["路线选择：主线打西；保守打7万", "优先清理：2万、7万、西、北"],
            },
        }
    )

    assert "本地" in text
    assert "方向：围绕索子 122334 推进" in text
    assert "…" not in text


def test_overlay_text_shows_detailed_open_hand_lines() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "local_direction": "副露一向听",
                "local_plan": "已副露2组，优先打向听最低、有效牌最多的牌；留6索、7索、8索、9索，先看打6索，当前1向听",
                "target_shapes": ["已成役：役牌副露 白", "保留：6索、7索、8索、9索"],
                "caution_points": [
                    "副露收束：主线打6索；不硬染打2万",
                    "副露牌效：估算2组，当前1向听；打6索后1向听，有效8种28枚，已扣可见牌（1万、2万）。",
                    "鸣牌：继续快攻，但只开能进听或明显增加有效牌的牌。",
                ],
            },
        }
    )

    assert "方向：副露一向听" in text
    assert "役：役牌副露 白" in text
    assert "..." not in text


def test_overlay_text_shows_round_idle_without_old_plan() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {"decision_type": "round_idle", "summary": "等待下一局"},
            "round_state": {"current_plan": "上一局旧主线"},
        }
    )

    assert "等待下一局" in text
    assert "上一局已结束" in text
    assert "上一局旧主线" not in text


def test_live_round_idle_resets_old_plan_after_missing_hand_streak() -> None:
    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._engine = RoundCoachEngine(MahjongCoachConfig())
    plugin._engine.state.opening_emitted = True
    plugin._engine.state.current_plan = "上一局旧主线"
    plugin._live_missing_hand_frames = 0
    plugin._live_state = LiveSessionState(running=True)
    plugin._live_last_hand_signature = "old"
    plugin._live_last_checkpoint_at = 10.0
    plugin._last_decision = {}
    decision = SimpleNamespace(
        action_required=False,
        hand_tiles=[],
        reason_codes=["hand_unstable_hand_count"],
        perception={"hand": {"ok": False}},
    )

    assert plugin._maybe_reset_live_round_idle(decision) is False
    assert plugin._maybe_reset_live_round_idle(decision) is False
    assert plugin._maybe_reset_live_round_idle(decision) is False
    assert plugin._maybe_reset_live_round_idle(decision) is True
    assert plugin._engine.state.current_plan == ""
    assert plugin._live_state.observed_hand_changes == 0
    assert plugin._last_decision["decision_type"] == "round_idle"


class _FakeOverlay:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> bool:
        self.calls.append("start")
        return True

    def show_config(self) -> None:
        self.calls.append("config")

    def show_strategy(self) -> None:
        self.calls.append("strategy")
