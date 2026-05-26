from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugin.plugins.mahjong_coach import MahjongCoachPlugin
from plugin.plugins.mahjong_coach.coach import RoundCoachEngine, build_round_plan
from plugin.plugins.mahjong_coach.decision_coordinator import DecisionCoordinator
from plugin.plugins.mahjong_coach.models import LiveSessionState, MahjongCoachConfig
from plugin.plugins.mahjong_coach.overlay import _overlay_geometry, overlay_text_from_payload
from plugin.plugins.mahjong_coach.perception.fast_hand_path import FastHandResult
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
    assert "不等待 LLM" in decision.suggestion


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


def test_round_plan_prefers_seven_pairs_when_pairs_are_dense() -> None:
    plan = build_round_plan(["1m", "1m", "2p", "2p", "3s", "3s", "4m", "4m", "5p", "5p", "6s", "7s", "8s", "9s"])

    assert "七对子" in plan["summary"]
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
    assert "7筒" in decision.suggestion
    assert "2索" in decision.suggestion
    assert decision.coach_state["last_visible_discards"] == ["7p", "2s"]


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


def test_live_config_defaults_are_disk_friendly() -> None:
    cfg = MahjongCoachConfig.from_payload({})

    assert cfg.live_keep_frames == 30
    assert cfg.live_save_format == "jpg"


def test_llm_config_from_payload() -> None:
    cfg = MahjongCoachConfig.from_payload(
        {
            "llm": {
                "enabled": False,
                "timeout": 4.5,
                "opening_enabled": False,
                "checkpoint_enabled": True,
            }
        }
    )

    assert cfg.llm_enabled is False
    assert cfg.llm_timeout == 4.5
    assert cfg.llm_opening_enabled is False
    assert cfg.llm_checkpoint_enabled is True


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
    assert "默认跳过" in text


def test_overlay_text_shows_riichi_fast_judgement() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {
                "decision_type": "riichi_window",
                "action_required": True,
                "summary": "立直窗口",
                "suggestion": "推荐立直：打7索听5索、8索，有效2种8枚；好形/枚数够；本地快判，不等待 LLM。",
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
    assert "AI" in text
    assert "方向：围绕索子 122334 推进" in text
    assert "打：2万、7万、西等4张" in text
    assert "…" not in text


def test_overlay_text_marks_ai_enhanced_decision() -> None:
    text = overlay_text_from_payload(
        {
            "last_decision": {"decision_type": "observe"},
            "round_state": {
                "plan_source": "llm",
                "local_plan": "主线：牌效推进，先打东。",
                "ai_plan": "主线：筒子占比很高，保留同色块，先清西。",
                "ai_cautions": ["优先清理：西"],
            },
        }
    )

    assert "本地" in text
    assert "AI" in text
    assert "筒子多" in text
    assert "留：同色" in text
    assert "打：西" in text
    assert "…" not in text


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


def test_decision_coordinator_only_enhances_opening_and_checkpoint() -> None:
    coordinator = DecisionCoordinator()
    cfg = MahjongCoachConfig()

    assert coordinator.should_enhance_with_llm(SimpleNamespace(decision_type="opening_plan", action_required=False, hand_tiles=list(HAND)), cfg) is True
    assert coordinator.should_enhance_with_llm(SimpleNamespace(decision_type="coach_checkpoint", action_required=False, hand_tiles=list(HAND)), cfg) is True
    assert coordinator.should_enhance_with_llm(SimpleNamespace(decision_type="call_window", action_required=True, hand_tiles=list(HAND)), cfg) is False
    assert coordinator.should_enhance_with_llm(SimpleNamespace(decision_type="defense_alert", action_required=True, hand_tiles=list(HAND)), cfg) is False


@pytest.mark.asyncio
async def test_manual_llm_enhancement_updates_opening_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import plugin.plugins.mahjong_coach as mahjong_plugin
    from plugin.plugins.mahjong_coach.models import CoachDecision

    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig(llm_timeout=1.0)
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine.state.last_hand_signature = hand_signature(HAND)
    plugin._engine.state.last_hand_tiles = list(HAND)
    plugin._engine.state.update_count = 1
    plugin._live_state = LiveSessionState(observed_hand_changes=1)
    plugin._decision_coordinator = DecisionCoordinator()
    plugin._last_decision = {}
    seen_kwargs: dict[str, object] = {}

    async def fake_build_round_plan_llm(*_args, **_kwargs):
        seen_kwargs.update(_kwargs)
        return {
            "summary": "AI主线：索子速度",
            "detail": "继续保留连续索子。",
            "bias": "attack",
            "targets": ["保留：234索"],
            "cautions": ["吃碰杠：默认跳过"],
            "discard_priority": ["孤字"],
        }

    monkeypatch.setattr(mahjong_plugin, "build_round_plan_llm", fake_build_round_plan_llm)
    monkeypatch.setitem(MahjongCoachPlugin._enhance_decision_with_llm.__globals__, "build_round_plan_llm", fake_build_round_plan_llm)
    decision = CoachDecision(decision_type="opening_plan", suggestion="启发式主线", detail="旧细节", hand_tiles=list(HAND))

    enhanced = await plugin._enhance_decision_with_llm(decision)

    assert enhanced.analysis_source == "llm"
    assert plugin._engine.state.plan_source == "llm"
    assert plugin._engine.state.opening_plan == "AI主线：索子速度"
    assert plugin._engine.state.current_plan == "AI主线：索子速度"
    assert plugin._engine.state.target_shapes == ["保留：234索"]
    assert plugin._engine.state.local_plan == "启发式主线"
    assert plugin._engine.state.local_detail == "旧细节"
    assert plugin._engine.state.local_targets == []
    assert plugin._engine.state.local_cautions == []
    assert plugin._engine.state.ai_plan == "AI主线：索子速度"
    assert plugin._engine.state.ai_detail == "继续保留连续索子。"
    assert plugin._engine.state.ai_targets == ["保留：234索"]
    assert plugin._engine.state.ai_cautions == ["吃碰杠：默认跳过"]
    assert seen_kwargs["previous_plan"] == ""
    assert seen_kwargs["turn_number"] is None


def test_stale_llm_result_from_previous_round_does_not_update_state() -> None:
    from plugin.plugins.mahjong_coach.models import CoachDecision

    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine.state.last_hand_signature = hand_signature(HAND)
    plugin._engine.state.update_count = 1
    plugin._decision_coordinator = DecisionCoordinator()
    plugin._last_decision = {}
    decision = CoachDecision(decision_type="coach_checkpoint", suggestion="启发式", hand_tiles=list(HAND))
    token = plugin._decision_coordinator.build_enhancement_token(decision, plugin._engine.state, plugin._engine.state.last_hand_signature)
    plugin._engine.state.round_id = "next-round"

    result = plugin._apply_llm_enhancement(
        decision,
        token,
        {"summary": "启发式", "detail": "", "bias": "neutral", "targets": [], "cautions": [], "discard_priority": []},
        {"summary": "AI不该覆盖", "detail": "", "bias": "attack", "targets": [], "cautions": [], "discard_priority": []},
    )

    assert result is None
    assert plugin._engine.state.current_plan == ""


def test_live_llm_result_updates_ai_box_after_hand_signature_drift() -> None:
    from plugin.plugins.mahjong_coach.models import CoachDecision

    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine.state.last_hand_signature = hand_signature(HAND)
    plugin._engine.state.update_count = 1
    plugin._engine.state.current_plan = "本地新手牌"
    plugin._engine.state.local_plan = "本地新手牌"
    plugin._decision_coordinator = DecisionCoordinator()
    plugin._last_decision = {}
    decision = CoachDecision(decision_type="coach_checkpoint", suggestion="本地旧手牌", detail="旧细节", hand_tiles=list(HAND))
    token = plugin._decision_coordinator.build_enhancement_token(decision, plugin._engine.state, plugin._engine.state.last_hand_signature)
    plugin._engine.state.last_hand_signature = "new-hand"

    result = plugin._apply_llm_enhancement(
        decision,
        token,
        {"summary": "本地旧手牌", "detail": "旧细节", "bias": "neutral", "targets": [], "cautions": [], "discard_priority": []},
        {
            "summary": "AI旧手牌参考",
            "detail": "手牌变化后的迟到模型结果只进 AI 框。",
            "bias": "attack",
            "targets": ["AI目标"],
            "cautions": ["AI风险"],
            "discard_priority": [],
        },
    )

    assert result is not None
    assert plugin._engine.state.current_plan == "本地新手牌"
    assert plugin._engine.state.plan_source == "heuristic"
    assert plugin._engine.state.llm_status == "ready_previous_hand"
    assert plugin._engine.state.ai_plan == "AI旧手牌参考"


def test_live_llm_result_updates_ai_box_after_local_checkpoint_drift() -> None:
    from plugin.plugins.mahjong_coach.models import CoachDecision

    plugin = MahjongCoachPlugin.__new__(MahjongCoachPlugin)
    plugin._cfg = MahjongCoachConfig()
    plugin._engine = RoundCoachEngine(plugin._cfg)
    plugin._engine.state.last_hand_signature = hand_signature(HAND)
    plugin._engine.state.update_count = 1
    plugin._engine.state.current_plan = "本地新主线"
    plugin._engine.state.local_plan = "本地新主线"
    plugin._decision_coordinator = DecisionCoordinator()
    plugin._last_decision = {}
    decision = CoachDecision(decision_type="coach_checkpoint", suggestion="本地旧主线", detail="旧细节", hand_tiles=list(HAND))
    token = plugin._decision_coordinator.build_enhancement_token(decision, plugin._engine.state, plugin._engine.state.last_hand_signature)
    plugin._engine.state.update_count = 2

    result = plugin._apply_llm_enhancement(
        decision,
        token,
        {"summary": "本地旧主线", "detail": "旧细节", "bias": "neutral", "targets": [], "cautions": [], "discard_priority": []},
        {
            "summary": "AI延迟建议",
            "detail": "同一手牌的迟到模型结果仍可进入 AI 框。",
            "bias": "attack",
            "targets": ["AI目标"],
            "cautions": ["AI风险"],
            "discard_priority": [],
        },
    )

    assert result is not None
    assert plugin._engine.state.current_plan == "本地新主线"
    assert plugin._engine.state.plan_source == "heuristic"
    assert plugin._engine.state.ai_plan == "AI延迟建议"
    assert plugin._engine.state.ai_detail == "同一手牌的迟到模型结果仍可进入 AI 框。"
    assert plugin._engine.state.ai_targets == ["AI目标"]
    assert plugin._engine.state.ai_cautions == ["AI风险"]
