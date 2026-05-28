from __future__ import annotations

import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import CoachDecision, MahjongCoachConfig, RoundCoachState, _valid_play_style
from .perception.action_detector import detect_action_buttons_fast
from .perception.riichi_detector import detect_riichi_sticks
from .perception.fast_hand_path import FastHandResult, detect_fast_hand_path, quick_frame_fingerprint
from .perception.meld_state import MeldStateResult, detect_meld_state_path
from .perception.river_state import RiverStateResult, detect_river_state_path
from .tile_labels import hand_signature, is_honor, is_simple, is_terminal, normalize_tile, tile_rank, tile_suit


CRITICAL_BUTTONS = {"chi", "pon", "kan", "ron", "tsumo", "riichi"}
CALL_BUTTONS = {"chi", "pon", "kan"}
WIN_BUTTONS = {"ron"}
SUIT_NAMES = {"m": "万", "p": "筒", "s": "索", "z": "字"}
HONOR_NAMES = {"1z": "东", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "发", "7z": "中"}
TEXT_TILE_ALIASES = {"东": "1z", "南": "2z", "西": "3z", "北": "4z", "白": "5z", "发": "6z", "發": "6z", "中": "7z"}
TILE_TYPES = [f"{rank}{suit}" for suit in ("m", "p", "s") for rank in range(1, 10)] + [
    f"{rank}z" for rank in range(1, 8)
]
ORPHAN_TYPES = {"1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z"}

_RIICHI_SUIT_THRESHOLD = 7
_FAST_SUIT_THRESHOLD = 9
_RIICHI_PAIR_THRESHOLD = 4
_FAST_PAIR_THRESHOLD = 5
_RIICHI_SIMPLE_THRESHOLD = 10
_FAST_SIMPLE_THRESHOLD = 8


class RoundCoachEngine:
    def __init__(
        self,
        config: MahjongCoachConfig | None = None,
        *,
        calibration_dir: Path | None = None,
    ) -> None:
        self.config = config or MahjongCoachConfig()
        self.calibration_dir = calibration_dir
        self.state = RoundCoachState()
        self._last_fingerprints: dict[str, bytes] = {}

    def reset_round(self, round_id: str = "default") -> RoundCoachState:
        self.state = RoundCoachState(round_id=round_id or "default", play_style=self.config.play_style)
        self._last_fingerprints = {}
        return self.state

    def analyze_frame(
        self,
        image_path: str | Path | None = None,
        *,
        observed_buttons: list[str] | None = None,
        self_turn_index: int | None = None,
        riichi_players: list[str] | None = None,
        force_checkpoint: bool = False,
    ) -> CoachDecision:
        started = time.perf_counter()
        path = Path(image_path) if image_path else None
        riichi_players = [str(item) for item in (riichi_players or []) if str(item).strip()]

        if not self.state.opening_emitted:
            hand_result = self._detect_hand(path)
            meld_result = self._detect_melds(path, hand_result=hand_result)
            self._detect_riichi_players(path)
            if meld_result.ok:
                self._remember_melds(meld_result)
            if not hand_result.ok and meld_result.ok and _play_style(self.config) == "fast":
                hand_result = self._detect_hand(path, min_hand_tiles=2 if meld_result.open_meld_count >= 3 else 4)
            if hand_result.ok:
                self._remember_hand(hand_result)
            action_meta = {"source": "opening_hand_scan", "skipped": True}
            river_result = self._river_result_from_state("opening_skips_river_scan", ok_if_cached=False)
            if hand_result.ok:
                return self._opening_decision(
                    hand_result,
                    river_result,
                    started,
                    action_meta=action_meta,
                    meld_result=meld_result,
                )
            return self._observe_decision(
                hand_result,
                action_meta,
                river_result,
                started,
                phase="opening_hand_scan",
                meld_result=meld_result,
            )

        buttons, action_meta = self._resolve_buttons(path, observed_buttons)
        critical = [button for button in buttons if button in CRITICAL_BUTTONS]
        win_buttons = [button for button in critical if button in WIN_BUTTONS]
        if self.config.critical_action_interrupts and win_buttons:
            river_result = self._river_result_from_state("action_window_uses_cached_river")
            return self._critical_decision(win_buttons, action_meta, started, river_result=river_result)

        # Fingerprint gate: skip all detection if nothing changed
        fp = quick_frame_fingerprint(path, self._last_fingerprints) if path else None
        if fp is not None:
            self._last_fingerprints = fp["hashes"]
        if fp and not fp["action_changed"] and not fp["hand_changed"]:
            river_result = self._river_result_from_state("fingerprint_no_change")
            return self._observe_decision(
                FastHandResult(reason="fingerprint_match"),
                action_meta,
                river_result,
                started,
                phase="fingerprint_no_change",
            )

        hand_result, meld_result = self._detect_and_remember(path)

        call_buttons = [button for button in critical if button in CALL_BUTTONS]
        riichi_buttons = [button for button in critical if button == "riichi"]
        if self.config.critical_action_interrupts and (call_buttons or riichi_buttons):
            river_result = self._river_result_from_state("action_window_uses_cached_river")
            return self._critical_decision(
                [*call_buttons, *riichi_buttons],
                action_meta,
                started,
                hand_result=hand_result,
                meld_result=meld_result,
                river_result=river_result,
            )

        if riichi_players:
            self.state.riichi_players = list(riichi_players)
        elif self.state.opening_emitted and self.state.update_count >= 2:
            riichi_players = self._detect_riichi_players(path)
            if riichi_players:
                self.state.riichi_players = list(riichi_players)

        if riichi_players:
            river_result = self._detect_river(path)
            if river_result.ok:
                self._remember_river(river_result)
            return self._defense_decision(riichi_players, hand_result, river_result, started, meld_result=meld_result)

        turn_number = _coerce_turn(self_turn_index)
        if hand_result.ok and self._checkpoint_due(turn_number, force_checkpoint=force_checkpoint):
            river_result = self._detect_river(path)
            if river_result.ok:
                self._remember_river(river_result)
            return self._checkpoint_decision(hand_result, river_result, turn_number, force_checkpoint, started, meld_result=meld_result)

        river_result = self._river_result_from_state("river_scan_not_due")
        return self._observe_decision(hand_result, action_meta, river_result, started, phase="normal_tracking", meld_result=meld_result)

    def _observe_decision(
        self,
        hand_result: FastHandResult,
        action_meta: dict[str, Any],
        river_result: RiverStateResult,
        started: float,
        *,
        phase: str,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        self.state.round_phase = phase
        summary, detail, reason_codes = self._observe_message(hand_result)
        if hand_result.ok and hand_result.hand_tiles:
            self.state.last_hand_tiles = [normalize_tile(t) for t in hand_result.hand_tiles if normalize_tile(t)]
        return CoachDecision(
            decision_type="observe",
            priority=5,
            action_required=False,
            summary=summary,
            detail=detail,
            suggestion=self.state.current_plan,
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=reason_codes,
            coach_state=self.state.to_dict(),
            perception={
                "hand": hand_result.to_dict(),
                "action": action_meta,
                "river": river_result.to_dict(),
                "meld": meld_result.to_dict() if meld_result is not None else {},
            },
            engine_meta=self._meta(started, "observe"),
        )

    def _resolve_buttons(
        self,
        path: Path | None,
        observed_buttons: list[str] | None,
    ) -> tuple[list[str], dict[str, Any]]:
        normalized = _normalize_buttons(observed_buttons)
        meta: dict[str, Any] = {
            "source": "provided" if normalized else "fast_color_scan",
            "provided_buttons": list(normalized),
        }
        if normalized or path is None:
            return normalized, meta
        detected, metrics = detect_action_buttons_fast(path)
        meta.update({"detected_buttons": detected, "metrics": metrics})
        return _normalize_buttons(detected), meta

    def _detect_hand(self, path: Path | None, *, min_hand_tiles: int | None = None) -> FastHandResult:
        if path is None:
            return FastHandResult(reason="image_path_missing")
        return detect_fast_hand_path(
            path,
            calibration_dir=self.calibration_dir,
            min_hand_tiles=max(1, int(min_hand_tiles or self._min_hand_tiles_for_scan())),
            use_onnx_hand=self.config.onnx_hand_enabled,
        )

    def _min_hand_tiles_for_scan(self) -> int:
        if not self.state.opening_emitted:
            return 12
        if _play_style(self.config) != "fast":
            return 12
        last_count = len([tile for tile in self.state.last_hand_tiles if normalize_tile(tile)])
        return 2 if 0 < last_count <= 5 else 4

    def _detect_and_remember(self, path: Path | None) -> tuple[FastHandResult, MeldStateResult]:
        hand_result = self._detect_hand(path)
        if hand_result.ok:
            self._remember_hand(hand_result)
        meld_result = self._detect_melds(path, hand_result=hand_result)
        if meld_result.ok:
            self._remember_melds(meld_result)
        return hand_result, meld_result

    def _detect_river(self, path: Path | None) -> RiverStateResult:
        if not self.config.river_recognition_enabled:
            return RiverStateResult(reason="river_recognition_disabled")
        if path is None:
            return RiverStateResult(reason="image_path_missing")
        return detect_river_state_path(
            path,
            calibration_dir=self.calibration_dir,
            min_confidence=self.config.river_min_confidence,
        )

    def _detect_riichi_players(self, path: Path | None) -> list[str]:
        if not self.config.opponent_riichi_recognition_enabled:
            self.state.riichi_pending = {}
            return []
        if path is None:
            return []
        result = detect_riichi_sticks(path)
        detected = self._riichi_players_from_stick_counter(result)
        confirmed: list[str] = []
        pending = dict(self.state.riichi_pending)
        for player in detected:
            pending[player] = pending.get(player, 0) + 1
            if pending[player] >= 2:
                confirmed.append(player)
        for player in list(pending):
            if player not in detected:
                pending.pop(player, None)
        self.state.riichi_pending = pending
        already = set(self.state.riichi_players)
        return sorted(set(confirmed) | already)

    def _riichi_players_from_stick_counter(self, result: Any) -> set[str]:
        stick_count = result.stick_count
        if stick_count is not None:
            self.state.last_riichi_stick_count = stick_count
            if self.state.riichi_stick_baseline is None:
                self.state.riichi_stick_baseline = stick_count
                return set()
            if stick_count > self.state.riichi_stick_baseline:
                return {"unknown"}
            return set()
        if self.state.riichi_stick_baseline is None:
            return set()
        return set(result.riichi_players)

    def _detect_melds(self, path: Path | None, *, hand_result: FastHandResult | None = None) -> MeldStateResult:
        if not self.config.meld_recognition_enabled:
            return MeldStateResult(reason="meld_recognition_disabled")
        if path is None:
            return MeldStateResult(reason="image_path_missing")
        closed_hand_count = len(hand_result.hand_tiles) if hand_result is not None and hand_result.hand_tiles else None
        return detect_meld_state_path(
            path,
            min_confidence=self.config.meld_min_confidence,
            closed_hand_count=closed_hand_count,
        )

    def _river_result_from_state(self, reason: str, *, ok_if_cached: bool = True) -> RiverStateResult:
        has_cached = bool(self.state.last_discard_piles or self.state.last_visible_discards)
        return RiverStateResult(
            ok=ok_if_cached and has_cached,
            discard_piles={
                player: [dict(item) for item in items]
                for player, items in self.state.last_discard_piles.items()
            },
            visible_tiles=list(self.state.last_visible_discards),
            confidence=float(self.state.last_river_confidence),
            reason=reason,
        )

    def _observe_message(self, hand_result: FastHandResult) -> tuple[str, str, list[str]]:
        if hand_result.ok:
            if self.state.current_plan:
                return (
                    "当前主线继续有效",
                    "继续按当前主线推进；等三巡、手牌结构明显变化、或出现立直/和牌压力时再看主线。",
                    ["coach_observe", "current_plan_active"],
                )
            return (
                "Watching round state",
                "No critical action or coach checkpoint is due.",
                ["coach_observe"],
            )
        accepted = sum(1 for item in hand_result.raw_detections if item.get("accepted"))
        occupied = sum(1 for item in hand_result.raw_detections if item.get("occupied"))
        reason = str(hand_result.reason or "hand_unavailable")
        if reason == "image_path_missing":
            detail = "No screenshot path was provided yet."
        elif reason == "image_missing":
            detail = "The screenshot file could not be read."
        elif reason == "missing_hand_tile_templates":
            detail = "This screenshot size has no legacy hand-template profile yet."
        elif accepted == 0:
            detail = "No stable hand tiles were detected; live capture may be grabbing the desktop, menu, or a covered game window."
        else:
            min_tiles = self._min_hand_tiles_for_scan()
            expected = f"at least {min_tiles} open-hand tiles" if min_tiles <= 4 else "12-14 stable tiles"
            detail = f"Hand scan accepted {accepted} tiles from {occupied} occupied-looking slots; waiting for {expected}."
        return "Waiting for stable hand", detail, ["coach_observe", f"hand_{reason}"]

    def _remember_hand(self, hand_result: FastHandResult) -> None:
        tiles = [normalize_tile(tile) for tile in hand_result.hand_tiles if normalize_tile(tile)]
        signature = hand_signature(tiles)
        self.state.last_hand_signature = signature
        self.state.last_hand_tiles = tiles
        self.state.last_hand_confidence = float(hand_result.confidence)

    def _remember_river(self, river_result: RiverStateResult) -> None:
        self.state.last_discard_piles = {
            player: [dict(item) for item in items]
            for player, items in river_result.discard_piles.items()
        }
        self.state.last_visible_discards = list(river_result.visible_tiles)
        self.state.last_river_confidence = float(river_result.confidence)

    def _remember_melds(self, meld_result: MeldStateResult) -> None:
        self.state.last_melds = [dict(item) for item in meld_result.melds]
        tile_identity_reliable = meld_result.analysis_hints.get("tile_identity_reliable") is not False
        self.state.last_meld_tiles = [str(tile) for tile in meld_result.tiles if str(tile).strip()] if tile_identity_reliable else []
        self.state.last_open_meld_count = meld_result.open_meld_count
        self.state.last_meld_confidence = float(meld_result.confidence)

    def _critical_decision(
        self,
        buttons: list[str],
        action_meta: dict[str, Any],
        started: float,
        hand_result: FastHandResult | None = None,
        meld_result: MeldStateResult | None = None,
        river_result: RiverStateResult | None = None,
    ) -> CoachDecision:
        if any(button in WIN_BUTTONS for button in buttons):
            summary = "和牌窗口"
            suggestion = "看到荣和/自摸直接点，不需要等待策略分析。"
            decision_type = "win_window"
            priority = 100
        elif any(button in CALL_BUTTONS for button in buttons):
            summary = "吃碰杠窗口"
            suggestion = self._call_suggestion(hand_result, buttons, river_result, meld_result=meld_result)
            decision_type = "call_window"
            priority = 95
        elif "riichi" in buttons:
            summary = "立直窗口"
            suggestion = self._riichi_suggestion(hand_result, river_result)
            decision_type = "riichi_window"
            priority = 90
        else:
            summary = "操作窗口"
            suggestion = "先处理当前按钮，再回到局面策略。"
            decision_type = "action_window"
            priority = 80
        self.state.round_phase = "action_window"
        self.state.last_update_reason = decision_type
        self.state.update_count += 1
        return CoachDecision(
            decision_type=decision_type,
            priority=priority,
            action_required=True,
            summary=summary,
            detail="动作窗口只使用当前策略与本地快判。",
            suggestion=suggestion,
            buttons=list(buttons),
            hand_tiles=list(hand_result.hand_tiles) if hand_result else [],
            reason_codes=["critical_action_interrupt"],
            coach_state=self.state.to_dict(),
            perception={
                "action": action_meta,
                "hand": hand_result.to_dict() if hand_result else {},
                "meld": meld_result.to_dict() if meld_result else {},
                "river": river_result.to_dict() if river_result else {},
            },
            engine_meta=self._meta(started, decision_type),
        )

    def _call_suggestion(
        self,
        hand_result: FastHandResult | None,
        buttons: list[str] | None = None,
        river_result: RiverStateResult | None = None,
        meld_result: MeldStateResult | None = None,
    ) -> str:
        plan = self.state.current_plan or self.state.opening_plan
        if not plan and hand_result is not None and hand_result.ok:
            built = build_round_plan(
                hand_result.hand_tiles,
                self.config,
                visible_tiles=_visible_tiles_for_plan(river_result),
                open_melds=self._open_meld_count_for_plan(meld_result),
                meld_tiles=self._meld_tiles_for_plan(meld_result),
            )
            plan = built["summary"]
            self.state.opening_emitted = True
            self.state.round_phase = "opening_strategy"
            self._remember_local_plan(built, opening=True)
        if hand_result is not None and hand_result.ok:
            call_policy = _call_policy(hand_result.hand_tiles, self.config, buttons or [])
            if plan:
                return f"{call_policy} 当前主线：{plan}"
            return call_policy
        if plan:
            return f"默认跳过，除非鸣牌能明确推进当前主线：{plan}"
        return "默认跳过；只有役牌对子、直接进听、明显加速主线或安全和牌时才吃碰杠。"

    def _riichi_suggestion(
        self,
        hand_result: FastHandResult | None,
        river_result: RiverStateResult | None = None,
    ) -> str:
        if hand_result is None or not hand_result.ok:
            return "可立直：未拿到稳定手牌，先按游戏给出的立直窗口谨慎确认。"
        advice = _riichi_advice(hand_result.hand_tiles, self.config, visible_tiles=_visible_tiles_for_plan(river_result))
        if advice:
            return advice
        return "可立直：本地暂未算出明确听牌形，先确认待牌和打点。"

    def _opening_decision(
        self,
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        started: float,
        *,
        action_meta: dict[str, Any] | None = None,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        plan = build_round_plan(
            hand_result.hand_tiles,
            self.config,
            visible_tiles=_visible_tiles_for_plan(river_result),
            open_melds=self._open_meld_count_for_plan(meld_result),
            meld_tiles=self._meld_tiles_for_plan(meld_result),
        )
        self.state.opening_emitted = True
        self.state.round_phase = "opening_strategy"
        self._remember_local_plan(plan, opening=True)
        self.state.last_update_reason = "opening_plan"
        self.state.update_count += 1
        return CoachDecision(
            decision_type="opening_plan",
            priority=60,
            summary="Opening plan ready",
            detail=plan["detail"],
            suggestion=plan["summary"],
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=["first_stable_hand"],
            coach_state=self.state.to_dict(),
            perception={
                "hand": hand_result.to_dict(),
                "action": action_meta or {},
                "meld": meld_result.to_dict() if meld_result is not None else {},
                "river": river_result.to_dict(),
            },
            engine_meta=self._meta(started, "opening_plan"),
        )

    def _checkpoint_due(self, turn_number: int | None, *, force_checkpoint: bool) -> bool:
        if force_checkpoint:
            return True
        if turn_number is None:
            return False
        if turn_number <= self.state.last_checkpoint_self_turn:
            return False
        return (turn_number - self.state.last_checkpoint_self_turn) >= self.config.coach_checkpoint_self_turns

    def _checkpoint_decision(
        self,
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        turn_number: int | None,
        force_checkpoint: bool,
        started: float,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        plan = build_round_plan(
            hand_result.hand_tiles,
            self.config,
            visible_tiles=_visible_tiles_for_plan(river_result),
            open_melds=self._open_meld_count_for_plan(meld_result),
            meld_tiles=self._meld_tiles_for_plan(meld_result),
        )
        changed = self._plan_materially_changed(plan)
        if turn_number is not None:
            self.state.last_checkpoint_self_turn = turn_number
        self.state.round_phase = "checkpoint_strategy"
        self._remember_local_plan(plan)
        reason = "forced_checkpoint" if force_checkpoint else "scheduled_checkpoint"
        self.state.last_update_reason = reason
        self.state.update_count += 1
        return CoachDecision(
            decision_type="coach_checkpoint",
            priority=50 if changed else 5,
            summary="Round checkpoint updated",
            detail=plan["detail"],
            suggestion=plan["summary"],
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=[reason],
            coach_state=self.state.to_dict(),
            perception={
                "hand": hand_result.to_dict(),
                "meld": meld_result.to_dict() if meld_result is not None else {},
                "river": river_result.to_dict(),
            },
            engine_meta=self._meta(started, "coach_checkpoint"),
            quiet=not changed,
        )

    def _plan_materially_changed(self, plan: dict[str, Any]) -> bool:
        new_direction = str(plan.get("direction") or "").strip()
        if new_direction and new_direction != self.state.local_direction:
            return True
        new_discard = [str(t) for t in plan.get("discard_priority", []) if str(t).strip()]
        if new_discard and self.state.prev_discard_priority:
            if new_discard[0] != self.state.prev_discard_priority[0]:
                return True
        new_targets = [str(t) for t in plan.get("targets", []) if str(t).strip()]
        old_targets = self.state.target_shapes
        if new_targets != old_targets:
            return True
        return False

    def _remember_local_plan(self, plan: dict[str, Any], *, opening: bool = False) -> None:
        direction = str(plan.get("direction") or "").strip()
        summary = str(plan.get("summary") or "").strip()
        detail = str(plan.get("detail") or "").strip()
        targets = [str(item) for item in plan.get("targets", []) if str(item).strip()]
        cautions = [str(item) for item in plan.get("cautions", []) if str(item).strip()]
        discard_priority = [str(item) for item in plan.get("discard_priority", []) if str(item).strip()]
        display = direction if direction else summary
        self.state.prev_direction = self.state.local_direction
        self.state.prev_discard_priority = list(self.state.prev_discard_priority)
        if opening:
            self.state.opening_plan = display
        self.state.current_plan = display
        self.state.plan_source = "heuristic"
        self.state.local_direction = direction
        self.state.local_plan = summary
        self.state.local_detail = detail
        self.state.attack_defense_bias = str(plan.get("bias") or "neutral")
        self.state.target_shapes = targets
        self.state.caution_points = cautions
        self.state.prev_discard_priority = discard_priority

    def _defense_decision(
        self,
        riichi_players: list[str],
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        started: float,
        meld_result: MeldStateResult | None = None,
    ) -> CoachDecision:
        self.state.round_phase = "defense_mode"
        self.state.attack_defense_bias = "defense"
        self.state.last_update_reason = "riichi_defense"
        self.state.update_count += 1
        return CoachDecision(
            decision_type="defense_alert",
            priority=85,
            action_required=True,
            summary="Defense checkpoint",
            detail=f"Riichi pressure from {', '.join(riichi_players)}.",
            suggestion=self._defense_suggestion(riichi_players, river_result, hand_tiles=list(hand_result.hand_tiles) if hand_result.ok else None),
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=["riichi_players_present"],
            coach_state=self.state.to_dict(),
            perception={
                "hand": hand_result.to_dict(),
                "meld": meld_result.to_dict() if meld_result is not None else {},
                "river": river_result.to_dict(),
            },
            engine_meta=self._meta(started, "defense_alert"),
        )

    def _meta(self, started: float, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "live_advice_mode": self.config.live_advice_mode,
            "per_turn_discard_prompt": self.config.per_turn_discard_prompt,
            "hand_recognition_backend": self.config.hand_recognition_backend,
            "onnx_hand_enabled": self.config.onnx_hand_enabled,
            "river_recognition_enabled": self.config.river_recognition_enabled,
            "river_recognition_backend": "onnx_discard_model",
            "meld_recognition_enabled": self.config.meld_recognition_enabled,
            "meld_recognition_backend": "onnx_tile_classifier",
            "opponent_riichi_recognition_enabled": self.config.opponent_riichi_recognition_enabled,
        }

    def _open_meld_count_for_plan(self, meld_result: MeldStateResult | None) -> int | None:
        if meld_result is not None and meld_result.ok:
            return meld_result.open_meld_count
        return self.state.last_open_meld_count if self.state.last_open_meld_count else None

    def _meld_tiles_for_plan(self, meld_result: MeldStateResult | None) -> list[str]:
        if meld_result is not None and meld_result.ok:
            if meld_result.analysis_hints.get("tile_identity_reliable") is False:
                return []
            return [str(tile) for tile in meld_result.tiles if str(tile).strip()]
        return list(self.state.last_meld_tiles)

    def _defense_suggestion(
        self,
        riichi_players: list[str],
        river_result: RiverStateResult,
        hand_tiles: list[str] | None = None,
    ) -> str:
        piles = river_result.discard_piles if river_result.ok else self.state.last_discard_piles
        hand_set = {normalize_tile(t) for t in (hand_tiles or []) if normalize_tile(t)} or set(self.state.last_hand_tiles)
        safe_tiles: list[str] = []
        for player in riichi_players:
            key = _player_key(player)
            if not key:
                continue
            safe_tiles.extend(str(item.get("tile") or "") for item in piles.get(key, []))
        if safe_tiles:
            held_safe = [t for t in safe_tiles if t and normalize_tile(t) in hand_set]
            if held_safe:
                safe_text = _tile_list_in_order(held_safe[-8:])
                return f"防守优先：先看立直家现物 {safe_text}；没有现物再考虑筋/壁，宝牌周边先保守。"
            all_safe_text = _tile_list_in_order([t for t in safe_tiles if t][-8:])
            return f"防守优先：立直家现物 {all_safe_text}，但手里没有；考虑筋/壁或其他安全牌，宝牌周边先保守。"
        if river_result.ok and river_result.visible_tiles:
            visible_text = _tile_list_in_order(river_result.visible_tiles[-10:])
            return f"牌河已识别，可见弃牌参考 {visible_text}；未标定立直家座位时先保守找现物。"
        return "Slow down and prefer safe tiles from visible information."


def _visible_tiles_for_plan(river_result: RiverStateResult | None) -> list[str]:
    if river_result is None or not river_result.ok:
        return []
    tiles = list(river_result.visible_tiles)
    if tiles:
        return tiles
    result: list[str] = []
    for items in river_result.discard_piles.values():
        result.extend(str(item.get("tile") or "") for item in items if isinstance(item, dict))
    return result


def build_round_plan(
    hand_tiles: list[str],
    config: MahjongCoachConfig | None = None,
    *,
    visible_tiles: list[str] | None = None,
    open_melds: int | None = None,
    meld_tiles: list[str] | None = None,
) -> dict[str, Any]:
    tiles = [normalize_tile(tile) for tile in hand_tiles if normalize_tile(tile)]
    meld_tile_list = [normalize_tile(tile) for tile in (meld_tiles or []) if normalize_tile(tile)]
    shape_tiles = [*tiles, *meld_tile_list]
    counts = Counter(tiles)
    visible_counts = _visible_counts([*(visible_tiles or []), *meld_tile_list])
    value_honors = _value_honor_tiles(config)
    dora_tiles = _dora_tiles(config)
    style = _play_style(config)
    explicit_open_melds = _coerce_open_melds(open_melds)
    inferred_open_melds = _inferred_open_melds(len(tiles)) if style == "fast" else 0
    open_meld_count = explicit_open_melds if explicit_open_melds is not None else inferred_open_melds
    suit_counts = Counter(tile_suit(tile) for tile in tiles if tile_suit(tile))
    honor_count = sum(1 for tile in tiles if is_honor(tile))
    terminal_count = sum(1 for tile in tiles if is_terminal(tile))
    simple_count = sum(1 for tile in tiles if is_simple(tile))
    pair_count = sum(1 for value in counts.values() if value >= 2)
    best_suit, best_suit_count = ("", 0)
    suited = {suit: count for suit, count in suit_counts.items() if suit in {"m", "p", "s"}}
    if suited:
        best_suit, best_suit_count = max(suited.items(), key=lambda item: item[1])
    second_suit_count = max((count for suit, count in suited.items() if suit != best_suit), default=0)
    pair_tiles = [tile for tile, value in counts.items() if value >= 2]
    value_pair_tiles = [tile for tile in pair_tiles if tile in value_honors]

    suit_threshold = _RIICHI_SUIT_THRESHOLD if style == "riichi" else _FAST_SUIT_THRESHOLD
    pair_threshold = _RIICHI_PAIR_THRESHOLD if style == "riichi" else _FAST_PAIR_THRESHOLD
    simple_threshold = _RIICHI_SIMPLE_THRESHOLD if style == "riichi" else _FAST_SIMPLE_THRESHOLD
    is_pair_route = open_meld_count == 0 and pair_count >= pair_threshold and best_suit_count < suit_threshold

    cleanup_tiles = _cleanup_candidates(tiles, counts, best_suit, value_honors, dora_tiles)
    keep_tiles = pair_tiles if is_pair_route else _keep_candidates(tiles, counts, best_suit, dora_tiles)
    discard_tiles = _discard_candidates(tiles, counts, best_suit, cleanup_tiles, keep_tiles, value_honors, dora_tiles)
    efficiency = _efficiency_analysis(tiles, discard_tiles, counts, visible_counts, best_suit, value_honors, dora_tiles, open_melds=open_meld_count)
    best_efficiency_discard = _best_efficiency_discard(efficiency)
    if best_efficiency_discard:
        discard_tiles = _merge_unique_tiles([best_efficiency_discard, *discard_tiles])
    discard_text = _tile_list_in_order(discard_tiles)
    keep_text = _tile_list(keep_tiles) or "连续搭子和中张"
    suit_text = _suit_breakdown(shape_tiles) if meld_tile_list else _suit_breakdown(tiles)
    best_shape = _suit_shape(shape_tiles, best_suit) if best_suit else ""
    meld_text = _tile_list_in_order(meld_tile_list)
    cleanup_text = _tile_list_in_order(cleanup_tiles) or discard_text or "孤张字牌和远张"
    primary_discard_text = _tile_name(best_efficiency_discard) if best_efficiency_discard else _first_tile_name(discard_tiles or cleanup_tiles)
    primary_discard_text = primary_discard_text or cleanup_text
    route_options = _route_discard_options(discard_tiles, counts, best_suit, best_suit_count, second_suit_count, pair_count, bool(value_pair_tiles), honor_count, terminal_count, simple_count)
    route_text = _route_options_text(route_options)

    direction, targets, summary = _choose_direction(
        open_meld_count, best_suit, best_suit_count, best_shape, second_suit_count,
        pair_count, pair_tiles, value_pair_tiles, is_pair_route, simple_count,
        simple_threshold, honor_count, terminal_count, keep_text, efficiency, meld_text, meld_tile_list,
        dora_tiles, shape_tiles, value_honors, style,
    )
    cautions = _build_cautions(
        honor_count, terminal_count, best_suit_count, open_meld_count, route_text,
        efficiency, cleanup_tiles, discard_text, cleanup_text, primary_discard_text,
        meld_tile_list, shape_tiles, value_honors, best_suit, counts, simple_count,
        is_pair_route, style,
    )

    bias = "attack" if simple_count >= 8 or best_suit_count >= 5 or value_pair_tiles else "neutral"
    summary = _plan_summary(summary, keep_text, primary_discard_text, efficiency)
    detail = (
        f"结构：{suit_text}；对子 {pair_count} 组。主线：{summary}。"
        f"{(' 已副露' + str(open_meld_count) + '组，') if open_meld_count else ' '}"
        f" 当前先保留 {keep_text}，{('路线选择：' + route_text) if route_text else ('候选打牌 ' + (discard_text or cleanup_text))}。"
        f" {_efficiency_detail(efficiency)}"
    )
    return {
        "direction": direction,
        "summary": summary,
        "detail": detail,
        "bias": bias,
        "targets": targets,
        "cautions": cautions,
        "discard_priority": list(discard_tiles),
        "efficiency": efficiency,
    }


def _choose_direction(
    open_meld_count: int,
    best_suit: str,
    best_suit_count: int,
    best_shape: str,
    second_suit_count: int,
    pair_count: int,
    pair_tiles: list[str],
    value_pair_tiles: list[str],
    is_pair_route: bool,
    simple_count: int,
    simple_threshold: int,
    honor_count: int,
    terminal_count: int,
    keep_text: str,
    efficiency: dict[str, Any],
    meld_text: str,
    meld_tile_list: list[str],
    dora_tiles: set[str],
    shape_tiles: list[str],
    value_honors: set[str],
    style: str,
) -> tuple[str, list[str], str]:
    suit_threshold = _RIICHI_SUIT_THRESHOLD if style == "riichi" else _FAST_SUIT_THRESHOLD
    direction = "牌效推进"
    targets: list[str] = []
    summary = ""

    if open_meld_count:
        direction = _open_hand_direction(open_meld_count, efficiency)
        targets.append(f"主线：已副露{open_meld_count}组，按最快进听/听牌收束")
        targets.append(f"保留：{keep_text}")
        summary = f"已副露{open_meld_count}组，优先打向听最低、有效牌最多的牌"
    elif best_suit_count >= suit_threshold:
        direction = f"染手({SUIT_NAMES.get(best_suit, best_suit)}子)"
        targets.append(f"主线：{SUIT_NAMES.get(best_suit, best_suit)}子清一色/混一色倾向")
        summary = f"{SUIT_NAMES.get(best_suit, best_suit)}子多，优先保留同色，杂色牌逐步清理"
    elif is_pair_route:
        direction = "七对子"
        targets.append(f"主线：七对子胚子，已有对子 {_tile_list(pair_tiles)}")
        summary = "保留现有对子，不再做面子手"
    elif value_pair_tiles:
        direction = "役牌速攻"
        targets.append(f"主线：役牌速度，保留/可碰 {_tile_list(value_pair_tiles)}")
        summary = f"围绕役牌 {_tile_list(value_pair_tiles)} 加速，同时保持听牌潜力"
    elif best_suit_count >= 5 and best_suit_count >= second_suit_count + 2:
        direction = f"围绕{SUIT_NAMES.get(best_suit, best_suit)}子"
        targets.append(f"主线：围绕{SUIT_NAMES.get(best_suit, best_suit)}子 {best_shape} 推进")
        summary = f"以{SUIT_NAMES.get(best_suit, best_suit)}子为主，{best_shape} 找顺子成型"
    elif simple_count >= simple_threshold and honor_count + terminal_count <= 3:
        direction = "断幺九/平和"
        targets.extend(["主线：断幺/平和速度", f"保留：{keep_text}"])
        summary = f"断幺九或平和速度路线，保留{keep_text}"
    elif honor_count >= 4:
        direction = "清字牌"
        targets.append("主线：先清孤字牌")
        summary = "字牌偏多，先清孤字，只留役牌对子"
    else:
        targets.append("主线：牌效推进")
        summary = f"牌效推进，保留{keep_text}"

    target_shape = _target_shape_text(direction, best_suit, best_shape, keep_text, efficiency, open_meld_count, bool(value_pair_tiles), is_pair_route)
    if target_shape:
        targets.append(f"目标形：{target_shape}")
    if keep_text and not any(item.startswith("保留：") for item in targets):
        targets.append(f"保留：{keep_text}")
    if meld_text:
        targets.append(f"副露识别：{meld_text}")
    if pair_tiles:
        targets.append(f"对子：{_tile_list(pair_tiles)}")
    if value_pair_tiles:
        targets.append(f"役牌对子：{_tile_list(value_pair_tiles)}")
    open_yaku_targets, _ = _open_yaku_notes(meld_tile_list, shape_tiles, value_honors, best_suit, open_meld_count)
    targets.extend(open_yaku_targets)
    dora_text = _tile_list([tile for tile in shape_tiles if _is_dora(tile, dora_tiles)])
    if dora_text:
        targets.append(f"宝牌/红5：{dora_text}")

    return direction, targets, summary


def _build_cautions(
    honor_count: int,
    terminal_count: int,
    best_suit_count: int,
    open_meld_count: int,
    route_text: str,
    efficiency: dict[str, Any],
    cleanup_tiles: list[str],
    discard_text: str,
    cleanup_text: str,
    primary_discard_text: str,
    meld_tile_list: list[str],
    shape_tiles: list[str],
    value_honors: set[str],
    best_suit: str,
    counts: Counter[str],
    simple_count: int,
    is_pair_route: bool,
    style: str,
) -> list[str]:
    cautions: list[str] = []
    if honor_count >= 3:
        cautions.append("孤字牌不要久留，除非成对或有役牌价值。")
    if terminal_count >= 4:
        cautions.append("孤幺九可清，已经组成边搭/对子再保留。")
    if best_suit_count >= 7:
        cautions.append("有染手分支，但摸到强中张时不要硬染。")
    if route_text:
        cautions.append(f"{'副露收束' if open_meld_count else '路线选择'}：{route_text}")
    efficiency_note = _efficiency_note(efficiency)
    if efficiency_note:
        cautions.append(efficiency_note)
    if cleanup_tiles:
        cautions.append(f"优先清理：{discard_text or cleanup_text}")
    next_step = _next_step_text(primary_discard_text, efficiency, open_meld_count)
    if next_step:
        cautions.append(f"下一步：{next_step}")
    _, open_yaku_cautions = _open_yaku_notes(meld_tile_list, shape_tiles, value_honors, best_suit, open_meld_count)
    if open_meld_count:
        cautions.extend(open_yaku_cautions)
        cautions.append(_open_hand_policy_line(open_meld_count))
    else:
        cautions.append(_open_policy_line(counts, value_honors, best_suit_count, simple_count, honor_count, terminal_count, is_pair_route, style))
    if not cautions:
        cautions.append("三巡后再看主线，不要每摸一张就推翻主线。")
    return cautions


def _tile_name(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in HONOR_NAMES:
        return HONOR_NAMES[normalized]
    if normalized in {"0m", "0p", "0s"}:
        return f"红5{SUIT_NAMES.get(tile_suit(normalized), '')}"
    if len(normalized) == 2:
        return f"{tile_rank(normalized)}{SUIT_NAMES.get(tile_suit(normalized), '')}"
    return normalized


def _tile_list(tiles: list[str]) -> str:
    unique = []
    for tile in sorted((normalize_tile(tile) for tile in tiles if normalize_tile(tile)), key=_tile_sort_key):
        if tile not in unique:
            unique.append(tile)
    return "、".join(_tile_name(tile) for tile in unique[:6])


def _tile_list_in_order(tiles: list[str]) -> str:
    unique = []
    for tile in (normalize_tile(tile) for tile in tiles if normalize_tile(tile)):
        if tile not in unique:
            unique.append(tile)
    return "、".join(_tile_name(tile) for tile in unique[:6])


def _tile_sort_key(tile: str) -> tuple[int, int, str]:
    suit_order = {"m": 0, "p": 1, "s": 2, "z": 3}
    normalized = normalize_tile(tile)
    rank = tile_rank(normalized)
    return (suit_order.get(tile_suit(normalized), 9), int(rank) if rank.isdigit() else 0, normalized)


_PLAYER_KEY_ALIASES = {
    "self": "self",
    "left": "left_opponent",
    "left_opponent": "left_opponent",
    "kamicha": "left_opponent",
    "top": "top_opponent",
    "top_opponent": "top_opponent",
    "toimen": "top_opponent",
    "right": "right_opponent",
    "right_opponent": "right_opponent",
    "shimocha": "right_opponent",
}


def _player_key(value: Any) -> str:
    return _PLAYER_KEY_ALIASES.get(str(value or "").strip().lower(), "")


def _suit_shape(tiles: list[str], suit: str) -> str:
    ranks = [tile_rank(tile) for tile in tiles if tile_suit(tile) == suit]
    return "".join(ranks) or "-"


def _suit_breakdown(tiles: list[str]) -> str:
    parts = []
    for suit in ("m", "p", "s", "z"):
        shape = _suit_shape(tiles, suit)
        if shape != "-":
            parts.append(f"{SUIT_NAMES[suit]}{shape}")
    return " / ".join(parts) if parts else "暂无稳定手牌"


def _keep_candidates(tiles: list[str], counts: Counter[str], best_suit: str, dora_tiles: set[str]) -> list[str]:
    keep: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if counts[normalized] >= 2 or normalized in {"0m", "0p", "0s"} or _is_dora(normalized, dora_tiles):
            keep.append(normalized)
            continue
        if tile_suit(normalized) == best_suit and _has_neighbor(normalized, counts, distance=2):
            keep.append(normalized)
    return keep


def _cleanup_candidates(
    tiles: list[str],
    counts: Counter[str],
    best_suit: str,
    value_honors: set[str],
    dora_tiles: set[str],
) -> list[str]:
    cleanup: list[str] = []
    for tile in sorted((normalize_tile(tile) for tile in tiles if normalize_tile(tile)), key=_tile_sort_key):
        if counts[tile] >= 2 or tile in {"0m", "0p", "0s"} or _is_dora(tile, dora_tiles):
            continue
        suit = tile_suit(tile)
        if suit == "z":
            cleanup.append(tile)
        elif suit != best_suit and not _has_neighbor(tile, counts, distance=1):
            cleanup.append(tile)
        elif suit == best_suit and is_terminal(tile) and not _has_neighbor(tile, counts, distance=1):
            cleanup.append(tile)
    return sorted(cleanup, key=lambda tile: _cleanup_score(tile, counts, best_suit, value_honors))[:5]


def _cleanup_score(tile: str, counts: Counter[str], best_suit: str, value_honors: set[str]) -> tuple[int, int, str]:
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    rank_value = int(rank) if rank.isdigit() else 0
    if suit == "z":
        if tile in value_honors:
            return (3, rank_value, tile)
        return (0, rank_value, tile)
    if best_suit and suit != best_suit and not _has_neighbor(tile, counts, distance=1):
        return (1 if is_terminal(tile) else 2, rank_value, tile)
    if suit == best_suit and is_terminal(tile) and not _has_neighbor(tile, counts, distance=1):
        return (4, rank_value, tile)
    return (6, rank_value, tile)


def _discard_candidates(
    tiles: list[str],
    counts: Counter[str],
    best_suit: str,
    cleanup_tiles: list[str],
    keep_tiles: list[str],
    value_honors: set[str],
    dora_tiles: set[str],
) -> list[str]:
    selected: list[str] = []
    keep_set = set(keep_tiles)
    for tile in sorted(cleanup_tiles, key=lambda tile: _discard_score(tile, counts, best_suit, keep_set, value_honors, dora_tiles)):
        if tile not in selected:
            selected.append(tile)

    pool: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if not normalized or normalized in selected or normalized in {"0m", "0p", "0s"} or _is_dora(normalized, dora_tiles):
            continue
        if counts[normalized] >= 2:
            continue
        pool.append(normalized)

    pool = sorted(set(pool), key=lambda tile: _discard_score(tile, counts, best_suit, keep_set, value_honors, dora_tiles))
    for tile in pool:
        if len(selected) >= 5:
            break
        selected.append(tile)
    return selected[:4]


def _efficiency_analysis(
    tiles: list[str],
    initial_discards: list[str],
    counts: Counter[str],
    visible_counts: Counter[str],
    best_suit: str,
    value_honors: set[str],
    dora_tiles: set[str],
    *,
    open_melds: int = 0,
) -> dict[str, Any]:
    canonical_tiles = [_canonical_tile(tile) for tile in tiles if _canonical_tile(tile) in TILE_TYPES]
    if not canonical_tiles:
        return {
            "current_shanten": 8,
            "best_path": "standard",
            "discard_options": [],
            "open_melds": _clamp_melds(open_melds),
            "closed_tile_count": 0,
        }

    open_meld_count = _clamp_melds(open_melds)
    current = _best_hand_shanten(canonical_tiles, open_melds=open_meld_count)
    candidates = _efficiency_discard_candidates(tiles, initial_discards, counts, best_suit, value_honors, dora_tiles)
    options: list[dict[str, Any]] = []
    visible_total = sum(visible_counts.values())

    if len(canonical_tiles) % 3 == 2:
        hand_counts = Counter(canonical_tiles)
        for tile in candidates:
            canonical = _canonical_tile(tile)
            if canonical not in hand_counts:
                continue
            remaining = list(canonical_tiles)
            remaining.remove(canonical)
            post_discard = _best_hand_shanten(remaining, open_melds=open_meld_count)["shanten"]
            effective_tiles: list[str] = []
            effective_count = 0
            for draw in TILE_TYPES:
                available = max(0, 4 - hand_counts.get(draw, 0) - visible_counts.get(draw, 0))
                if available <= 0:
                    continue
                if _best_hand_shanten([*remaining, draw], open_melds=open_meld_count)["shanten"] < post_discard:
                    effective_tiles.append(draw)
                    effective_count += available
            options.append(
                {
                    "tile": tile,
                    "shanten": post_discard,
                    "effective_types": len(effective_tiles),
                    "effective_count": effective_count,
                    "effective_tiles": sorted(effective_tiles, key=_tile_sort_key)[:10],
                    "heuristic_score": _discard_score(tile, counts, best_suit, set(), value_honors, dora_tiles),
                }
            )

    options.sort(
        key=lambda item: (
            item["shanten"],
            -item["effective_count"],
            -item["effective_types"],
            item["heuristic_score"],
        )
    )
    return {
        "current_shanten": current["shanten"],
        "best_path": current["path"],
        "open_melds": open_meld_count,
        "closed_tile_count": len(canonical_tiles),
        "discard_options": [
            {
                "tile": str(item["tile"]),
                "shanten": item["shanten"],
                "effective_types": item["effective_types"],
                "effective_count": item["effective_count"],
                "effective_tiles": list(item["effective_tiles"]),
            }
            for item in options[:5]
        ],
        "visible_tile_count": visible_total,
    }


def _efficiency_discard_candidates(
    tiles: list[str],
    initial_discards: list[str],
    counts: Counter[str],
    best_suit: str,
    value_honors: set[str],
    dora_tiles: set[str],
) -> list[str]:
    candidates = _merge_unique_tiles(initial_discards)
    keep_set: set[str] = set()
    for tile in sorted(
        {normalize_tile(tile) for tile in tiles if normalize_tile(tile)},
        key=lambda item: _discard_score(item, counts, best_suit, keep_set, value_honors, dora_tiles),
    ):
        if tile in {"0m", "0p", "0s"} or _is_dora(tile, dora_tiles):
            continue
        if tile not in candidates:
            candidates.append(tile)
    return candidates[:12]


def _best_hand_shanten(tiles: list[str], *, open_melds: int = 0) -> dict[str, Any]:
    counts = _shanten_counts(tiles)
    open_meld_count = _clamp_melds(open_melds)
    standard = _standard_shanten(tuple(counts), open_meld_count)
    options = [("standard", standard)]
    if open_meld_count == 0:
        options.extend(
            [
                ("seven_pairs", _seven_pairs_shanten(counts)),
                ("thirteen_orphans", _thirteen_orphans_shanten(counts)),
            ]
        )
    path, shanten = min(options, key=lambda item: item[1])
    return {"path": path, "shanten": int(shanten)}


def _shanten_counts(tiles: list[str]) -> list[int]:
    counts = [0] * len(TILE_TYPES)
    for tile in tiles:
        index = _tile_index(_canonical_tile(tile))
        if index is not None:
            counts[index] += 1
    return counts


def _visible_counts(tiles: list[str]) -> Counter[str]:
    return Counter(_canonical_tile(tile) for tile in tiles if _canonical_tile(tile) in TILE_TYPES)


@lru_cache(maxsize=4096)
def _standard_shanten(counts_tuple: tuple[int, ...], open_melds: int = 0) -> int:
    open_meld_count = _clamp_melds(open_melds)

    @lru_cache(maxsize=262144)
    def walk(counts_state: tuple[int, ...], melds: int, taatsu: int, pair: int) -> int:
        counts = list(counts_state)
        capped_taatsu = min(taatsu, max(0, 4 - melds))
        best = 8 - (2 * melds) - capped_taatsu - pair
        index = next((offset for offset, value in enumerate(counts) if value > 0), -1)
        if index < 0:
            return best

        counts[index] -= 1
        best = min(best, walk(tuple(counts), melds, taatsu, pair))
        counts[index] += 1

        if melds < 4 and counts[index] >= 3:
            counts[index] -= 3
            best = min(best, walk(tuple(counts), melds + 1, taatsu, pair))
            counts[index] += 3

        if melds < 4 and _can_form_sequence(index) and counts[index + 1] > 0 and counts[index + 2] > 0:
            counts[index] -= 1
            counts[index + 1] -= 1
            counts[index + 2] -= 1
            best = min(best, walk(tuple(counts), melds + 1, taatsu, pair))
            counts[index] += 1
            counts[index + 1] += 1
            counts[index + 2] += 1

        if pair == 0 and counts[index] >= 2:
            counts[index] -= 2
            best = min(best, walk(tuple(counts), melds, taatsu, 1))
            counts[index] += 2

        if taatsu < 4 and counts[index] >= 2:
            counts[index] -= 2
            best = min(best, walk(tuple(counts), melds, taatsu + 1, pair))
            counts[index] += 2

        if taatsu < 4 and _can_form_sequence(index) and counts[index + 1] > 0:
            counts[index] -= 1
            counts[index + 1] -= 1
            best = min(best, walk(tuple(counts), melds, taatsu + 1, pair))
            counts[index] += 1
            counts[index + 1] += 1

        if taatsu < 4 and _can_form_sequence(index) and counts[index + 2] > 0:
            counts[index] -= 1
            counts[index + 2] -= 1
            best = min(best, walk(tuple(counts), melds, taatsu + 1, pair))
            counts[index] += 1
            counts[index + 2] += 1

        return best

    return walk(counts_tuple, open_meld_count, 0, 0)


def _seven_pairs_shanten(counts: list[int]) -> int:
    pairs = sum(1 for count in counts if count >= 2)
    unique = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - unique)


def _thirteen_orphans_shanten(counts: list[int]) -> int:
    orphan_indexes = [_tile_index(tile) for tile in ORPHAN_TYPES]
    present = sum(1 for index in orphan_indexes if index is not None and counts[index] > 0)
    has_pair = any(index is not None and counts[index] >= 2 for index in orphan_indexes)
    return 13 - present - int(has_pair)


def _can_form_sequence(index: int) -> bool:
    return 0 <= index < 27 and index % 9 <= 6


def _tile_index(tile: str) -> int | None:
    canonical = _canonical_tile(tile)
    try:
        return TILE_TYPES.index(canonical)
    except ValueError:
        return None


def _canonical_tile(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in {"0m", "0p", "0s"}:
        return f"5{normalized[1]}"
    return normalized


def _best_efficiency_discard(efficiency: dict[str, Any]) -> str:
    options = efficiency.get("discard_options")
    if isinstance(options, list) and options:
        first = options[0]
        if isinstance(first, dict):
            return str(first.get("tile") or "")
    return ""


def _efficiency_note(efficiency: dict[str, Any]) -> str:
    current = _shanten_text(int(efficiency.get("current_shanten", 8)))
    open_melds = int(efficiency.get("open_melds", 0))
    prefix = f"副露牌效：估算{open_melds}组，" if open_melds else "牌效："
    options = efficiency.get("discard_options")
    if isinstance(options, list) and options:
        best = options[0]
        if isinstance(best, dict):
            tile = _tile_name(str(best.get("tile") or ""))
            shanten = _shanten_text(int(best.get("shanten", 8)))
            effective_types = int(best.get("effective_types", 0))
            effective_count = int(best.get("effective_count", 0))
            waits = _tile_list_in_order([str(tile) for tile in best.get("effective_tiles", [])])
            wait_text = f"（{waits}）" if waits else ""
            visible_count = int(efficiency.get("visible_tile_count", 0))
            visible_text = "，已扣可见牌" if visible_count else ""
            return f"{prefix}当前{current}；打{tile}后{shanten}，有效{effective_types}种{effective_count}枚{visible_text}{wait_text}。"
    return f"{prefix}当前{current}；未到出牌手数，先按主线保留结构。"


def _efficiency_detail(efficiency: dict[str, Any]) -> str:
    path_names = {"standard": "面子手", "seven_pairs": "七对子", "thirteen_orphans": "国士"}
    path = path_names.get(str(efficiency.get("best_path") or "standard"), "面子手")
    return f"牌效按{path}估算，{_efficiency_note(efficiency)}"


def _shanten_text(value: int) -> str:
    if value <= -1:
        return "和牌形"
    if value == 0:
        return "听牌"
    return f"{value}向听"


def _plan_summary(summary: str, keep_text: str, primary_discard_text: str, efficiency: dict[str, Any]) -> str:
    shanten = _shanten_text(int(efficiency.get("current_shanten", 8)))
    discard = str(primary_discard_text or "").strip()
    keep = str(keep_text or "").strip()
    if discard:
        return f"{summary}；留{keep}，先看打{discard}，当前{shanten}"
    return f"{summary}；留{keep}，当前{shanten}"


def _target_shape_text(
    direction: str,
    best_suit: str,
    best_shape: str,
    keep_text: str,
    efficiency: dict[str, Any],
    open_melds: int,
    has_value_pair: bool,
    pair_route: bool,
) -> str:
    shanten = _shanten_text(int(efficiency.get("current_shanten", 8)))
    if open_melds:
        return f"{shanten}，剩余手牌只保留进听块，避免再转慢速大牌"
    if pair_route:
        return "七对子优先，继续收对子；孤张只保留安全、宝牌或强中张"
    if has_value_pair:
        return f"役牌对子可碰，碰成刻子后按{shanten}收束"
    if best_suit and "染手" in direction:
        return f"{SUIT_NAMES.get(best_suit, best_suit)}子{best_shape}，保留同色和役牌，杂色逐步切"
    if "断幺" in direction:
        return "断幺/平和速度，保留中张两面，孤幺九和孤字先处理"
    return f"{shanten}，保留{keep_text}，每次优先比较向听和有效牌"


def _next_step_text(primary_discard_text: str, efficiency: dict[str, Any], open_melds: int) -> str:
    discard = str(primary_discard_text or "").strip()
    options = efficiency.get("discard_options")
    if isinstance(options, list) and options:
        best = options[0]
        if isinstance(best, dict):
            effective = int(best.get("effective_count", 0) or 0)
            types = int(best.get("effective_types", 0) or 0)
            shanten = _shanten_text(int(best.get("shanten", 8)))
            tile = _tile_name(str(best.get("tile") or "")) or discard
            open_note = "；副露后只开能进听/听牌的牌" if open_melds else ""
            return f"优先看打{tile}后{shanten}，有效{types}种{effective}枚{open_note}"
    if discard:
        return f"先处理{discard}，三巡后再复核主线"
    return ""


def _merge_unique_tiles(tiles: list[str]) -> list[str]:
    result: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _first_tile_name(tiles: list[str]) -> str:
    first = _first_tile(tiles)
    return _tile_name(first) if first else ""


def _value_honor_tiles(config: MahjongCoachConfig | None) -> set[str]:
    values = {"5z", "6z", "7z"}
    if config is None:
        return values
    for tile in (config.round_wind, config.seat_wind):
        normalized = _context_tile(tile)
        if tile_suit(normalized) == "z":
            values.add(normalized)
    return values


def _dora_tiles(config: MahjongCoachConfig | None) -> set[str]:
    if config is None:
        return set()
    return {
        _canonical_tile(_context_tile(tile))
        for tile in config.dora_tiles
        if _canonical_tile(_context_tile(tile)) in TILE_TYPES
    }


def _is_dora(tile: str, dora_tiles: set[str]) -> bool:
    return _canonical_tile(tile) in dora_tiles


def _context_tile(tile: str) -> str:
    raw = str(tile or "").strip()
    return normalize_tile(TEXT_TILE_ALIASES.get(raw, raw))


def _play_style(config: MahjongCoachConfig | None) -> str:
    if config is None:
        return "riichi"
    return _valid_play_style(config.play_style)


def _inferred_open_melds(closed_tile_count: int) -> int:
    count = max(0, int(closed_tile_count or 0))
    if count >= 12:
        return 0
    if count >= 9:
        return 1
    if count >= 6:
        return 2
    if count >= 3:
        return 3
    return 4 if count > 0 else 0


def _coerce_open_melds(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        return max(0, min(4, int(value)))
    except (TypeError, ValueError):
        return None


def _clamp_melds(value: int) -> int:
    return max(0, min(4, int(value or 0)))


def _open_yaku_notes(
    meld_tiles: list[str],
    shape_tiles: list[str],
    value_honors: set[str],
    best_suit: str,
    open_melds: int,
) -> tuple[list[str], list[str]]:
    if open_melds <= 0:
        return [], []
    canonical_melds = [_canonical_tile(tile) for tile in meld_tiles if _canonical_tile(tile) in TILE_TYPES]
    if not canonical_melds:
        return [], ["确认役种：副露牌没识别到，别只凭剩余手牌判断有役。"]

    targets: list[str] = []
    cautions: list[str] = []
    meld_counts = Counter(canonical_melds)
    value_triplets = [tile for tile, count in meld_counts.items() if count >= 3 and tile in value_honors]
    if value_triplets:
        targets.append(f"已成役：役牌副露 {_tile_list(value_triplets)}")

    canonical_shape = [_canonical_tile(tile) for tile in shape_tiles if _canonical_tile(tile) in TILE_TYPES]
    if canonical_shape and all(is_simple(tile) for tile in canonical_shape):
        targets.append("可成役：断幺九")
    else:
        non_honor_suits = {tile_suit(tile) for tile in canonical_shape if tile_suit(tile) in {"m", "p", "s"}}
        if best_suit and len(non_honor_suits) == 1:
            suit_name = SUIT_NAMES.get(best_suit, best_suit)
            if any(is_honor(tile) for tile in canonical_shape):
                targets.append(f"可成役：混一色({suit_name}子)")
            else:
                targets.append(f"可成役：清一色({suit_name}子)")

    if not targets:
        cautions.append("确认役种：没有役牌、断幺或染手时，副露手可能没役。")
    return targets, cautions


def _open_hand_direction(open_melds: int, efficiency: dict[str, Any]) -> str:
    shanten = int(efficiency.get("current_shanten", 8))
    if shanten <= 0:
        return "副露听牌/收束"
    if shanten == 1:
        return "副露一向听"
    if open_melds >= 2:
        return "副露进听优先"
    return "副露加速"


def _open_hand_policy_line(open_melds: int) -> str:
    if open_melds >= 3:
        return "鸣牌：只开能听牌/和牌的牌，其他先别破坏待牌。"
    if open_melds >= 2:
        return "鸣牌：继续快攻，但只开能进听或明显增加有效牌的牌。"
    return "鸣牌：能推进主线就开，开过后优先进听而不是改做大牌。"


def _meld_policy(
    value_pairs: str,
    best_suit_count: int,
    simple_count: int,
    honor_count: int,
    terminal_count: int,
    pair_route: bool,
    style: str,
    *,
    is_action_window: bool = False,
) -> str:
    prefix = "" if is_action_window else "鸣牌："
    if style == "fast":
        if value_pairs:
            return f"役牌对子{value_pairs}，碰到可以开；中张吃碰能加速也开。" if is_action_window else f"鸣牌：{value_pairs}对子可以碰；中张吃碰能加速也开。"
        if pair_route:
            return "七对子胚子一般不鸣；但役牌刻子或明显加速可开。"
        if best_suit_count >= 6:
            return f"{prefix}染手主线可开同色/役牌，能推进就吃碰。"
        if simple_count >= 7 and honor_count + terminal_count <= 4:
            return f"{prefix}断幺速度手积极开中张，能进听或加速就吃碰。"
        return f"{prefix}快攻风格，能推进主线或进听就吃碰；安全牌也考虑和牌。" if is_action_window else "鸣牌：快攻风格，能推进主线或进听就吃碰；安全牌也考虑和牌。"
    if value_pairs:
        return f"役牌对子{value_pairs}，碰到可以开；没碰到役牌就默认跳过。" if is_action_window else f"鸣牌：{value_pairs}对子可以碰；其余仍要能听牌或明显加速。"
    if pair_route:
        return "七对子胚子一般不鸣；只有役牌碰成刻子，或直接听牌才开。" if is_action_window else "鸣牌：七对子胚子一般不鸣，除非役牌刻子直接成型。"
    if best_suit_count >= 7:
        return f"{prefix}染手主线可开同色/役牌，非主线牌默认跳过。"
    if simple_count >= 9 and honor_count + terminal_count <= 3:
        return f"{prefix}断幺速度手可开中张；不能进听或破坏好形就跳过。"
    return "默认跳过；只有役牌、直接进听、明显加速主线或安全和牌时才吃碰杠。" if is_action_window else "鸣牌：默认跳过，只有役牌、直接听牌、或明显加速主线才开。"


def _open_policy_line(
    counts: Counter[str],
    value_honors: set[str],
    best_suit_count: int,
    simple_count: int,
    honor_count: int,
    terminal_count: int,
    pair_route: bool,
    style: str = "riichi",
) -> str:
    value_pairs = _tile_list([tile for tile, value in counts.items() if value >= 2 and tile in value_honors])
    return _meld_policy(value_pairs, best_suit_count, simple_count, honor_count, terminal_count, pair_route, style)


def _call_policy(hand_tiles: list[str], config: MahjongCoachConfig, buttons: list[str]) -> str:
    tiles = [normalize_tile(tile) for tile in hand_tiles if normalize_tile(tile)]
    counts = Counter(tiles)
    value_honors = _value_honor_tiles(config)
    suit_counts = Counter(tile_suit(tile) for tile in tiles if tile_suit(tile))
    suited = {suit: count for suit, count in suit_counts.items() if suit in {"m", "p", "s"}}
    best_suit_count = max(suited.values(), default=0)
    honor_count = sum(1 for tile in tiles if is_honor(tile))
    terminal_count = sum(1 for tile in tiles if is_terminal(tile))
    simple_count = sum(1 for tile in tiles if is_simple(tile))
    pair_count = sum(1 for value in counts.values() if value >= 2)
    all_value_pairs = _tile_list([tile for tile, value in counts.items() if value >= 2 and tile in value_honors])
    call_buttons = set(_normalize_buttons(buttons))
    style = _play_style(config)

    active_value_pairs = all_value_pairs if (not call_buttons or call_buttons & {"pon", "kan"}) else ""
    return _meld_policy(active_value_pairs, best_suit_count, simple_count, honor_count, terminal_count, pair_count >= 4 and best_suit_count < 8, style, is_action_window=True)


def _riichi_advice(
    hand_tiles: list[str],
    config: MahjongCoachConfig | None = None,
    *,
    visible_tiles: list[str] | None = None,
    efficiency: dict[str, Any] | None = None,
) -> str:
    if efficiency is None:
        plan = build_round_plan(hand_tiles, config, visible_tiles=visible_tiles)
        efficiency = plan.get("efficiency", {})
    options = efficiency.get("discard_options")
    if not isinstance(options, list):
        return ""
    tenpai_options = [item for item in options if isinstance(item, dict) and int(item.get("shanten", 8)) == 0]
    if not tenpai_options:
        return ""
    best = tenpai_options[0]
    discard_tile = str(best.get("tile") or "")
    discard = _tile_name(discard_tile)
    wait_tiles, wait_count = _riichi_waits_after_discard(
        hand_tiles,
        [str(tile) for tile in best.get("effective_tiles", [])],
        discard_tile,
        visible_tiles or [],
    )
    wait_text = _tile_list_in_order(wait_tiles) or "待牌未明"
    effective_types = len(wait_tiles) or int(best.get("effective_types", 0))
    effective_count = wait_count if wait_tiles else int(best.get("effective_count", 0))
    quality = _wait_quality(effective_types, effective_count)
    style = _play_style(config)
    value_text = _riichi_value_text(hand_tiles, config)
    visible_text = "，已扣可见牌" if int(efficiency.get("visible_tile_count", 0)) else ""
    if style == "fast":
        if quality == "good" and effective_count >= 8:
            action = "推荐立直"
            reason = "好形/枚数够"
        elif quality == "good":
            action = "可以立直"
            reason = "待牌尚可"
        else:
            action = "谨慎立直"
            reason = "愚形或枚数少"
    else:
        if quality == "good":
            action = "推荐立直"
            reason = "好形/枚数够"
        elif quality == "medium":
            action = "可以立直"
            reason = "待牌尚可"
        else:
            action = "谨慎立直"
            reason = "愚形或枚数少"
    return f"{action}：打{discard}听{wait_text}，有效{effective_types}种{effective_count}枚{visible_text}；{reason}{value_text}；本地快判。"


def _riichi_waits_after_discard(
    hand_tiles: list[str],
    effective_tiles: list[str],
    discard_tile: str,
    visible_tiles: list[str],
) -> tuple[list[str], int]:
    hand_counts = Counter(_canonical_tile(tile) for tile in hand_tiles if _canonical_tile(tile) in TILE_TYPES)
    visible_counts = _visible_counts(visible_tiles)
    discarded = _canonical_tile(discard_tile)
    waits = [
        _canonical_tile(tile)
        for tile in effective_tiles
        if _canonical_tile(tile) in TILE_TYPES and _canonical_tile(tile) != discarded
    ]
    wait_count = sum(max(0, 4 - hand_counts.get(tile, 0) - visible_counts.get(tile, 0)) for tile in waits)
    return waits, wait_count


def _wait_quality(effective_types: int, effective_count: int) -> str:
    if effective_types >= 2 and effective_count >= 6:
        return "good"
    if effective_count >= 5:
        return "medium"
    return "poor"


def _riichi_value_text(hand_tiles: list[str], config: MahjongCoachConfig | None) -> str:
    tiles = [normalize_tile(tile) for tile in hand_tiles if normalize_tile(tile)]
    counts = Counter(tiles)
    value_pairs = _tile_list([tile for tile, value in counts.items() if value >= 2 and tile in _value_honor_tiles(config)])
    dora_tiles = _dora_tiles(config)
    dora_count = sum(1 for tile in tiles if tile in {"0m", "0p", "0s"} or _is_dora(tile, dora_tiles))
    parts: list[str] = []
    if dora_count:
        parts.append(f"有宝牌/红5 {dora_count}张")
    if value_pairs:
        parts.append(f"有役牌对子{value_pairs}")
    return f"，{'，'.join(parts)}" if parts else ""


def _route_discard_options(
    discard_tiles: list[str],
    counts: Counter[str],
    best_suit: str,
    best_suit_count: int,
    second_suit_count: int,
    pair_count: int,
    has_value_pair: bool,
    honor_count: int,
    terminal_count: int,
    simple_count: int,
) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    primary = _first_tile(discard_tiles)
    if not primary:
        return options

    if best_suit_count >= 8:
        primary_label = f"{SUIT_NAMES.get(best_suit, best_suit)}染"
    elif pair_count >= 4:
        primary_label = "七对"
    elif has_value_pair:
        primary_label = "役牌"
    elif best_suit_count >= 5 and best_suit_count >= second_suit_count + 2:
        primary_label = "主线"
    elif simple_count >= 9 and honor_count + terminal_count <= 3:
        primary_label = "断幺"
    elif honor_count >= 4:
        primary_label = "孤字先"
    else:
        primary_label = "牌效"
    options.append((primary_label, primary))

    alternate = _alternate_route_tile(discard_tiles, primary, counts, best_suit)
    if alternate:
        if best_suit_count >= 7:
            alternate_label = "不硬染"
        elif pair_count >= 4:
            alternate_label = "面子手"
        else:
            alternate_label = "保守"
        options.append((alternate_label, alternate))
    return options


def _route_options_text(options: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for label, tile in options[:2]:
        name = _tile_name(tile)
        if label and name:
            parts.append(f"{label}打{name}")
    return "；".join(parts)


def _first_tile(tiles: list[str]) -> str:
    for tile in tiles:
        normalized = normalize_tile(tile)
        if normalized:
            return normalized
    return ""


def _alternate_route_tile(discard_tiles: list[str], primary: str, counts: Counter[str], best_suit: str) -> str:
    normalized = [normalize_tile(tile) for tile in discard_tiles if normalize_tile(tile)]
    for tile in normalized:
        if tile != primary:
            return tile

    for tile, value in sorted(counts.items(), key=lambda item: _tile_sort_key(item[0])):
        if value >= 2 or tile == primary or tile in {"0m", "0p", "0s"}:
            continue
        if best_suit and tile_suit(tile) == best_suit and _has_neighbor(tile, counts, distance=2):
            continue
        return tile
    return ""


def _discard_score(
    tile: str,
    counts: Counter[str],
    best_suit: str,
    keep_set: set[str],
    value_honors: set[str],
    dora_tiles: set[str],
) -> tuple[int, int, str]:
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    rank_value = int(rank) if rank.isdigit() else 0
    keep_penalty = 20 if tile in keep_set else 0
    pair_penalty = 50 if counts[tile] >= 2 else 0
    dora_penalty = 35 if _is_dora(tile, dora_tiles) else 0
    value_penalty = 8 if tile in value_honors else 0
    if suit == "z":
        return (pair_penalty + keep_penalty + dora_penalty + value_penalty, rank_value, tile)

    connected = _has_neighbor(tile, counts, distance=1)
    near = _has_neighbor(tile, counts, distance=2)
    base = 8
    if not connected and is_terminal(tile):
        base = 1
    elif not connected:
        base = 3
    elif not near:
        base = 5

    if best_suit and suit != best_suit:
        base -= 1
    if best_suit and suit == best_suit:
        base += 2
    return (pair_penalty + keep_penalty + dora_penalty + base, rank_value, tile)


def _has_neighbor(tile: str, counts: Counter[str], *, distance: int) -> bool:
    suit = tile_suit(tile)
    if suit not in {"m", "p", "s"}:
        return False
    rank = tile_rank(tile)
    if not rank.isdigit():
        return False
    value = int(rank)
    for offset in range(1, distance + 1):
        if counts.get(f"{value - offset}{suit}", 0) > 0 or counts.get(f"{value + offset}{suit}", 0) > 0:
            return True
    return False


def _normalize_buttons(buttons: list[str] | None) -> list[str]:
    result: list[str] = []
    for button in buttons or []:
        normalized = str(button or "").strip().lower()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _coerce_turn(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None
