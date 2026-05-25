from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from .models import CoachDecision, MahjongCoachConfig, RoundCoachState
from .perception.action_detector import detect_action_buttons_fast
from .perception.fast_hand_path import FastHandResult, detect_fast_hand_path
from .perception.river_state import RiverStateResult, detect_river_state_path
from .tile_labels import hand_signature, is_honor, is_simple, is_terminal, normalize_tile, tile_rank, tile_suit


CRITICAL_BUTTONS = {"chi", "pon", "kan", "ron", "tsumo", "riichi"}
CALL_BUTTONS = {"chi", "pon", "kan"}
WIN_BUTTONS = {"ron", "tsumo"}
SUIT_NAMES = {"m": "万", "p": "筒", "s": "索", "z": "字"}
HONOR_NAMES = {"1z": "东", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "发", "7z": "中"}


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

    def reset_round(self, round_id: str = "default") -> RoundCoachState:
        self.state = RoundCoachState(round_id=round_id or "default")
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
            if hand_result.ok:
                self._remember_hand(hand_result)
            action_meta = {"source": "opening_hand_scan", "skipped": True}
            river_result = self._river_result_from_state("opening_skips_river_scan", ok_if_cached=False)
            if hand_result.ok:
                return self._opening_decision(hand_result, river_result, started, action_meta=action_meta)
            return self._observe_decision(hand_result, action_meta, river_result, started, phase="opening_hand_scan")

        buttons, action_meta = self._resolve_buttons(path, observed_buttons)
        critical = [button for button in buttons if button in CRITICAL_BUTTONS]
        win_buttons = [button for button in critical if button in WIN_BUTTONS]
        if self.config.critical_action_interrupts and win_buttons:
            river_result = self._river_result_from_state("action_window_uses_cached_river")
            return self._critical_decision(win_buttons, action_meta, started, river_result=river_result)

        hand_result = self._detect_hand(path)
        if hand_result.ok:
            self._remember_hand(hand_result)

        call_buttons = [button for button in critical if button in CALL_BUTTONS]
        riichi_buttons = [button for button in critical if button == "riichi"]
        if self.config.critical_action_interrupts and (call_buttons or riichi_buttons):
            river_result = self._river_result_from_state("action_window_uses_cached_river")
            return self._critical_decision(
                [*call_buttons, *riichi_buttons],
                action_meta,
                started,
                hand_result=hand_result,
                river_result=river_result,
            )

        if riichi_players:
            river_result = self._detect_river(path)
            if river_result.ok:
                self._remember_river(river_result)
            return self._defense_decision(riichi_players, hand_result, river_result, started)

        turn_number = _coerce_turn(self_turn_index)
        if hand_result.ok and self._checkpoint_due(turn_number, force_checkpoint=force_checkpoint):
            river_result = self._detect_river(path)
            if river_result.ok:
                self._remember_river(river_result)
            return self._checkpoint_decision(hand_result, river_result, turn_number, force_checkpoint, started)

        river_result = self._river_result_from_state("river_scan_not_due")
        return self._observe_decision(hand_result, action_meta, river_result, started, phase="normal_tracking")

    def _observe_decision(
        self,
        hand_result: FastHandResult,
        action_meta: dict[str, Any],
        river_result: RiverStateResult,
        started: float,
        *,
        phase: str,
    ) -> CoachDecision:
        self.state.round_phase = phase
        summary, detail, reason_codes = self._observe_message(hand_result)
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
            perception={"hand": hand_result.to_dict(), "action": action_meta, "river": river_result.to_dict()},
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

    def _detect_hand(self, path: Path | None) -> FastHandResult:
        if path is None:
            return FastHandResult(reason="image_path_missing")
        return detect_fast_hand_path(path, calibration_dir=self.calibration_dir)

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
                    "继续按当前主线推进；等三巡、手牌结构明显变化、或出现立直/和牌压力时再复盘。",
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
            detail = f"Hand scan accepted {accepted} tiles from {occupied} occupied-looking slots; waiting for 13 or 14 stable tiles."
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

    def _critical_decision(
        self,
        buttons: list[str],
        action_meta: dict[str, Any],
        started: float,
        hand_result: FastHandResult | None = None,
        river_result: RiverStateResult | None = None,
    ) -> CoachDecision:
        if any(button in WIN_BUTTONS for button in buttons):
            summary = "和牌窗口"
            suggestion = "看到荣和/自摸直接点，不需要等待策略分析。"
            decision_type = "win_window"
            priority = 100
        elif any(button in CALL_BUTTONS for button in buttons):
            summary = "吃碰杠窗口"
            suggestion = self._call_suggestion(hand_result)
            decision_type = "call_window"
            priority = 95
        elif "riichi" in buttons:
            summary = "立直窗口"
            suggestion = "可立直：用当前主线快速确认待牌和打点；不等待 LLM。"
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
            detail="动作窗口不等待 LLM；吃碰杠和立直只使用当前策略与本地快判。",
            suggestion=suggestion,
            buttons=list(buttons),
            hand_tiles=list(hand_result.hand_tiles) if hand_result else [],
            reason_codes=["critical_action_interrupt"],
            coach_state=self.state.to_dict(),
            perception={
                "action": action_meta,
                "hand": hand_result.to_dict() if hand_result else {},
                "river": river_result.to_dict() if river_result else {},
            },
            engine_meta=self._meta(started, decision_type),
        )

    def _call_suggestion(self, hand_result: FastHandResult | None) -> str:
        plan = self.state.current_plan or self.state.opening_plan
        if not plan and hand_result is not None and hand_result.ok:
            built = build_round_plan(hand_result.hand_tiles)
            plan = built["summary"]
            self.state.opening_emitted = True
            self.state.round_phase = "opening_strategy"
            self.state.opening_plan = plan
            self.state.current_plan = plan
            self.state.attack_defense_bias = built["bias"]
            self.state.target_shapes = list(built["targets"])
            self.state.caution_points = list(built["cautions"])
        if plan:
            return f"默认跳过，除非鸣牌能明确推进当前主线：{plan}"
        return "默认跳过；只有役牌对子、直接进听、明显加速主线或安全和牌时才吃碰杠。"

    def _opening_decision(
        self,
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        started: float,
        *,
        action_meta: dict[str, Any] | None = None,
    ) -> CoachDecision:
        plan = build_round_plan(hand_result.hand_tiles)
        self.state.opening_emitted = True
        self.state.round_phase = "opening_strategy"
        self.state.opening_plan = plan["summary"]
        self.state.current_plan = plan["summary"]
        self.state.attack_defense_bias = plan["bias"]
        self.state.target_shapes = list(plan["targets"])
        self.state.caution_points = list(plan["cautions"])
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
            perception={"hand": hand_result.to_dict(), "action": action_meta or {}, "river": river_result.to_dict()},
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
    ) -> CoachDecision:
        plan = build_round_plan(hand_result.hand_tiles)
        if turn_number is not None:
            self.state.last_checkpoint_self_turn = turn_number
        self.state.round_phase = "checkpoint_strategy"
        self.state.current_plan = plan["summary"]
        self.state.attack_defense_bias = plan["bias"]
        self.state.target_shapes = list(plan["targets"])
        self.state.caution_points = list(plan["cautions"])
        self.state.last_update_reason = "forced_checkpoint" if force_checkpoint else "scheduled_checkpoint"
        self.state.update_count += 1
        return CoachDecision(
            decision_type="coach_checkpoint",
            priority=50,
            summary="Round checkpoint updated",
            detail=plan["detail"],
            suggestion=plan["summary"],
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=[self.state.last_update_reason],
            coach_state=self.state.to_dict(),
            perception={"hand": hand_result.to_dict(), "river": river_result.to_dict()},
            engine_meta=self._meta(started, "coach_checkpoint"),
        )

    def _defense_decision(
        self,
        riichi_players: list[str],
        hand_result: FastHandResult,
        river_result: RiverStateResult,
        started: float,
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
            suggestion=self._defense_suggestion(riichi_players, river_result),
            hand_tiles=list(hand_result.hand_tiles),
            reason_codes=["riichi_players_present"],
            coach_state=self.state.to_dict(),
            perception={"hand": hand_result.to_dict(), "river": river_result.to_dict()},
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
        }

    def _defense_suggestion(self, riichi_players: list[str], river_result: RiverStateResult) -> str:
        piles = river_result.discard_piles if river_result.ok else self.state.last_discard_piles
        safe_tiles: list[str] = []
        for player in riichi_players:
            key = _player_key(player)
            if not key:
                continue
            safe_tiles.extend(str(item.get("tile") or "") for item in piles.get(key, []))
        if safe_tiles:
            safe_text = _tile_list_in_order([tile for tile in safe_tiles if tile][-8:])
            return f"防守优先：先看立直家现物 {safe_text}；没有现物再考虑筋/壁，宝牌周边先保守。"
        if river_result.ok and river_result.visible_tiles:
            visible_text = _tile_list_in_order(river_result.visible_tiles[-10:])
            return f"牌河已识别，可见弃牌参考 {visible_text}；未标定立直家座位时先保守找现物。"
        return "Slow down and prefer safe tiles from visible information."


def build_round_plan(hand_tiles: list[str]) -> dict[str, Any]:
    tiles = [normalize_tile(tile) for tile in hand_tiles if normalize_tile(tile)]
    counts = Counter(tiles)
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
    cleanup_tiles = _cleanup_candidates(tiles, counts, best_suit)
    keep_tiles = _keep_candidates(tiles, counts, best_suit)
    discard_tiles = _discard_candidates(tiles, counts, best_suit, cleanup_tiles, keep_tiles)
    discard_text = _tile_list_in_order(discard_tiles)
    route_options = _route_discard_options(
        discard_tiles,
        counts,
        best_suit,
        best_suit_count,
        second_suit_count,
        pair_count,
        honor_count,
        terminal_count,
        simple_count,
    )
    route_text = _route_options_text(route_options)
    cleanup_text = _tile_list_in_order(cleanup_tiles) or discard_text or "孤张字牌和远张"
    keep_text = _tile_list(keep_tiles) or "连续搭子和中张"
    suit_text = _suit_breakdown(tiles)
    best_shape = _suit_shape(tiles, best_suit) if best_suit else ""

    targets: list[str] = []
    cautions: list[str] = []
    if best_suit_count >= 8:
        targets.append(f"主线：{SUIT_NAMES.get(best_suit, best_suit)}子清一色/混一色倾向")
        summary = f"主线：{SUIT_NAMES.get(best_suit, best_suit)}子占比很高，保留同色块，先清{cleanup_text}。"
    elif best_suit_count >= 5 and best_suit_count >= second_suit_count + 2:
        targets.append(f"主线：围绕{SUIT_NAMES.get(best_suit, best_suit)}子 {best_shape} 推进")
        summary = f"主线：围绕{SUIT_NAMES.get(best_suit, best_suit)}子 {best_shape} 做搭子，先清{cleanup_text}。"
    elif simple_count >= 9 and honor_count + terminal_count <= 3:
        targets.extend(["主线：断幺/平和速度", f"保留：{keep_text}"])
        summary = f"主线：断幺/平和速度手，保留{keep_text}，别贪孤张字牌。"
    elif pair_count >= 4:
        targets.append(f"副线：七对子，已有对子 {_tile_list(pair_tiles)}")
        summary = f"主线：对子价值不错，保留对子；如果后续继续成对再转七对。"
    elif honor_count >= 4:
        targets.append("主线：先清孤字牌")
        summary = f"主线：字牌偏多，未成对的先清；只保留役牌对子或安全价值。"
    else:
        targets.append("主线：牌效推进")
        summary = f"主线：牌效推进，保留{keep_text}，先清{cleanup_text}。"

    if keep_text:
        targets.append(f"保留：{keep_text}")
    if pair_tiles:
        targets.append(f"对子：{_tile_list(pair_tiles)}")

    if honor_count >= 3:
        cautions.append("孤字牌不要久留，除非成对或有役牌价值。")
    if terminal_count >= 4:
        cautions.append("孤幺九可清，已经组成边搭/对子再保留。")
    if best_suit_count >= 7:
        cautions.append("有染手分支，但摸到强中张时不要硬染。")
    if route_text:
        cautions.append(f"路线选择：{route_text}")
    if cleanup_tiles:
        cautions.append(f"优先清理：{cleanup_text}")
    cautions.append("吃碰杠：默认跳过；只有役牌对子、直接进听、或明显加速主线才开口。")
    if not cautions:
        cautions.append("三巡后复盘，不要每摸一张就推翻主线。")

    bias = "attack" if simple_count >= 8 or best_suit_count >= 5 else "neutral"
    detail = (
        f"结构：{suit_text}；对子 {pair_count} 组。"
        f" 当前先保留 {keep_text}，{('路线选择：' + route_text) if route_text else ('候选打牌 ' + (discard_text or cleanup_text))}。"
    )
    return {
        "summary": summary,
        "detail": detail,
        "bias": bias,
        "targets": targets,
        "cautions": cautions,
    }


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


def _player_key(value: Any) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower()
    aliases = {
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
    return aliases.get(lowered, raw if raw in {"self", "left_opponent", "top_opponent", "right_opponent"} else "")


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


def _keep_candidates(tiles: list[str], counts: Counter[str], best_suit: str) -> list[str]:
    keep: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if counts[normalized] >= 2 or normalized in {"0m", "0p", "0s"}:
            keep.append(normalized)
            continue
        if tile_suit(normalized) == best_suit and _has_neighbor(normalized, counts, distance=2):
            keep.append(normalized)
    return keep


def _cleanup_candidates(tiles: list[str], counts: Counter[str], best_suit: str) -> list[str]:
    cleanup: list[str] = []
    for tile in sorted((normalize_tile(tile) for tile in tiles if normalize_tile(tile)), key=_tile_sort_key):
        if counts[tile] >= 2 or tile in {"0m", "0p", "0s"}:
            continue
        suit = tile_suit(tile)
        if suit == "z":
            cleanup.append(tile)
        elif suit != best_suit and not _has_neighbor(tile, counts, distance=1):
            cleanup.append(tile)
        elif suit == best_suit and is_terminal(tile) and not _has_neighbor(tile, counts, distance=1):
            cleanup.append(tile)
    return sorted(cleanup, key=lambda tile: _cleanup_score(tile, counts, best_suit))[:5]


def _cleanup_score(tile: str, counts: Counter[str], best_suit: str) -> tuple[int, int, str]:
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    rank_value = int(rank) if rank.isdigit() else 0
    if suit == "z":
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
) -> list[str]:
    selected: list[str] = []
    keep_set = set(keep_tiles)
    for tile in sorted(cleanup_tiles, key=lambda tile: _discard_score(tile, counts, best_suit, keep_set)):
        if tile not in selected:
            selected.append(tile)

    pool: list[str] = []
    for tile in tiles:
        normalized = normalize_tile(tile)
        if not normalized or normalized in selected or normalized in {"0m", "0p", "0s"}:
            continue
        if counts[normalized] >= 2:
            continue
        pool.append(normalized)

    pool = sorted(set(pool), key=lambda tile: _discard_score(tile, counts, best_suit, keep_set))
    for tile in pool:
        if len(selected) >= 5:
            break
        selected.append(tile)
    return selected[:4]


def _route_discard_options(
    discard_tiles: list[str],
    counts: Counter[str],
    best_suit: str,
    best_suit_count: int,
    second_suit_count: int,
    pair_count: int,
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
    elif best_suit_count >= 5 and best_suit_count >= second_suit_count + 2:
        primary_label = "主线"
    elif simple_count >= 9 and honor_count + terminal_count <= 3:
        primary_label = "断幺"
    elif honor_count >= 4:
        primary_label = "孤字先"
    elif pair_count >= 4:
        primary_label = "七对"
    else:
        primary_label = "牌效"
    options.append((primary_label, primary))

    alternate = _alternate_route_tile(discard_tiles, primary, counts, best_suit)
    if alternate:
        if best_suit_count >= 7:
            alternate_label = "不硬染"
        elif pair_count >= 4:
            alternate_label = "牌效"
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
        if tile != primary and tile_suit(tile) != "z":
            return tile
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


def _discard_score(tile: str, counts: Counter[str], best_suit: str, keep_set: set[str]) -> tuple[int, int, str]:
    suit = tile_suit(tile)
    rank = tile_rank(tile)
    rank_value = int(rank) if rank.isdigit() else 0
    keep_penalty = 20 if tile in keep_set else 0
    pair_penalty = 50 if counts[tile] >= 2 else 0
    if suit == "z":
        return (pair_penalty + keep_penalty, rank_value, tile)

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
    return (pair_penalty + keep_penalty + base, rank_value, tile)


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
