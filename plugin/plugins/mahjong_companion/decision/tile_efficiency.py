from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

from ..contracts import MahjongAnalysis, PerceivedGameState
from ..tile_labels import (
    dedupe as _dedupe,
    format_tile_label,
    normalize_tile as _normalize_tile,
    normalize_tile_list as _normalize_tile_list,
    normalize_tile_set as _normalize_tile_set,
    replace_tile_codes_in_text,
)
from .mahjong_analysis import attach_confidence_metadata
from .risk_estimator import estimate_defense_alerts, normalize_riichi_player
from .utils import coerce_int, meld_group_count

OPPONENT_PLAYERS = {"left_opponent", "top_opponent", "right_opponent"}
_UNKNOWN_SHANTEN_SORT_VALUE = 99
_SAFETY_SORT_VALUE = {
    "genbutsu": 4.0,
    "dead": 3.6,
    "suji": 2.6,
    "high": 2.0,
    "medium": 1.0,
    "low": 0.0,
    "unknown": 0.0,
}


def build_mahjong_analysis(
    state: PerceivedGameState,
    *,
    recommended_focus: str = "",
    review_tags: list[str] | None = None,
) -> MahjongAnalysis:
    hints = _state_analysis_hints(state)
    if _has_structured_tile_input(state, hints):
        analysis = _build_structured_analysis(
            state,
            hints=hints,
            recommended_focus=recommended_focus,
            review_tags=review_tags or [],
        )
        return attach_confidence_metadata(analysis, state=state, hints=hints)
    analysis = _build_fallback_analysis(
        state,
        recommended_focus=recommended_focus,
        review_tags=review_tags or [],
    )
    return attach_confidence_metadata(analysis, state=state, hints=hints)


def _state_analysis_hints(state: PerceivedGameState) -> dict[str, Any]:
    hints = dict(state.analysis_hints) if isinstance(state.analysis_hints, dict) else {}
    if state.visible_tiles and not hints.get("visible_tiles"):
        hints["visible_tiles"] = list(state.visible_tiles)
    if state.known_genbutsu_tiles and not any(
        hints.get(key) for key in ("genbutsu_tiles", "known_genbutsu_tiles", "confirmed_safe_tiles")
    ):
        hints["known_genbutsu_tiles"] = list(state.known_genbutsu_tiles)
    return hints


def _has_structured_tile_input(state: PerceivedGameState, hints: dict[str, Any]) -> bool:
    if state.hand_tiles:
        return True
    if hints.get("tile_level_available"):
        return True
    if isinstance(hints.get("candidate_discards"), list) and hints.get("candidate_discards"):
        return True
    if hints.get("shanten_estimate") is not None:
        return True
    return False


def _build_structured_analysis(
    state: PerceivedGameState,
    *,
    hints: dict[str, Any],
    recommended_focus: str,
    review_tags: list[str],
) -> MahjongAnalysis:
    analysis_version = str(hints.get("analysis_version", "mahjong-lite-v2")).strip() or "mahjong-lite-v2"
    candidate_discards = _normalize_candidate_discards(hints.get("candidate_discards"))
    if not candidate_discards and state.hand_tiles:
        candidate_discards = _estimate_candidate_discards(
            state.hand_tiles,
            dora_indicators=state.dora_indicators,
            hints=hints,
            state=state,
        )

    shanten_estimate = coerce_int(hints.get("shanten_estimate"))
    if shanten_estimate is None and state.hand_tiles:
        shanten_estimate = _estimate_open_hand_shanten(state.hand_tiles, meld_group_count(state))

    ukeire_estimate = coerce_int(hints.get("ukeire_estimate"))
    if ukeire_estimate is None:
        ukeire_estimate = _estimate_ukeire(candidate_discards)

    bias = str(hints.get("attack_defense_bias", "")).strip()
    if not bias:
        bias = _derive_attack_defense_bias(
            state=state,
            shanten_estimate=shanten_estimate,
            candidate_discards=candidate_discards,
        )

    teaching_points = _build_teaching_points(
        state=state,
        recommended_focus=recommended_focus,
        review_tags=review_tags,
        bias=bias,
        candidate_discards=candidate_discards,
        shanten_estimate=shanten_estimate,
        hints=hints,
    )
    defense_alerts = estimate_defense_alerts(
        state,
        candidate_discards=candidate_discards,
        shanten_estimate=shanten_estimate,
        attack_defense_bias=bias,
        hints=hints,
    )

    hand_shape_confidence = _coerce_float(hints.get("hand_shape_confidence"))
    if hand_shape_confidence is None:
        hand_shape_confidence = 0.72 if len(state.hand_tiles) >= 13 else 0.58

    tile_level_available = bool(candidate_discards or state.hand_tiles or shanten_estimate is not None)
    tile_level_state = "tile_level_reliable" if candidate_discards else "tile_level_partial"

    return MahjongAnalysis(
        analysis_version=analysis_version,
        tile_level_available=tile_level_available,
        tile_level_state=tile_level_state,
        analysis_confidence=float(hints.get("analysis_confidence", 0.72) or 0.72),
        hand_shape_confidence=hand_shape_confidence,
        shanten_estimate=shanten_estimate,
        ukeire_estimate=ukeire_estimate,
        candidate_discards=candidate_discards,
        attack_defense_bias=bias,
        defense_alerts=defense_alerts,
        teaching_points=teaching_points,
    )


def _build_fallback_analysis(
    state: PerceivedGameState,
    *,
    recommended_focus: str,
    review_tags: list[str],
) -> MahjongAnalysis:
    tags = set(review_tags)

    teaching_points: list[str] = []
    attack_defense_bias = "neutral"

    if recommended_focus == "win_confirmation":
        teaching_points.append("这一刻先确认和牌语义，别让高价值窗口从眼前滑过去。")
        attack_defense_bias = "attack"
    elif recommended_focus == "riichi_decision":
        teaching_points.append("立直窗口更像路线选择点，先确认现在是继续进攻还是先收一手。")
        attack_defense_bias = "slightly_attack"
    elif recommended_focus in {"kan_decision", "call_decision"}:
        teaching_points.append("副露或开杠会改变路线，这时更适合先看节奏而不是立刻点下去。")
        attack_defense_bias = "slightly_defensive"
    elif recommended_focus in {"dialog_confirmation", "confirm_or_skip"}:
        teaching_points.append("先看清按钮语义，再决定是否继续，不要把确认窗口当作普通过渡帧。")
    elif recommended_focus == "turn_observe":
        teaching_points.append("这巡更适合先观察摸牌、牌河和副露信息，再决定后续方向。")
    elif recommended_focus == "replay_observe":
        teaching_points.append("回放里先盯关键节点，后面更适合再组织成一段复盘摘要。")
    else:
        teaching_points.append("当前先按规则焦点做提醒，牌级建议尚未启用。")

    if "low_confidence" in tags or state.confidence < 0.45:
        teaching_points.append("当前识别置信度偏低，这类建议更适合结合截图二次确认。")
    defense_alerts = estimate_defense_alerts(
        state,
        candidate_discards=[],
        shanten_estimate=None,
        attack_defense_bias=attack_defense_bias,
        hints={},
    )

    return MahjongAnalysis(
        analysis_version="mahjong-lite-v1",
        tile_level_available=False,
        tile_level_state="tile_level_unavailable",
        analysis_confidence=0.0,
        hand_shape_confidence=0.0,
        shanten_estimate=None,
        ukeire_estimate=None,
        candidate_discards=[],
        attack_defense_bias=attack_defense_bias,
        defense_alerts=defense_alerts,
        teaching_points=teaching_points,
    )


def _normalize_candidate_discards(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tile = str(item.get("tile", "")).strip()
        if not tile:
            continue
        candidate = {
            "tile": tile,
            "score": round(_coerce_float(item.get("score")) or 0.0, 3),
            "ukeire_estimate": coerce_int(item.get("ukeire_estimate")),
            "safety_hint": str(item.get("safety_hint", "unknown")).strip() or "unknown",
            "reason": str(item.get("reason", "")).strip(),
        }
        for key in ("strategy_score", "defense_score", "dora_value"):
            value = _coerce_float(item.get(key))
            if value is not None:
                candidate[key] = round(value, 3)
        wait_quality_bonus = coerce_int(item.get("wait_quality_bonus"))
        if wait_quality_bonus is not None:
            candidate["wait_quality_bonus"] = wait_quality_bonus
        strategy_mode = str(item.get("strategy_mode", "")).strip()
        if strategy_mode:
            candidate["strategy_mode"] = strategy_mode
        current_shanten = coerce_int(item.get("current_shanten"))
        if current_shanten is not None:
            candidate["current_shanten"] = current_shanten
        post_discard_shanten = coerce_int(item.get("post_discard_shanten"))
        if post_discard_shanten is not None:
            candidate["post_discard_shanten"] = post_discard_shanten
        source = str(item.get("source", "")).strip()
        if source:
            candidate["source"] = source
        normalized.append(candidate)
    return normalized[:5]


def _estimate_candidate_discards(
    hand_tiles: list[str],
    *,
    dora_indicators: list[str],
    hints: dict[str, Any] | None = None,
    state: PerceivedGameState | None = None,
) -> list[dict[str, Any]]:
    hints = hints if isinstance(hints, dict) else {}
    counts = Counter(_normalize_tile(tile) for tile in hand_tiles if _normalize_tile(tile))
    if not counts:
        return []

    dora_tiles = _derive_dora_tiles(dora_indicators)
    genbutsu_tiles = _normalize_tile_set(
        hints.get("genbutsu_tiles")
        or hints.get("known_genbutsu_tiles")
        or hints.get("confirmed_safe_tiles"),
    )
    visible_tiles = _normalize_tile_list(
        hints.get("visible_tiles")
        or hints.get("dead_tiles")
        or hints.get("known_visible_tiles"),
    )
    deck_state_trusted = _deck_state_trusted(hints, visible_tiles)
    open_melds = coerce_int(hints.get("recognized_meld_group_count")) or 0
    remaining_counts = _remaining_tile_counts(counts, visible_tiles if deck_state_trusted else [])

    current_shanten = _estimate_shanten_from_counts(counts, open_melds)
    at_tenpai_edge = current_shanten is not None and current_shanten <= 1
    has_riichi_pressure = _has_opponent_riichi(state=state, hints=hints)
    strategy_mode = _discard_strategy_mode(
        has_riichi_pressure=has_riichi_pressure,
        shanten_estimate=current_shanten,
        hints=hints,
    )

    candidates: list[dict[str, Any]] = []
    for tile in counts:
        post_discard_shanten = _estimate_post_discard_shanten_from_counts(counts, tile, open_melds)
        real_ukeire = _calculate_discard_ukeire_from_counts(counts, tile, remaining_counts, open_melds)
        raw_ukeire = real_ukeire if real_ukeire is not None else _raw_ukeire_score(tile, counts)
        wait_bonus = 0
        if at_tenpai_edge:
            wait_bonus = _tenpai_wait_quality_bonus(counts, tile)
            if real_ukeire is None:
                raw_ukeire = max(0, raw_ukeire + wait_bonus)
        score = _raw_discard_score(tile, counts, dora_tiles)
        safety_hint = _defensive_safety_hint(
            tile,
            counts=counts,
            visible_tiles=visible_tiles,
            genbutsu_tiles=genbutsu_tiles,
            has_riichi_pressure=has_riichi_pressure,
        )
        dora_value = _dora_value(tile, dora_tiles)
        defense_score = _discard_defense_score(safety_hint=safety_hint, raw_score=score, dora_value=dora_value)
        strategy_score = _discard_strategy_score(
            strategy_mode=strategy_mode,
            ukeire=raw_ukeire,
            raw_score=score,
            safety_hint=safety_hint,
            dora_value=dora_value,
            wait_quality_bonus=wait_bonus,
        )
        candidates.append(
            {
                "tile": tile,
                "raw_score": score,
                "strategy_score": strategy_score,
                "defense_score": defense_score,
                "dora_value": dora_value,
                "strategy_mode": strategy_mode,
                "wait_quality_bonus": wait_bonus,
                "ukeire_estimate": raw_ukeire,
                "current_shanten": current_shanten,
                "post_discard_shanten": post_discard_shanten,
                "safety_hint": safety_hint,
                "reason": _reason_for_tile(
                    tile,
                    counts,
                    dora_tiles,
                    uses_visible_tiles=deck_state_trusted,
                    visible_tile_count=len(visible_tiles),
                ),
            }
        )

    _rank_discard_candidates(candidates)
    if not candidates:
        return []

    highest = max(_candidate_recommendation_score(item) for item in candidates)
    lowest = min(_candidate_recommendation_score(item) for item in candidates)
    span = highest - lowest
    normalized: list[dict[str, Any]] = []
    for item in candidates[:3]:
        strategy_score = _candidate_output_score(item)
        recommendation_score = _candidate_recommendation_score(item)
        score = 0.0 if span <= 0 else (recommendation_score - lowest) / span
        reason = item["reason"]
        if deck_state_trusted:
            reason = _visible_tile_candidate_reason(item, visible_tile_count=len(visible_tiles))
        else:
            reason = _candidate_metrics_reason(item, reason)
        normalized.append(
            {
                "tile": item["tile"],
                "score": round(score, 3),
                "strategy_score": round(strategy_score, 3),
                "defense_score": round(_coerce_float(item.get("defense_score")) or 0.0, 3),
                "dora_value": round(_coerce_float(item.get("dora_value")) or 0.0, 3),
                "strategy_mode": item["strategy_mode"],
                "wait_quality_bonus": item["wait_quality_bonus"],
                "ukeire_estimate": item["ukeire_estimate"],
                "current_shanten": item["current_shanten"],
                "post_discard_shanten": item["post_discard_shanten"],
                "safety_hint": item["safety_hint"],
                "reason": reason,
            }
        )
    return normalized


def build_incremental_draw_candidates(
    waiting_hand_tiles: list[str],
    drawn_tile: str,
    cached_candidate_discards: list[dict[str, Any]],
    *,
    dora_indicators: list[str] | None = None,
    hints: dict[str, Any] | None = None,
    riichi_players: list[str] | None = None,
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    """Rank the pre-turn discard plan against the newly drawn tile.

    This intentionally evaluates only the cached candidates plus the draw. It
    keeps the hot path small when the user's discard clock starts.
    """

    hints = hints if isinstance(hints, dict) else {}
    dora_indicators = dora_indicators or []
    normalized_waiting_hand = [_normalize_tile(tile) for tile in waiting_hand_tiles]
    normalized_waiting_hand = [tile for tile in normalized_waiting_hand if tile]
    normalized_drawn_tile = _normalize_tile(drawn_tile)
    if not normalized_waiting_hand or not normalized_drawn_tile:
        return []

    hand_counts = Counter(normalized_waiting_hand)
    hand_counts[normalized_drawn_tile] += 1
    cached_candidates = _normalize_candidate_discards(cached_candidate_discards)
    if not cached_candidates:
        return []

    candidate_sources: dict[str, dict[str, Any]] = {}
    for item in cached_candidates[: max(1, max_candidates)]:
        tile = _normalize_tile(item.get("tile"))
        if not tile or hand_counts.get(tile, 0) <= 0:
            continue
        candidate_sources.setdefault(
            tile,
            {
                "source": "preturn_cached",
                "reason": str(item.get("reason", "")).strip(),
                "safety_hint": str(item.get("safety_hint", "")).strip(),
            },
        )
    candidate_sources[normalized_drawn_tile] = {
        "source": "drawn_tile",
        "reason": _reason_for_tile(normalized_drawn_tile, hand_counts, _derive_dora_tiles(dora_indicators)),
        "safety_hint": "",
    }

    if not candidate_sources:
        return []

    dora_tiles = _derive_dora_tiles(dora_indicators)
    genbutsu_tiles = _normalize_tile_set(
        hints.get("genbutsu_tiles")
        or hints.get("known_genbutsu_tiles")
        or hints.get("confirmed_safe_tiles"),
    )
    visible_tiles = _normalize_tile_list(
        hints.get("visible_tiles")
        or hints.get("dead_tiles")
        or hints.get("known_visible_tiles"),
    )
    deck_state_trusted = _deck_state_trusted(hints, visible_tiles)
    open_melds = coerce_int(hints.get("recognized_meld_group_count")) or 0
    remaining_counts = _remaining_tile_counts(hand_counts, visible_tiles if deck_state_trusted else [])

    current_shanten = _estimate_shanten_from_counts(hand_counts, open_melds)
    at_tenpai_edge = current_shanten is not None and current_shanten <= 1
    has_riichi_pressure = _has_opponent_riichi(riichi_players=riichi_players, hints=hints)
    strategy_mode = _discard_strategy_mode(
        has_riichi_pressure=has_riichi_pressure,
        shanten_estimate=current_shanten,
        hints=hints,
    )
    candidates: list[dict[str, Any]] = []
    for tile, source in candidate_sources.items():
        post_discard_shanten = _estimate_post_discard_shanten_from_counts(hand_counts, tile, open_melds)
        real_ukeire = _calculate_discard_ukeire_from_counts(hand_counts, tile, remaining_counts, open_melds)
        raw_score = _raw_discard_score(tile, hand_counts, dora_tiles)
        raw_ukeire = real_ukeire if real_ukeire is not None else _raw_ukeire_score(tile, hand_counts)
        wait_bonus = 0
        if at_tenpai_edge:
            wait_bonus = _tenpai_wait_quality_bonus(hand_counts, tile)
            if real_ukeire is None:
                raw_ukeire = max(0, raw_ukeire + wait_bonus)
        safety_hint = _defensive_safety_hint(
            tile,
            counts=hand_counts,
            visible_tiles=visible_tiles,
            genbutsu_tiles=genbutsu_tiles,
            has_riichi_pressure=has_riichi_pressure,
        )
        if safety_hint == "unknown":
            safety_hint = str(source.get("safety_hint", "")).strip() or "unknown"
        dora_value = _dora_value(tile, dora_tiles)
        defense_score = _discard_defense_score(safety_hint=safety_hint, raw_score=raw_score, dora_value=dora_value)
        strategy_score = _discard_strategy_score(
            strategy_mode=strategy_mode,
            ukeire=raw_ukeire,
            raw_score=raw_score,
            safety_hint=safety_hint,
            dora_value=dora_value,
            wait_quality_bonus=wait_bonus,
        )
        candidates.append(
            {
                "tile": tile,
                "raw_score": raw_score,
                "strategy_score": strategy_score,
                "defense_score": defense_score,
                "dora_value": dora_value,
                "strategy_mode": strategy_mode,
                "wait_quality_bonus": wait_bonus,
                "ukeire_estimate": raw_ukeire,
                "current_shanten": current_shanten,
                "post_discard_shanten": post_discard_shanten,
                "safety_hint": safety_hint,
                "reason": str(source.get("reason", "")).strip() or _reason_for_tile(
                    tile,
                    hand_counts,
                    dora_tiles,
                    uses_visible_tiles=deck_state_trusted,
                    visible_tile_count=len(visible_tiles),
                ),
                "source": source.get("source", "preturn_cached"),
            }
        )

    _rank_discard_candidates(candidates)

    highest = max(_candidate_recommendation_score(item) for item in candidates)
    lowest = min(_candidate_recommendation_score(item) for item in candidates)
    span = highest - lowest
    normalized: list[dict[str, Any]] = []
    for item in candidates[: max(1, max_candidates)]:
        strategy_score = _candidate_output_score(item)
        recommendation_score = _candidate_recommendation_score(item)
        score = 0.0 if span <= 0 else (recommendation_score - lowest) / span
        reason = item["reason"]
        if deck_state_trusted:
            reason = _visible_tile_candidate_reason(item, visible_tile_count=len(visible_tiles))
        else:
            reason = _candidate_metrics_reason(item, reason)
        normalized.append(
            {
                "tile": item["tile"],
                "score": round(score, 3),
                "strategy_score": round(strategy_score, 3),
                "defense_score": round(_coerce_float(item.get("defense_score")) or 0.0, 3),
                "dora_value": round(_coerce_float(item.get("dora_value")) or 0.0, 3),
                "strategy_mode": item["strategy_mode"],
                "wait_quality_bonus": item["wait_quality_bonus"],
                "ukeire_estimate": item["ukeire_estimate"],
                "current_shanten": item["current_shanten"],
                "post_discard_shanten": item["post_discard_shanten"],
                "safety_hint": item["safety_hint"],
                "reason": reason,
                "source": item["source"],
            }
        )
    return normalized


def _estimate_shanten(hand_tiles: list[str]) -> int | None:
    normalized = [_normalize_tile(tile) for tile in hand_tiles]
    normalized = [tile for tile in normalized if tile]
    if len(normalized) < 5:
        return None

    best = _estimate_raw_shanten_from_counts(_counter_to_tile_counts(Counter(normalized)))
    return max(0, min(8, best))


def _estimate_open_hand_shanten(hand_tiles: list[str], meld_count: int) -> int | None:
    normalized = [_normalize_tile(tile) for tile in hand_tiles]
    normalized = [tile for tile in normalized if tile]
    if len(normalized) < 2:
        return None
    meld_count = max(0, min(4, int(meld_count or 0)))
    if meld_count <= 0:
        return _estimate_shanten(normalized)
    best = _estimate_standard_shanten_with_open_melds(tuple(_counter_to_tile_counts(Counter(normalized))), meld_count)
    return max(0, min(8, best))


def _estimate_shanten_from_counts(counts: Counter[str], open_melds: int = 0) -> int | None:
    best = _estimate_unclamped_shanten_from_counts(counts, open_melds)
    if best is None:
        return None
    return max(0, min(8, best))


def _estimate_unclamped_shanten_from_counts(counts: Counter[str], open_melds: int = 0) -> int | None:
    if not counts:
        return None
    if open_melds > 0:
        best = _estimate_standard_shanten_with_open_melds(tuple(_counter_to_tile_counts(counts)), open_melds)
    else:
        best = _estimate_raw_shanten_from_counts(_counter_to_tile_counts(counts))
    return min(8, best)


def _estimate_raw_shanten_from_counts(tile_counts: list[int]) -> int:
    standard = _estimate_standard_shanten(tuple(tile_counts))
    special = [
        _estimate_chiitoi_shanten(tile_counts),
        _estimate_kokushi_shanten(tile_counts),
    ]
    return min([standard, *special])


# Lifted to module level so @lru_cache survives across estimator calls.
# Previously these were inner closures, which means each outer call rebuilt the
# function and its cache — fast-poll path saw ~476 cold-start recursions per frame
# (CODE_REVIEW_v1.2 N-H1).
@lru_cache(maxsize=131072)
def _taatsu_search(state: tuple[int, ...]) -> int:
    first = _first_non_empty_index(state)
    if first is None:
        return 0

    best = 0
    current = list(state)
    current[first] -= 1
    best = max(best, _taatsu_search(tuple(current)))

    if state[first] >= 2:
        next_state = list(state)
        next_state[first] -= 2
        best = max(best, 1 + _taatsu_search(tuple(next_state)))

    if _is_suited_index(first):
        suit_pos = first % 9
        if suit_pos <= 7 and state[first + 1] > 0:
            next_state = list(state)
            next_state[first] -= 1
            next_state[first + 1] -= 1
            best = max(best, 1 + _taatsu_search(tuple(next_state)))
        if suit_pos <= 6 and state[first + 2] > 0:
            next_state = list(state)
            next_state[first] -= 1
            next_state[first + 2] -= 1
            best = max(best, 1 + _taatsu_search(tuple(next_state)))
    return min(4, best)


def _estimate_taatsu(counts: Counter[str]) -> int:
    tile_counts = tuple(_counter_to_tile_counts(counts))
    return _taatsu_search(tile_counts)


def _estimate_standard_shanten(initial_counts: tuple[int, ...]) -> int:
    return _estimate_standard_shanten_with_open_melds(initial_counts, 0)


@lru_cache(maxsize=131072)
def _standard_shanten_search(
    state: tuple[int, ...],
    mentsu: int,
    taatsu: int,
    pair: int,
) -> int:
    first = _first_non_empty_index(state)
    if first is None:
        usable_taatsu = min(taatsu, 4 - mentsu)
        return 8 - mentsu * 2 - usable_taatsu - pair

    best = 8

    current = list(state)
    current[first] -= 1
    best = min(best, _standard_shanten_search(tuple(current), mentsu, taatsu, pair))

    if state[first] >= 3:
        next_state = list(state)
        next_state[first] -= 3
        best = min(best, _standard_shanten_search(tuple(next_state), mentsu + 1, taatsu, pair))

    if _is_suited_index(first) and first % 9 <= 6 and state[first + 1] > 0 and state[first + 2] > 0:
        next_state = list(state)
        next_state[first] -= 1
        next_state[first + 1] -= 1
        next_state[first + 2] -= 1
        best = min(best, _standard_shanten_search(tuple(next_state), mentsu + 1, taatsu, pair))

    if state[first] >= 2:
        next_state = list(state)
        next_state[first] -= 2
        if pair == 0:
            best = min(best, _standard_shanten_search(tuple(next_state), mentsu, taatsu, 1))
        best = min(best, _standard_shanten_search(tuple(next_state), mentsu, taatsu + 1, pair))

    if _is_suited_index(first):
        suit_pos = first % 9
        if suit_pos <= 7 and state[first + 1] > 0:
            next_state = list(state)
            next_state[first] -= 1
            next_state[first + 1] -= 1
            best = min(best, _standard_shanten_search(tuple(next_state), mentsu, taatsu + 1, pair))
        if suit_pos <= 6 and state[first + 2] > 0:
            next_state = list(state)
            next_state[first] -= 1
            next_state[first + 2] -= 1
            best = min(best, _standard_shanten_search(tuple(next_state), mentsu, taatsu + 1, pair))

    return best


def _estimate_standard_shanten_with_open_melds(initial_counts: tuple[int, ...], open_melds: int) -> int:
    open_melds = max(0, min(4, int(open_melds or 0)))
    return _standard_shanten_search(initial_counts, open_melds, 0, 0)


def _estimate_chiitoi_shanten(counts: list[int]) -> int:
    pairs = sum(1 for count in counts if count >= 2)
    unique = sum(1 for count in counts if count > 0)
    return 6 - pairs + max(0, 7 - unique)


def _estimate_kokushi_shanten(counts: list[int]) -> int:
    terminal_honor_indexes = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    unique = sum(1 for index in terminal_honor_indexes if counts[index] > 0)
    has_pair = any(counts[index] >= 2 for index in terminal_honor_indexes)
    return 13 - unique - (1 if has_pair else 0)


def _counter_to_tile_counts(counts: Counter[str]) -> list[int]:
    tile_counts = [0] * 34
    for tile, count in counts.items():
        index = _tile_index(tile)
        if index is not None:
            tile_counts[index] += int(count)
    return tile_counts


def _first_non_empty_index(counts: tuple[int, ...]) -> int | None:
    for index, count in enumerate(counts):
        if count > 0:
            return index
    return None


def _is_suited_index(index: int) -> bool:
    return 0 <= index < 27


def _estimate_ukeire(candidate_discards: list[dict[str, Any]]) -> int | None:
    if not candidate_discards:
        return None
    top = candidate_discards[0].get("ukeire_estimate")
    value = coerce_int(top)
    if value is not None:
        return value
    return max(4, len(candidate_discards) * 4)


def _calculate_discard_ukeire(
    hand_tiles: list[str],
    discard_tile: str,
    *,
    visible_tiles: list[str] | None = None,
    open_melds: int = 0,
) -> int | None:
    counts = Counter(_normalize_tile(tile) for tile in hand_tiles if _normalize_tile(tile))
    normalized_discard = _normalize_tile(discard_tile)
    if not counts or not normalized_discard:
        return None
    remaining_counts = _remaining_tile_counts(counts, visible_tiles or [])
    return _calculate_discard_ukeire_from_counts(counts, normalized_discard, remaining_counts, open_melds)


def _calculate_discard_ukeire_from_counts(
    hand_counts: Counter[str],
    discard_tile: str,
    remaining_counts: list[int],
    open_melds: int = 0,
) -> int | None:
    normalized_discard = _normalize_tile(discard_tile)
    if not normalized_discard or hand_counts.get(normalized_discard, 0) <= 0:
        return None

    after_discard = _counts_after_discard(hand_counts, normalized_discard)
    if after_discard is None:
        return None

    baseline_shanten = _estimate_unclamped_shanten_from_counts(after_discard, open_melds)
    if baseline_shanten is None:
        return None
    ukeire = 0
    for index, remaining in enumerate(remaining_counts):
        if remaining <= 0:
            continue
        draw_tile = _tile_from_index(index)
        if draw_tile is None:
            continue
        drawn_counts = Counter(after_discard)
        drawn_counts[draw_tile] += 1
        shanten_after_draw = _estimate_unclamped_shanten_from_counts(drawn_counts, open_melds)
        if shanten_after_draw is None:
            continue
        if shanten_after_draw < baseline_shanten:
            ukeire += remaining
    return ukeire


def _estimate_post_discard_shanten_from_counts(
    hand_counts: Counter[str],
    discard_tile: str,
    open_melds: int = 0,
) -> int | None:
    after_discard = _counts_after_discard(hand_counts, discard_tile)
    if after_discard is None:
        return None
    return _estimate_shanten_from_counts(after_discard, open_melds)


def _counts_after_discard(hand_counts: Counter[str], discard_tile: str) -> Counter[str] | None:
    normalized_discard = _normalize_tile(discard_tile)
    if not normalized_discard or hand_counts.get(normalized_discard, 0) <= 0:
        return None
    after_discard = Counter(hand_counts)
    after_discard[normalized_discard] -= 1
    if after_discard[normalized_discard] <= 0:
        del after_discard[normalized_discard]
    return after_discard


def _rank_discard_candidates(candidates: list[dict[str, Any]]) -> None:
    candidates.sort(key=_discard_candidate_sort_key)


def _discard_candidate_sort_key(item: dict[str, Any]) -> tuple[int, float, float, float, float]:
    post_discard_shanten = coerce_int(item.get("post_discard_shanten"))
    shanten_rank = post_discard_shanten if post_discard_shanten is not None else _UNKNOWN_SHANTEN_SORT_VALUE
    ukeire_rank = coerce_int(item.get("ukeire_estimate")) or 0
    raw_score = _coerce_float(item.get("raw_score")) or 0.0
    safety = _SAFETY_SORT_VALUE.get(str(item.get("safety_hint", "unknown")).strip(), 0)
    strategy_score = _candidate_strategy_sort_score(item, raw_score=raw_score, ukeire=ukeire_rank, safety=safety)
    strategy_mode = str(item.get("strategy_mode", "")).strip()
    if strategy_mode in {"defense", "guarded_push"}:
        return (shanten_rank, -strategy_score, -float(ukeire_rank), -raw_score, -safety)
    return (shanten_rank, -float(ukeire_rank), -strategy_score, -raw_score, -safety)


def _candidate_strategy_sort_score(
    item: dict[str, Any],
    *,
    raw_score: float,
    ukeire: int,
    safety: float,
) -> float:
    strategy_score = _coerce_float(item.get("strategy_score"))
    if strategy_score is not None:
        return strategy_score
    hinted_score = _coerce_float(item.get("score"))
    if hinted_score is not None:
        return hinted_score
    return raw_score + ukeire * 0.25 + safety * 0.35


def _candidate_output_score(item: dict[str, Any]) -> float:
    strategy_score = _coerce_float(item.get("strategy_score"))
    if strategy_score is not None:
        return strategy_score
    return _coerce_float(item.get("raw_score")) or 0.0


def _candidate_recommendation_score(item: dict[str, Any]) -> float:
    post_discard_shanten = coerce_int(item.get("post_discard_shanten"))
    shanten_rank = post_discard_shanten if post_discard_shanten is not None else _UNKNOWN_SHANTEN_SORT_VALUE
    return _candidate_output_score(item) - shanten_rank * 100.0


def _discard_strategy_mode(
    *,
    has_riichi_pressure: bool,
    shanten_estimate: int | None,
    hints: dict[str, Any],
) -> str:
    hinted = str(hints.get("strategy_mode") or hints.get("risk_policy") or "").strip()
    if hinted in {"defense", "fold"}:
        return "defense"
    if hinted in {"guarded_push", "push", "balanced"}:
        return hinted

    bias = str(hints.get("attack_defense_bias", "")).strip()
    if bias == "defensive":
        return "defense"
    if bias == "slightly_defensive":
        return "guarded_push" if shanten_estimate is not None and shanten_estimate <= 1 else "defense"
    if bias in {"attack", "slightly_attack"}:
        return "push"

    if has_riichi_pressure:
        return "guarded_push" if shanten_estimate is not None and shanten_estimate <= 1 else "defense"
    if shanten_estimate is not None and shanten_estimate <= 1:
        return "push"
    return "balanced"


def _discard_strategy_score(
    *,
    strategy_mode: str,
    ukeire: int | None,
    raw_score: float,
    safety_hint: str,
    dora_value: float,
    wait_quality_bonus: int,
) -> float:
    ukeire_value = float(max(0, ukeire or 0))
    safety_value = float(_SAFETY_SORT_VALUE.get(str(safety_hint).strip(), 0))
    attack_score = ukeire_value * 0.62 + raw_score * 1.15 + wait_quality_bonus * 0.45 - dora_value * 3.45
    defense_score = safety_value * 7.5 + raw_score * 0.25 + ukeire_value * 0.08 - dora_value * 0.35

    if strategy_mode == "defense":
        return defense_score
    if strategy_mode == "guarded_push":
        return attack_score * 0.58 + defense_score * 0.42
    return attack_score + safety_value * 0.18


def _discard_defense_score(
    *,
    safety_hint: str,
    raw_score: float,
    dora_value: float,
) -> float:
    safety_value = float(_SAFETY_SORT_VALUE.get(str(safety_hint).strip(), 0))
    return safety_value * 7.5 + raw_score * 0.25 - dora_value * 0.35


def _dora_value(tile: str, dora_tiles: set[str]) -> float:
    return 1.0 if _normalize_tile(tile) in dora_tiles else 0.0


def _remaining_tile_counts(hand_counts: Counter[str], visible_tiles: list[str]) -> list[int]:
    remaining = [4] * 34
    for tile, count in hand_counts.items():
        index = _tile_index(tile)
        if index is not None:
            remaining[index] -= int(count)
    for tile in visible_tiles:
        index = _tile_index(tile)
        if index is not None:
            remaining[index] -= 1
    return [max(0, count) for count in remaining]


def _deck_state_trusted(hints: dict[str, Any], visible_tiles: list[str]) -> bool:
    if not visible_tiles:
        return False
    if bool(hints.get("deck_state_complete")):
        return True
    try:
        recognized_count = int(hints.get("recognized_discard_tile_count", 0) or 0)
    except (TypeError, ValueError):
        recognized_count = 0
    try:
        confidence = float(hints.get("discard_analysis_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    source = str(hints.get("discard_parser_source", "")).strip()
    if source == "external_discard_recognizer":
        return recognized_count >= 1
    return recognized_count >= 4 and confidence >= 0.62


def _derive_attack_defense_bias(
    *,
    state: PerceivedGameState,
    shanten_estimate: int | None,
    candidate_discards: list[dict[str, Any]],
) -> str:
    if _opponent_riichi_players(state):
        return "slightly_defensive"
    if shanten_estimate is not None and shanten_estimate <= 1:
        return "slightly_attack"
    return "neutral"


def _build_teaching_points(
    *,
    state: PerceivedGameState,
    recommended_focus: str,
    review_tags: list[str],
    bias: str,
    candidate_discards: list[dict[str, Any]],
    shanten_estimate: int | None,
    hints: dict[str, Any],
) -> list[str]:
    teaching_points: list[str] = []
    top = candidate_discards[0] if candidate_discards else None

    if top is not None:
        teaching_points.append("这巡已经拿到结构化手牌信息，可以开始给出轻量牌效率建议。")
        tile = str(top.get("tile", "")).strip()
        tile_label = format_tile_label(tile)
        reason = replace_tile_codes_in_text(top.get("reason", ""))
        if tile_label and reason:
            teaching_points.append(f"当前更自然的处理方向是先考虑 {tile_label}，因为{reason}。")
    elif recommended_focus == "turn_observe":
        teaching_points.append("虽然轮到你关注了，但当前牌理输入还不够完整，先别把轻量建议当成精算答案。")

    if shanten_estimate is not None:
        if shanten_estimate <= 1:
            teaching_points.append("当前已经接近成型，优先别打散已经连起来的块。")
        elif shanten_estimate >= 3:
            teaching_points.append("当前离成型还比较远，更适合先处理改善较弱的孤张或边张。")

    if bias == "slightly_defensive":
        teaching_points.append("这手更适合稍微稳一点，先保留更容易回旋的形。")
    elif bias == "slightly_attack":
        teaching_points.append("这手已经有继续推进的空间，可以优先保留更能改善进张的部分。")

    if _opponent_riichi_players(state):
        teaching_points.append("场上已经有人立直，轻量牌理建议也要带一点防守意识。")

    visible_tiles = _normalize_tile_list(
        hints.get("visible_tiles")
        or hints.get("dead_tiles")
        or hints.get("known_visible_tiles"),
    )
    if _deck_state_trusted(hints, visible_tiles):
        teaching_points.append(f"这次已参考牌河里 {len(visible_tiles)} 张可见牌，进张估计会按剩余牌修正。")

    extra_points = hints.get("teaching_points")
    if isinstance(extra_points, list):
        for item in extra_points:
            text = str(item).strip()
            if text:
                teaching_points.append(replace_tile_codes_in_text(text))

    if "low_confidence" in set(review_tags) or state.confidence < 0.45:
        teaching_points.append("当前识别置信度仍偏低，最好把这类建议和截图一起看。")

    return _dedupe(teaching_points)[:4]


_SHAPE_PENALTY = {
    "complete": -2.2,
    "ryanmen": -1.8,
    "kanchan": -0.8,
    "penchan": -0.7,
}


def _raw_discard_score(tile: str, counts: Counter[str], dora_tiles: set[str]) -> float:
    normalized = _normalize_tile(tile)
    if not normalized:
        return 0.0
    count = counts[normalized]
    base = 0.0
    if _is_honor(normalized):
        if count == 1:
            base += 2.2
        elif count >= 2:
            base -= 1.2 * min(count, 3)
    else:
        number, suit = _parse_suited_tile(normalized) or (0, "")
        base += 1.3 if number in {1, 9} else 0.2
        base += _best_shape_penalty(normalized, number, suit, counts)
        if count >= 2:
            base -= 1.4 * min(count, 3)
    if normalized in dora_tiles:
        base -= 1.6
    return base


def _best_shape_penalty(tile: str, number: int, suit: str, counts: Counter[str]) -> float:
    left1 = counts.get(f"{number - 1}{suit}", 0)
    left2 = counts.get(f"{number - 2}{suit}", 0)
    right1 = counts.get(f"{number + 1}{suit}", 0)
    right2 = counts.get(f"{number + 2}{suit}", 0)

    if left1 > 0 and right1 > 0:
        return _SHAPE_PENALTY["complete"]

    best: float = 0.0

    if left1 > 0:
        if number - 1 >= 2 and number <= 8:
            best = min(best, _SHAPE_PENALTY["ryanmen"])
        else:
            best = min(best, _SHAPE_PENALTY["penchan"])

    if right1 > 0:
        if number >= 2 and number + 1 <= 8:
            best = min(best, _SHAPE_PENALTY["ryanmen"])
        else:
            best = min(best, _SHAPE_PENALTY["penchan"])

    if left1 == 0 and left2 > 0:
        best = min(best, _SHAPE_PENALTY["kanchan"])
    if right1 == 0 and right2 > 0:
        best = min(best, _SHAPE_PENALTY["kanchan"])

    return best


def _tenpai_wait_quality_bonus(counts: Counter[str], discard_tile: str) -> int:
    """Estimate the quality of the expected wait after reaching tenpai.

    At 1-shanten, prefers routes that preserve ryanmen shapes so the final
    wait is more likely to be a two-sided (8-tile) wait rather than a
    kanchan/penchan/tanki wait.  Returns a ukeire bonus (positive = better).
    """
    normalized_discard = _normalize_tile(discard_tile)
    after = Counter(counts)
    after[normalized_discard] -= 1
    if after[normalized_discard] <= 0:
        del after[normalized_discard]

    ryanmen = 0
    for tile, count in after.items():
        n, s = _parse_suited_tile(tile) or (None, "")
        if n is None or n < 2 or n >= 8:
            continue
        right = f"{n + 1}{s}"
        if after.get(right, 0) > 0:
            ryanmen += 1

    if ryanmen >= 2:
        return 3
    if ryanmen == 1:
        return 2
    return -1


def _raw_ukeire_score(tile: str, counts: Counter[str]) -> int:
    normalized = _normalize_tile(tile)
    if not normalized:
        return 0
    if _is_honor(normalized):
        return 8 if counts[normalized] == 1 else 4
    parsed = _parse_suited_tile(normalized)
    if parsed is None:
        return 2
    number, suit = parsed
    shape = min(0.0, _best_shape_penalty(normalized, number, suit, counts))
    if shape <= _SHAPE_PENALTY["complete"]:
        return 2
    if shape <= _SHAPE_PENALTY["ryanmen"]:
        return 4
    if shape <= _SHAPE_PENALTY["kanchan"]:
        return 5
    if shape <= _SHAPE_PENALTY["penchan"]:
        return 6
    return 8


def _reason_for_tile(
    tile: str,
    counts: Counter[str],
    dora_tiles: set[str],
    *,
    uses_visible_tiles: bool = False,
    visible_tile_count: int = 0,
) -> str:
    normalized = _normalize_tile(tile)
    if not normalized:
        return "当前结构信息不足"
    suffix = (
        f"，并已参考牌河里 {visible_tile_count} 张可见牌修正剩余进张"
        if uses_visible_tiles and visible_tile_count > 0
        else ""
    )
    if normalized in dora_tiles:
        return "它本身接近宝牌价值，轻量建议并不倾向优先打掉"
    count = counts[normalized]
    if _is_honor(normalized):
        if count >= 3:
            return "它已经接近刻子价值，轻量建议通常不倾向拆掉"
        if count >= 2:
            return "它已经形成对子，轻量建议通常更想先保留一下"
        return f"字牌在当前样本里更像单张孤张，改善空间通常偏窄{suffix}"
    number, suit = _parse_suited_tile(normalized) or (0, "")
    shape = min(0.0, _best_shape_penalty(normalized, number, suit, counts))
    if shape <= _SHAPE_PENALTY["complete"]:
        return "它已经嵌在完整顺子当中，拆掉会直接破坏成型结构"
    if count >= 3:
        return "它已经形成刻子，轻量建议通常不倾向拆掉"
    if count >= 2:
        if shape <= _SHAPE_PENALTY["ryanmen"]:
            return "它既是对子又搭着两面搭子，保留下来的改善空间偏大"
        return "它已经形成对子，轻量建议通常更想先保留一下"
    if shape <= _SHAPE_PENALTY["ryanmen"]:
        return "它参与了两面搭子，进张范围更宽，不建议优先拆"
    if shape <= _SHAPE_PENALTY["kanchan"]:
        return "它参与了一组嵌张搭子，改善空间比孤张更强但不如两面"
    if shape <= _SHAPE_PENALTY["penchan"]:
        return f"它参与了一组边张搭子，只能单边等进张，改善优先级偏后{suffix}"
    if number in {1, 9}:
        return f"它更像孤张幺九，和当前主线的连接感偏弱{suffix}"
    return f"它暂时比较孤立，改善路线没有那么多{suffix}"


def _visible_tile_candidate_reason(item: dict[str, Any], *, visible_tile_count: int) -> str:
    tile = _normalize_tile(item.get("tile"))
    ukeire = coerce_int(item.get("ukeire_estimate"))
    post_discard_shanten = coerce_int(item.get("post_discard_shanten"))
    safety_hint = str(item.get("safety_hint", "")).strip()
    parts: list[str] = []
    if post_discard_shanten is not None:
        parts.append(f"打出后约 {post_discard_shanten} 向听")
    if ukeire is not None:
        parts.append(f"参考牌河里 {visible_tile_count} 张可见牌后，这张打出后的剩余有效牌约 {ukeire} 张")
    else:
        parts.append(f"已参考牌河里 {visible_tile_count} 张可见牌修正剩余牌判断")
    if safety_hint == "genbutsu":
        parts.append("它同时是现物，防守压力下更稳")
    elif safety_hint == "dead":
        parts.append("这张已经接近壁牌/死牌，放铳风险更低")
    elif safety_hint == "suji":
        parts.append("它符合筋的防守线索，立直压力下比无筋更稳")
    elif safety_hint == "high":
        parts.append("它的安全度相对更高")
    elif tile and _is_honor(tile):
        parts.append("它是字牌，通常比中张更容易作为早处理对象")
    return "，".join(parts)


def _candidate_metrics_reason(item: dict[str, Any], reason: str) -> str:
    post_discard_shanten = coerce_int(item.get("post_discard_shanten"))
    ukeire = coerce_int(item.get("ukeire_estimate"))
    parts: list[str] = []
    if post_discard_shanten is not None:
        parts.append(f"打出后约 {post_discard_shanten} 向听")
    if ukeire is not None:
        parts.append(f"有效牌估计约 {ukeire} 张")
    if not parts:
        return reason
    metrics = "，".join(parts)
    return f"{metrics}；{reason}" if reason else metrics


def _defensive_safety_hint(
    tile: str,
    *,
    counts: Counter[str],
    visible_tiles: list[str],
    genbutsu_tiles: set[str],
    has_riichi_pressure: bool,
) -> str:
    normalized = _normalize_tile(tile)
    if not normalized:
        return "unknown"
    if normalized in genbutsu_tiles:
        return "genbutsu"
    if not has_riichi_pressure:
        return _safety_hint(normalized, counts)

    if counts.get(normalized, 0) + Counter(visible_tiles).get(normalized, 0) >= 4:
        return "dead"
    if _is_suji_safe(normalized, genbutsu_tiles):
        return "suji"
    return _safety_hint(normalized, counts)


def _is_suji_safe(tile: str, genbutsu_tiles: set[str]) -> bool:
    parsed = _parse_suited_tile(tile)
    if parsed is None:
        return False
    number, suit = parsed
    if number in {1, 7}:
        return f"4{suit}" in genbutsu_tiles
    if number in {2, 8}:
        return f"5{suit}" in genbutsu_tiles
    if number in {3, 9}:
        return f"6{suit}" in genbutsu_tiles
    if number == 4:
        return f"1{suit}" in genbutsu_tiles and f"7{suit}" in genbutsu_tiles
    if number == 5:
        return f"2{suit}" in genbutsu_tiles and f"8{suit}" in genbutsu_tiles
    if number == 6:
        return f"3{suit}" in genbutsu_tiles and f"9{suit}" in genbutsu_tiles
    return False


def _safety_hint(tile: str, counts: Counter[str]) -> str:
    normalized = _normalize_tile(tile)
    if not normalized:
        return "unknown"
    if _is_honor(normalized):
        return "high" if counts[normalized] == 1 else "medium"
    number, suit = _parse_suited_tile(normalized) or (0, "")
    support = counts.get(f"{number - 1}{suit}", 0) + counts.get(f"{number + 1}{suit}", 0)
    if number in {1, 9} and support == 0:
        return "high"
    if support <= 1:
        return "medium"
    return "low"


def _derive_dora_tiles(indicators: list[str]) -> set[str]:
    derived: set[str] = set()
    for indicator in indicators:
        normalized = _normalize_tile(indicator)
        if not normalized:
            continue
        if _is_honor(normalized):
            if normalized in {"1z", "2z", "3z", "4z"}:
                winds = ["1z", "2z", "3z", "4z"]
                derived.add(winds[(winds.index(normalized) + 1) % 4])
            elif normalized in {"5z", "6z", "7z"}:
                dragons = ["5z", "6z", "7z"]
                derived.add(dragons[(dragons.index(normalized) + 1) % 3])
            continue
        parsed = _parse_suited_tile(normalized)
        if parsed is None:
            continue
        number, suit = parsed
        derived.add(f"{1 if number == 9 else number + 1}{suit}")
    return derived




def _parse_suited_tile(tile: str) -> tuple[int, str] | None:
    normalized = _normalize_tile(tile)
    if len(normalized) != 2 or normalized[1] not in {"m", "p", "s"}:
        return None
    return int(normalized[0]), normalized[1]


def _tile_index(tile: str) -> int | None:
    normalized = _normalize_tile(tile)
    if not normalized:
        return None
    number = int(normalized[0])
    suit = normalized[1]
    if suit == "m":
        return number - 1
    if suit == "p":
        return 9 + number - 1
    if suit == "s":
        return 18 + number - 1
    if suit == "z" and 1 <= number <= 7:
        return 27 + number - 1
    return None


def _tile_from_index(index: int) -> str | None:
    if 0 <= index < 9:
        return f"{index + 1}m"
    if 9 <= index < 18:
        return f"{index - 8}p"
    if 18 <= index < 27:
        return f"{index - 17}s"
    if 27 <= index < 34:
        return f"{index - 26}z"
    return None


def _is_honor(tile: str) -> bool:
    return _normalize_tile(tile).endswith("z")


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _opponent_riichi_players(state: PerceivedGameState) -> list[str]:
    return [
        player
        for player in (normalize_riichi_player(item) for item in state.riichi_players)
        if player in OPPONENT_PLAYERS
    ]


def _has_opponent_riichi(
    *,
    state: PerceivedGameState | None = None,
    riichi_players: list[str] | None = None,
    hints: dict[str, Any] | None = None,
) -> bool:
    players: list[Any] = []
    if state is not None:
        players.extend(state.riichi_players)
    if isinstance(riichi_players, list):
        players.extend(riichi_players)
    if isinstance(hints, dict):
        hinted_players = hints.get("riichi_players") or hints.get("opponent_riichi_players")
        if isinstance(hinted_players, list):
            players.extend(hinted_players)
        elif hinted_players:
            players.append(hinted_players)
        if bool(hints.get("opponent_riichi") or hints.get("riichi_pressure")):
            return True
    return any(normalize_riichi_player(item) in OPPONENT_PLAYERS for item in players)
