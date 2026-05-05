from __future__ import annotations

from typing import Any

from ..contracts import PerceivedGameState
from ..tile_labels import (
    dedupe as _dedupe,
    format_tile_label,
    normalize_tile as _normalize_tile,
    normalize_tile_set as _normalize_tile_set,
    replace_tile_codes_in_text,
)

OPPONENT_PLAYERS = {"left_opponent", "top_opponent", "right_opponent"}
PLAYER_ALIASES = {
    "left": "left_opponent",
    "kamicha": "left_opponent",
    "top": "top_opponent",
    "opposite": "top_opponent",
    "toimen": "top_opponent",
    "right": "right_opponent",
    "shimocha": "right_opponent",
}


def normalize_riichi_player(player: Any) -> str:
    value = str(player).strip()
    return PLAYER_ALIASES.get(value, value)


def estimate_defense_alerts(
    state: PerceivedGameState,
    *,
    candidate_discards: list[dict[str, Any]] | None = None,
    shanten_estimate: int | None = None,
    attack_defense_bias: str = "neutral",
    hints: dict[str, Any] | None = None,
) -> list[str]:
    hints = hints if isinstance(hints, dict) else {}
    hinted = hints.get("defense_alerts")
    if isinstance(hinted, list) and hinted:
        normalized = [replace_tile_codes_in_text(item) for item in hinted if str(item).strip()]
        if normalized:
            return _dedupe(normalized)[:3]

    candidate_discards = candidate_discards or []
    alerts: list[str] = []
    genbutsu_tiles = _normalize_tile_set(
        hints.get("genbutsu_tiles")
        or hints.get("known_genbutsu_tiles")
        or hints.get("confirmed_safe_tiles"),
    )
    opponent_riichi_players = [
        player
        for player in (normalize_riichi_player(item) for item in state.riichi_players)
        if player in OPPONENT_PLAYERS
    ]

    if opponent_riichi_players:
        alerts.append("场上已经有立直压力，这巡先确认现物与安牌会更稳。")

    candidate_genbutsu = [
        str(item.get("tile", "")).strip()
        for item in candidate_discards
        if _normalize_tile(item.get("tile")) in genbutsu_tiles and str(item.get("tile", "")).strip()
    ]
    if candidate_genbutsu and opponent_riichi_players:
        alerts.append(f"已确认现物：{format_tile_label(candidate_genbutsu[0])}，需要防守时优先看它。")
    elif genbutsu_tiles and opponent_riichi_players:
        alerts.append(f"已确认现物：{format_tile_label(sorted(genbutsu_tiles)[0])}，候选弃牌需要优先对照这批安全信息。")
    elif opponent_riichi_players:
        alerts.append("目前还没有确认现物保护，防守判断先保持保守。")

    high_safety_tiles = [
        str(item.get("tile", "")).strip()
        for item in candidate_discards
        if str(item.get("safety_hint", "")).strip() in {"high", "genbutsu"}
        and str(item.get("tile", "")).strip()
    ]
    if high_safety_tiles and opponent_riichi_players:
        alerts.append(f"候选里已有相对安全的牌，例如 {format_tile_label(high_safety_tiles[0])}，需要时可以先用它过渡。")

    if shanten_estimate is not None and shanten_estimate >= 2 and attack_defense_bias in {"slightly_defensive", "defensive"}:
        alerts.append("当前离成型还不算近，面对场压时先别急着强行推进。")

    if "kan" in state.buttons and opponent_riichi_players:
        alerts.append("场上有立直时再开杠风险会更高，先确认这手值不值得冒险。")

    if attack_defense_bias == "defensive" and not alerts:
        alerts.append("当前分析更偏防守，优先保留退路会更自然。")

    return _dedupe(alerts)[:3]
