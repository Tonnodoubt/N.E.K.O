from __future__ import annotations

from typing import Any

from .mahjong_analysis import PerceivedGameState


def coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def meld_group_count(state: PerceivedGameState) -> int:
    if isinstance(state.melds, list) and state.melds:
        return len([group for group in state.melds if isinstance(group, list) and group])
    hints = state.analysis_hints if isinstance(state.analysis_hints, dict) else {}
    return max(0, coerce_int(hints.get("recognized_meld_group_count")) or 0)


def recognized_hand_tile_count(state: PerceivedGameState) -> int:
    hints = state.analysis_hints if isinstance(state.analysis_hints, dict) else {}
    hinted = coerce_int(hints.get("recognized_hand_tile_count")) or 0
    return max(hinted, len(state.hand_tiles))
