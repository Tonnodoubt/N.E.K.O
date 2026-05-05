from __future__ import annotations

from typing import Any

from ..contracts import MahjongAnalysis, PerceivedGameState


def derive_tile_level_state(state: PerceivedGameState, hints: dict[str, Any]) -> str:
    hinted = str(hints.get("tile_level_state", "")).strip()
    if hinted:
        return hinted
    if state.hand_tiles:
        return "tile_level_reliable"
    if hints.get("tile_level_available"):
        return "tile_level_partial"
    return "tile_level_unavailable"


def derive_analysis_confidence(state: PerceivedGameState, hints: dict[str, Any]) -> float:
    raw = hints.get("analysis_confidence")
    try:
        value = float(raw)
    except Exception:
        value = 0.0
    if value > 0.0:
        return max(0.0, min(1.0, value))
    if state.hand_tiles:
        return 0.78
    if hints.get("tile_level_available"):
        return 0.46
    return 0.0


def attach_confidence_metadata(
    analysis: MahjongAnalysis,
    *,
    state: PerceivedGameState,
    hints: dict[str, Any],
) -> MahjongAnalysis:
    analysis.tile_level_state = derive_tile_level_state(state, hints)
    analysis.analysis_confidence = derive_analysis_confidence(state, hints)
    if analysis.tile_level_state == "tile_level_reliable" and not analysis.tile_level_available:
        analysis.tile_level_available = True
    return analysis
