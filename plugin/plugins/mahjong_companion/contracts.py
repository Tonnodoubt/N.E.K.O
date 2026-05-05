from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FramePacket:
    timestamp_ms: int
    image_path: str = ""
    window_title: str = ""
    width: int = 0
    height: int = 0
    source: str = "pyautogui"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerceivedGameState:
    scene: str = "unknown"
    confidence: float = 0.0
    is_user_turn: bool = False
    buttons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    roi_hits: dict[str, bool] = field(default_factory=dict)
    hand_tiles: list[str] = field(default_factory=list)
    melds: list[list[str]] = field(default_factory=list)
    dora_indicators: list[str] = field(default_factory=list)
    riichi_players: list[str] = field(default_factory=list)
    discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visible_tiles: list[str] = field(default_factory=list)
    known_genbutsu_tiles: list[str] = field(default_factory=list)
    button_regions: list[dict[str, Any]] = field(default_factory=list)
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    analysis_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionResult:
    decision_type: str = "uncertain_state"
    priority: int = 0
    risk_level: str = "unknown"
    action_required: bool = False
    speakable: bool = False
    summary: str = ""
    detail: str = ""
    suggestion: str = ""
    recommended_focus: str = ""
    scene: str = "unknown"
    buttons: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    review_tags: list[str] = field(default_factory=list)
    review_summary_snippet: str = ""
    mahjong_analysis: dict[str, Any] = field(default_factory=dict)
    engine_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MahjongAnalysis:
    analysis_version: str = "mahjong-lite-v1"
    tile_level_available: bool = False
    tile_level_state: str = "tile_level_unavailable"
    analysis_confidence: float = 0.0
    hand_shape_confidence: float = 0.0
    shanten_estimate: int | None = None
    ukeire_estimate: int | None = None
    candidate_discards: list[dict[str, Any]] = field(default_factory=list)
    attack_defense_bias: str = "neutral"
    defense_alerts: list[str] = field(default_factory=list)
    teaching_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
