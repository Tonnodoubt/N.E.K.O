from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MahjongCoachConfig:
    live_advice_mode: str = "coach"
    coach_checkpoint_self_turns: int = 3
    critical_action_interrupts: bool = True
    per_turn_discard_prompt: bool = False
    hand_recognition_backend: str = "legacy_templates"
    onnx_hand_enabled: bool = False
    river_recognition_enabled: bool = True
    river_min_confidence: float = 0.90
    live_window_keywords: list[str] = field(default_factory=lambda: ["雀魂", "Mahjong Soul"])
    live_interval_ms: int = 1200
    live_fast_interval_ms: int = 300
    live_keep_frames: int = 30
    live_checkpoint_interval_seconds: int = 20
    live_overlay_enabled: bool = True
    live_save_format: str = "jpg"
    llm_enabled: bool = True
    llm_timeout: float = 8.0
    llm_opening_enabled: bool = True
    llm_checkpoint_enabled: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "MahjongCoachConfig":
        payload = payload or {}
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        perception = payload.get("perception") if isinstance(payload.get("perception"), dict) else {}
        live = payload.get("live") if isinstance(payload.get("live"), dict) else {}
        llm = payload.get("llm") if isinstance(payload.get("llm"), dict) else {}
        return cls(
            live_advice_mode=str(decision.get("live_advice_mode") or "coach"),
            coach_checkpoint_self_turns=max(1, int(decision.get("coach_checkpoint_self_turns") or 3)),
            critical_action_interrupts=bool(decision.get("critical_action_interrupts", True)),
            per_turn_discard_prompt=bool(decision.get("per_turn_discard_prompt", False)),
            hand_recognition_backend=str(perception.get("hand_recognition_backend") or "legacy_templates"),
            onnx_hand_enabled=bool(perception.get("onnx_hand_enabled", False)),
            river_recognition_enabled=bool(perception.get("river_recognition_enabled", True)),
            river_min_confidence=max(0.0, min(1.0, float(perception.get("river_min_confidence") or 0.90))),
            live_window_keywords=_string_list(live.get("window_keywords"), ["雀魂", "Mahjong Soul"]),
            live_interval_ms=max(200, int(live.get("interval_ms") or 1200)),
            live_fast_interval_ms=max(100, int(live.get("fast_interval_ms") or 300)),
            live_keep_frames=max(5, int(live.get("keep_frames") or 30)),
            live_checkpoint_interval_seconds=max(5, int(live.get("checkpoint_interval_seconds") or 20)),
            live_overlay_enabled=bool(live.get("overlay_enabled", True)),
            live_save_format=str(live.get("save_format") or "jpg"),
            llm_enabled=bool(llm.get("enabled", True)),
            llm_timeout=max(1.0, float(llm.get("timeout") or 8.0)),
            llm_opening_enabled=bool(llm.get("opening_enabled", True)),
            llm_checkpoint_enabled=bool(llm.get("checkpoint_enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoundCoachState:
    round_id: str = "default"
    round_phase: str = "round_idle"
    opening_emitted: bool = False
    opening_plan: str = ""
    current_plan: str = ""
    attack_defense_bias: str = "neutral"
    target_shapes: list[str] = field(default_factory=list)
    caution_points: list[str] = field(default_factory=list)
    last_hand_signature: str = ""
    last_hand_tiles: list[str] = field(default_factory=list)
    last_hand_confidence: float = 0.0
    last_discard_piles: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    last_visible_discards: list[str] = field(default_factory=list)
    last_river_confidence: float = 0.0
    last_checkpoint_self_turn: int = 0
    last_update_reason: str = ""
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoachDecision:
    decision_type: str = "observe"
    priority: int = 0
    action_required: bool = False
    summary: str = ""
    detail: str = ""
    suggestion: str = ""
    buttons: list[str] = field(default_factory=list)
    hand_tiles: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    coach_state: dict[str, Any] = field(default_factory=dict)
    perception: dict[str, Any] = field(default_factory=dict)
    engine_meta: dict[str, Any] = field(default_factory=dict)
    llm_enhanced: bool = False
    analysis_source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FramePacket:
    timestamp_ms: int
    image_path: str = ""
    window_title: str = ""
    width: int = 0
    height: int = 0
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveSessionState:
    running: bool = False
    status: str = "stopped"
    frame_index: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    last_error: str = ""
    last_frame_path: str = ""
    last_capture_source: str = ""
    last_window_title: str = ""
    last_binding: dict[str, Any] = field(default_factory=dict)
    observed_hand_changes: int = 0
    missing_hand_frames: int = 0
    overlay_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _string_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
        return result or list(fallback)
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
        return result or list(fallback)
    return list(fallback)
