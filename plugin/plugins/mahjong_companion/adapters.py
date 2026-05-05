from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from .contracts import DecisionResult, PerceivedGameState
from .narration.events import NarrationEvent
from .narration.generator import generate_narration
from .narration.view_model import CompanionViewModel
from .perception.pipeline import analyze_image_path


class PerceptionAdapter(Protocol):
    def analyze(self, image_path: Path, *, live: bool = False) -> tuple[PerceivedGameState, dict[str, Any]]:
        ...


class NarrationAdapter(Protocol):
    def generate(self, decision: DecisionResult) -> tuple[NarrationEvent, CompanionViewModel, dict[str, Any]]:
        ...


class DefaultPerceptionAdapter:
    def __init__(
        self,
        *,
        calibration_dir: Path | None = None,
        template_dir: Path | None = None,
        fixture_mode: str = "auto",
        tile_classifier_config: dict[str, Any] | None = None,
    ) -> None:
        self._calibration_dir = calibration_dir
        self._template_dir = template_dir
        self._fixture_mode = fixture_mode
        self._tile_classifier_config = dict(tile_classifier_config or {})

    def apply_perception_config(self, perception_cfg: dict[str, Any]) -> None:
        classifier_cfg = perception_cfg.get("tile_classifier") if isinstance(perception_cfg, dict) else {}
        self._tile_classifier_config = dict(classifier_cfg) if isinstance(classifier_cfg, dict) else {}

    def analyze(self, image_path: Path, *, live: bool = False) -> tuple[PerceivedGameState, dict[str, Any]]:
        return analyze_image_path(
            image_path,
            calibration_dir=self._calibration_dir,
            template_dir=self._template_dir,
            fixture_mode=self._fixture_mode,
            tile_classifier_config=self._tile_classifier_config_for_call(live=live),
        )

    def _tile_classifier_config_for_call(self, *, live: bool) -> dict[str, Any]:
        config = dict(self._tile_classifier_config)
        if live and not _live_tile_classifier_enabled(config):
            config["enabled"] = False
            config["force_disabled"] = True
            config["disabled_reason"] = "live_tile_classifier_disabled"
        return config


def _live_tile_classifier_enabled(config: dict[str, Any]) -> bool:
    env_value = os.environ.get("MAHJONG_COMPANION_VIT_LIVE_ENABLED")
    if env_value is not None:
        return _truthy(env_value)
    return _truthy(config.get("live_enabled", False))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "disabled"}


class DefaultNarrationAdapter:
    def generate(self, decision: DecisionResult) -> tuple[NarrationEvent, CompanionViewModel, dict[str, Any]]:
        return generate_narration(decision)
