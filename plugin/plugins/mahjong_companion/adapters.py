from __future__ import annotations

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
    ) -> None:
        self._calibration_dir = calibration_dir
        self._template_dir = template_dir
        self._fixture_mode = fixture_mode

    def apply_perception_config(self, perception_cfg: dict[str, Any]) -> None:
        pass

    def analyze(self, image_path: Path, *, live: bool = False) -> tuple[PerceivedGameState, dict[str, Any]]:
        return analyze_image_path(
            image_path,
            calibration_dir=self._calibration_dir,
            template_dir=self._template_dir,
            fixture_mode=self._fixture_mode,
        )


class DefaultNarrationAdapter:
    def generate(self, decision: DecisionResult) -> tuple[NarrationEvent, CompanionViewModel, dict[str, Any]]:
        return generate_narration(decision)
