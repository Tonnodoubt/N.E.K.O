from __future__ import annotations

from typing import Protocol

from ..contracts import DecisionResult, PerceivedGameState
from .generator import build_decision


class DecisionAdapter(Protocol):
    def suggest(self, state: PerceivedGameState) -> DecisionResult:
        ...


class DefaultDecisionAdapter:
    def suggest(self, state: PerceivedGameState) -> DecisionResult:
        return build_decision(state)
