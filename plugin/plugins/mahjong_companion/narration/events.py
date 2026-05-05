from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NarrationEvent:
    event_type: str = "scene_update"
    channel: str = "silent_ui"
    delivery: str = "silent_ui"
    priority: int = 0
    summary: str = ""
    detail: str = ""
    risk_level: str = "unknown"
    scene: str = "unknown"
    buttons: list[str] = field(default_factory=list)
    text: str = ""
    speakable: bool = False
    dedupe_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
