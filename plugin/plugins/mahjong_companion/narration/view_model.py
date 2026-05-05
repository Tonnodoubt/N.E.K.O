from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CompanionViewModel:
    headline: str = ""
    subline: str = ""
    mood: str = "calm"
    suggestion_level: str = "silent"
    speakable: bool = False
    delivery: str = "silent_ui"
    text: str = ""
    detail_collapsed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
