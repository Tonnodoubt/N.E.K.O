from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import NarrationEvent
from .view_model import CompanionViewModel
from ..storage import write_json_atomic


def write_debug_artifacts(
    frame_path: Path,
    event: NarrationEvent,
    view_model: CompanionViewModel,
    debug_payload: dict[str, Any],
) -> dict[str, str]:
    base_path = frame_path.with_suffix("")
    narration_path = base_path.with_name(base_path.name + "-narration.json")
    write_json_atomic(
        narration_path,
        {
            "frame_path": str(frame_path),
            "narration_event": event.to_dict(),
            "companion_view_model": view_model.to_dict(),
            "debug": debug_payload,
        },
    )
    return {"narration_path": str(narration_path)}
