from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import PerceivedGameState
from ..storage import write_json_atomic


def write_debug_artifacts(
    frame_path: Path,
    perceived: PerceivedGameState | None,
    debug_payload: dict[str, Any],
) -> dict[str, str]:
    base_path = frame_path.with_suffix("")
    perception_path = base_path.with_name(base_path.name + "-perception.json")
    overlay_path = base_path.with_name(base_path.name + "-overlay.json")

    write_json_atomic(
        perception_path,
        {
            "frame_path": str(frame_path),
            "perceived_state": perceived.to_dict() if perceived is not None else None,
            "debug": debug_payload,
        },
    )
    write_json_atomic(
        overlay_path,
        {
            "frame_path": str(frame_path),
            "roi_boxes": debug_payload.get("roi_boxes", {}),
            "roi_hits": list(perceived.roi_hits) if perceived is not None else [],
            "button_regions": list(perceived.button_regions) if perceived is not None else [],
            "discard_piles": dict(perceived.discard_piles) if perceived is not None else {},
            "notes": list(perceived.notes) if perceived is not None else [],
        },
    )
    artifacts = {
        "perception_path": str(perception_path),
        "overlay_path": str(overlay_path),
    }
    return artifacts
