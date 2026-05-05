from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import DecisionResult
from ..storage import write_json_atomic


def write_debug_artifacts(
    frame_path: Path,
    decision: DecisionResult,
    debug_payload: dict[str, Any],
) -> dict[str, str]:
    base_path = frame_path.with_suffix("")
    decision_path = base_path.with_name(base_path.name + "-decision.json")
    write_json_atomic(
        decision_path,
        {
            "frame_path": str(frame_path),
            "decision_result": decision.to_dict(),
            "debug": debug_payload,
        },
    )
    return {"decision_path": str(decision_path)}
