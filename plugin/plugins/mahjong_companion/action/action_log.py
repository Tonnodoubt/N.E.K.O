"""Action log: record and query assist-action audit entries."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..storage import load_json_payload, locked_json_path, write_json_atomic


@dataclass
class ActionLogEntry:
    action_id: str
    executed_at: str
    ok: bool
    blocked_reason: str = ""
    guard_aborted: bool = False
    window_title: str = ""
    trigger_source: str = "manual"
    allow_reason: str = ""
    locator_source: str = ""
    button_region: dict[str, Any] | None = None
    target_x: int | None = None
    target_y: int | None = None
    risk_level: str = "safe"
    confirmation_chain: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_action_log(
    cache_dir: Path,
    entry: ActionLogEntry,
    *,
    max_entries: int = 200,
) -> Path:
    """Append an action log entry to action_log.json, capped at max_entries."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = cache_dir / "action_log.json"

    with locked_json_path(log_path):
        entries = load_json_payload(log_path, default=[], expected_type=list)
        entries = [item for item in entries if isinstance(item, dict)]
        entries.append(entry.to_dict())
        if len(entries) > max_entries:
            entries = entries[-max_entries:]
        write_json_atomic(log_path, entries)
    return log_path


def load_action_log(cache_dir: Path) -> list[dict[str, Any]]:
    """Load all action log entries from cache_dir/action_log.json."""
    log_path = cache_dir / "action_log.json"
    raw = load_json_payload(log_path, default=[], expected_type=list)
    return [_normalize_action_log_entry(item) for item in raw if isinstance(item, dict)]


def clear_action_log(cache_dir: Path) -> bool:
    """Delete the action log file. Returns True if the file existed and was removed."""
    log_path = cache_dir / "action_log.json"
    with locked_json_path(log_path):
        if log_path.exists():
            log_path.unlink()
            return True
        return False


def _normalize_action_log_entry(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    normalized.setdefault("blocked_reason", "")
    normalized.setdefault("guard_aborted", False)
    normalized.setdefault("window_title", "")
    normalized.setdefault("trigger_source", "manual")
    normalized.setdefault("allow_reason", "")
    normalized.setdefault("locator_source", "")
    normalized.setdefault("button_region", None)
    normalized.setdefault("target_x", None)
    normalized.setdefault("target_y", None)
    normalized.setdefault("risk_level", "safe")
    chain = normalized.get("confirmation_chain")
    normalized["confirmation_chain"] = chain if isinstance(chain, list) else []
    return normalized
