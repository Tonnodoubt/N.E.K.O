from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class RuntimeOutboxMessage:
    message_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dedupe_key: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeOutbox:
    """Game -> catgirl queue with priority, throttle and dedupe semantics."""

    def __init__(
        self,
        *,
        max_pending: int = 128,
        dedupe_window_sec: int = 8,
        throttle_per_tick: int = 1,
    ) -> None:
        self._max_pending = max(1, int(max_pending))
        self._dedupe_window_sec = max(0, int(dedupe_window_sec))
        self._throttle_per_tick = max(1, int(throttle_per_tick))
        self._queue: list[RuntimeOutboxMessage] = []
        self._dropped = 0
        self._deduped = 0
        self._last_message_id = ""
        self._last_sent_at_by_key: dict[str, str] = {}

    @property
    def throttle_per_tick(self) -> int:
        return self._throttle_per_tick

    def configure(
        self,
        *,
        max_pending: int,
        dedupe_window_sec: int,
        throttle_per_tick: int,
    ) -> None:
        self._max_pending = max(1, int(max_pending))
        self._dedupe_window_sec = max(0, int(dedupe_window_sec))
        self._throttle_per_tick = max(1, int(throttle_per_tick))
        while len(self._queue) > self._max_pending:
            self._queue.pop(0)
            self._dropped += 1
        self._prune_sent_dedupe_keys(datetime.now(timezone.utc))

    def enqueue(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        dedupe_key: str = "",
    ) -> RuntimeOutboxMessage | None:
        normalized_key = str(dedupe_key).strip()
        now = datetime.now(timezone.utc)
        self._prune_sent_dedupe_keys(now)

        if normalized_key:
            for item in self._queue:
                if item.dedupe_key == normalized_key:
                    self._deduped += 1
                    return None

        if normalized_key and self._dedupe_window_sec > 0:
            sent_at = _parse_iso(self._last_sent_at_by_key.get(normalized_key, ""))
            if sent_at is not None:
                delta = (now - sent_at).total_seconds()
                if delta < self._dedupe_window_sec:
                    self._deduped += 1
                    return None

        message = RuntimeOutboxMessage(
            message_id=f"outbound-{uuid4().hex[:12]}",
            event_type=str(event_type).strip() or "runtime_event",
            payload=dict(payload or {}),
            priority=int(priority),
            dedupe_key=normalized_key,
            created_at=now.isoformat(),
        )

        if len(self._queue) >= self._max_pending:
            self._queue.pop(0)
            self._dropped += 1

        self._queue.append(message)
        self._last_message_id = message.message_id
        return message

    def pop_batch(self, *, limit: int | None = None) -> list[RuntimeOutboxMessage]:
        if not self._queue:
            return []
        max_items = self._throttle_per_tick if limit is None else max(1, int(limit))
        max_items = min(max_items, len(self._queue))
        self._queue.sort(
            key=lambda item: (
                -item.priority,
                _parse_iso(item.created_at) or datetime.min.replace(tzinfo=timezone.utc),
            )
        )
        selected = self._queue[:max_items]
        self._queue = self._queue[max_items:]

        for item in selected:
            if item.dedupe_key and self._dedupe_window_sec > 0:
                self._last_sent_at_by_key[item.dedupe_key] = _now_iso()

        return selected

    def _prune_sent_dedupe_keys(self, now: datetime) -> None:
        if self._dedupe_window_sec <= 0:
            self._last_sent_at_by_key.clear()
            return
        if not self._last_sent_at_by_key:
            return
        stale_keys: list[str] = []
        for key, sent_at_raw in self._last_sent_at_by_key.items():
            sent_at = _parse_iso(sent_at_raw)
            if sent_at is None or (now - sent_at).total_seconds() >= self._dedupe_window_sec:
                stale_keys.append(key)
        for key in stale_keys:
            self._last_sent_at_by_key.pop(key, None)

    def clear(self) -> None:
        self._queue.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending": len(self._queue),
            "dropped": self._dropped,
            "deduped": self._deduped,
            "last_message_id": self._last_message_id,
            "max_pending": self._max_pending,
            "throttle_per_tick": self._throttle_per_tick,
            "dedupe_window_sec": self._dedupe_window_sec,
        }
