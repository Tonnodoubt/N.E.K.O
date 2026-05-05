from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeInboxMessage:
    message_id: str
    source: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    interrupt: bool = False
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeInbox:
    """Catgirl -> game runtime message queue with interrupt semantics."""

    def __init__(self, *, max_pending: int = 32) -> None:
        self._max_pending = max(1, int(max_pending))
        self._queue: deque[RuntimeInboxMessage] = deque()
        self._dropped = 0
        self._interrupts = 0
        self._last_message_id = ""

    @property
    def max_pending(self) -> int:
        return self._max_pending

    def configure(self, *, max_pending: int) -> None:
        self._max_pending = max(1, int(max_pending))
        while len(self._queue) > self._max_pending:
            self._queue.popleft()
            self._dropped += 1

    def enqueue(
        self,
        *,
        action: str,
        payload: dict[str, Any] | None = None,
        source: str = "catgirl",
        interrupt: bool = False,
    ) -> RuntimeInboxMessage:
        if interrupt:
            if self._queue:
                self._dropped += len(self._queue)
            self._queue.clear()
            self._interrupts += 1

        message = RuntimeInboxMessage(
            message_id=f"inbound-{uuid4().hex[:12]}",
            source=str(source).strip() or "catgirl",
            action=str(action).strip(),
            payload=dict(payload or {}),
            interrupt=bool(interrupt),
        )
        if len(self._queue) >= self._max_pending:
            self._queue.popleft()
            self._dropped += 1
        self._queue.append(message)
        self._last_message_id = message.message_id
        return message

    def pop(self) -> RuntimeInboxMessage | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def clear(self) -> None:
        self._queue.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending": len(self._queue),
            "dropped": self._dropped,
            "interrupts": self._interrupts,
            "last_message_id": self._last_message_id,
            "max_pending": self._max_pending,
        }
