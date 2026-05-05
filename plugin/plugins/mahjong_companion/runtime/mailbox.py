from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeInboundMessage:
    message_id: str
    source: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    interrupt: bool = False
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeOutboundMessage:
    message_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeMailbox:
    """In-process mailbox for catgirl->game and game->catgirl communication."""

    def __init__(self, *, max_inbound: int = 32, max_outbound: int = 128) -> None:
        self._max_inbound = max(1, int(max_inbound))
        self._max_outbound = max(1, int(max_outbound))
        self._inbound: deque[RuntimeInboundMessage] = deque()
        self._outbound: deque[RuntimeOutboundMessage] = deque()
        self._dropped_inbound = 0
        self._dropped_outbound = 0
        self._last_inbound_id = ""
        self._last_outbound_id = ""

    def enqueue_inbound(
        self,
        *,
        action: str,
        payload: dict[str, Any] | None = None,
        source: str = "catgirl",
        interrupt: bool = False,
    ) -> RuntimeInboundMessage:
        if interrupt:
            self._inbound.clear()

        message = RuntimeInboundMessage(
            message_id=f"inbound-{uuid4().hex[:12]}",
            source=str(source).strip() or "catgirl",
            action=str(action).strip(),
            payload=dict(payload or {}),
            interrupt=bool(interrupt),
        )
        if len(self._inbound) >= self._max_inbound:
            self._inbound.popleft()
            self._dropped_inbound += 1
        self._inbound.append(message)
        self._last_inbound_id = message.message_id
        return message

    def pop_inbound(self) -> RuntimeInboundMessage | None:
        if not self._inbound:
            return None
        return self._inbound.popleft()

    def enqueue_outbound(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeOutboundMessage:
        message = RuntimeOutboundMessage(
            message_id=f"outbound-{uuid4().hex[:12]}",
            event_type=str(event_type).strip() or "runtime_event",
            payload=dict(payload or {}),
        )
        if len(self._outbound) >= self._max_outbound:
            self._outbound.popleft()
            self._dropped_outbound += 1
        self._outbound.append(message)
        self._last_outbound_id = message.message_id
        return message

    def pop_outbound_batch(self, *, limit: int = 1) -> list[RuntimeOutboundMessage]:
        max_items = max(1, int(limit))
        items: list[RuntimeOutboundMessage] = []
        for _ in range(max_items):
            if not self._outbound:
                break
            items.append(self._outbound.popleft())
        return items

    def clear_inbound(self) -> None:
        self._inbound.clear()

    def clear_outbound(self) -> None:
        self._outbound.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "inbound_pending": len(self._inbound),
            "outbound_pending": len(self._outbound),
            "dropped_inbound": self._dropped_inbound,
            "dropped_outbound": self._dropped_outbound,
            "last_inbound_id": self._last_inbound_id,
            "last_outbound_id": self._last_outbound_id,
        }
