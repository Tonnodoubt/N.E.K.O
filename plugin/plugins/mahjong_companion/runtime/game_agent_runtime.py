from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .inbox import RuntimeInbox, RuntimeInboxMessage
from .outbox import RuntimeOutbox, RuntimeOutboxMessage


@dataclass
class GameAgentRuntimeConfig:
    mode: str = "active"
    inbound_queue_limit: int = 32
    outbound_queue_limit: int = 128
    outbound_flush_per_tick: int = 1
    outbound_dedupe_window_sec: int = 8


class GameAgentRuntime:
    """Background runtime for game-agent loop and mailbox semantics."""

    def __init__(self, config: GameAgentRuntimeConfig | None = None) -> None:
        self._config = config or GameAgentRuntimeConfig()
        self._mode = self._normalize_mode(self._config.mode)
        self._status = "idle"
        self._inbox = RuntimeInbox(max_pending=self._config.inbound_queue_limit)
        self._outbox = RuntimeOutbox(
            max_pending=self._config.outbound_queue_limit,
            dedupe_window_sec=self._config.outbound_dedupe_window_sec,
            throttle_per_tick=self._config.outbound_flush_per_tick,
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def status(self) -> str:
        return self._status

    @property
    def flush_per_tick(self) -> int:
        return self._config.outbound_flush_per_tick

    def set_mode(self, mode: str) -> str:
        self._mode = self._normalize_mode(mode)
        return self._mode

    def set_status(self, status: str) -> str:
        self._status = str(status).strip() or "idle"
        return self._status

    def configure(self, config: GameAgentRuntimeConfig) -> None:
        mode = self._mode
        status = self._status
        self._config = config
        self._mode = self._normalize_mode(config.mode or mode)
        self._status = status
        self._inbox.configure(max_pending=config.inbound_queue_limit)
        self._outbox.configure(
            max_pending=config.outbound_queue_limit,
            dedupe_window_sec=config.outbound_dedupe_window_sec,
            throttle_per_tick=config.outbound_flush_per_tick,
        )

    def enqueue_inbound(
        self,
        *,
        action: str,
        payload: dict[str, Any] | None = None,
        source: str = "catgirl",
        interrupt: bool = False,
    ) -> RuntimeInboxMessage:
        return self._inbox.enqueue(
            action=action,
            payload=payload,
            source=source,
            interrupt=interrupt,
        )

    def pop_inbound(self) -> RuntimeInboxMessage | None:
        return self._inbox.pop()

    def enqueue_outbound(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        dedupe_key: str = "",
    ) -> RuntimeOutboxMessage | None:
        return self._outbox.enqueue(
            event_type=event_type,
            payload=payload,
            priority=priority,
            dedupe_key=dedupe_key,
        )

    def pop_outbound_batch(self, *, limit: int | None = None) -> list[RuntimeOutboxMessage]:
        return self._outbox.pop_batch(limit=limit)

    def snapshot(self) -> dict[str, Any]:
        inbox = self._inbox.snapshot()
        outbox = self._outbox.snapshot()
        return {
            "mode": self._mode,
            "status": self._status,
            "inbox": inbox,
            "outbox": outbox,
            "inbound_pending": inbox["pending"],
            "outbound_pending": outbox["pending"],
            "dropped_inbound": inbox["dropped"],
            "dropped_outbound": outbox["dropped"],
            "deduped_outbound": outbox["deduped"],
            "last_inbound_id": inbox["last_message_id"],
            "last_outbound_id": outbox["last_message_id"],
            "interrupts_inbound": inbox["interrupts"],
            "outbound_flush_per_tick": outbox["throttle_per_tick"],
            "outbound_dedupe_window_sec": outbox["dedupe_window_sec"],
        }

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"active", "standby", "off"}:
            return "active"
        return normalized
