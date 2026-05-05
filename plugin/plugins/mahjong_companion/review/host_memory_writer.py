from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class HostMemoryWriteResult:
    ok: bool
    status: str
    writer: str = ""
    error: str = ""


class HostMemoryWriter(Protocol):
    writer_name: str
    available: bool

    def write(self, bucket_id: str, payload: dict[str, Any]) -> HostMemoryWriteResult:
        ...


class UnavailableHostMemoryWriter:
    writer_name = ""
    available = False

    def write(self, bucket_id: str, payload: dict[str, Any]) -> HostMemoryWriteResult:
        return HostMemoryWriteResult(
            ok=False,
            status="host_memory_write_unavailable",
            writer=self.writer_name,
            error="sdk_memory_write_unavailable",
        )


class SdkHostMemoryWriter:
    def __init__(self, memory_client: Any | None) -> None:
        self._writer, self.writer_name = self._resolve_writer(memory_client)
        self.available = self._writer is not None

    def write(self, bucket_id: str, payload: dict[str, Any]) -> HostMemoryWriteResult:
        if self._writer is None:
            return UnavailableHostMemoryWriter().write(bucket_id, payload)
        try:
            result = self._writer(bucket_id=bucket_id, payload=payload)
        except Exception as exc:
            return HostMemoryWriteResult(
                ok=False,
                status="host_memory_write_failed",
                writer=self.writer_name,
                error=str(exc),
            )
        return HostMemoryWriteResult(
            ok=bool(result) or result is None,
            status="host_memory_write_complete" if (bool(result) or result is None) else "host_memory_write_rejected",
            writer=self.writer_name,
            error="" if (bool(result) or result is None) else "writer_returned_false",
        )

    @staticmethod
    def _resolve_writer(memory_client: Any | None) -> tuple[Callable[..., Any] | None, str]:
        if memory_client is None:
            return None, ""
        for name in ("put", "append", "write", "upsert", "add"):
            candidate = getattr(memory_client, name, None)
            if callable(candidate):
                return candidate, name
        return None, ""
