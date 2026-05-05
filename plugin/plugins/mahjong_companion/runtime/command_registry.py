from __future__ import annotations

from typing import Any, Callable


RuntimeCommandHandler = Callable[[dict[str, Any]], dict[str, Any]]


class RuntimeCommandRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, RuntimeCommandHandler] = {}

    def register(self, name: str, handler: RuntimeCommandHandler) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("command name must be non-empty")
        if normalized in self._handlers:
            raise ValueError(f"command already registered: {normalized}")
        self._handlers[normalized] = handler

    def dispatch(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = str(name).strip().lower()
        if not normalized:
            return {"ok": False, "error": "runtime action is empty"}
        handler = self._handlers.get(normalized)
        if handler is None:
            return {"ok": False, "error": f"unsupported runtime action: {name}"}
        return handler(payload)

    def known_commands(self) -> list[str]:
        return sorted(self._handlers)
