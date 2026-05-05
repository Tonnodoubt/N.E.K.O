from __future__ import annotations

import queue
from typing import Any

CLOSE_MESSAGE = {"__control__": "close"}


def put_latest(target: Any, payload: dict[str, Any], *, max_drops: int = 3) -> None:
    drops = 0
    while True:
        try:
            target.put_nowait(dict(payload))
            return
        except queue.Full:
            if drops >= max_drops:
                return
            drops += 1
            try:
                target.get_nowait()
            except queue.Empty:
                return


def drain_queue(source: Any) -> list[Any]:
    items: list[Any] = []
    while True:
        try:
            items.append(source.get_nowait())
        except queue.Empty:
            return items

