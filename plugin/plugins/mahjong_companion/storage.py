from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None  # type: ignore[assignment]


_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


def _thread_lock_for(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[resolved] = lock
        return lock


@contextmanager
def locked_json_path(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    thread_lock = _thread_lock_for(lock_path)
    with thread_lock:
        with lock_path.open("a", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_json_payload(
    path: Path,
    *,
    default: Any,
    expected_type: type | tuple[type, ...] = dict,
    logger: Any | None = None,
) -> Any:
    if not path.exists():
        return _copy_default(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _log_warning(logger, "corrupted JSON at %s: %s", path, exc)
        return _copy_default(default)
    except OSError as exc:
        _log_warning(logger, "failed to read %s: %s", path, exc)
        return _copy_default(default)
    if not isinstance(payload, expected_type):
        _log_warning(logger, "unexpected JSON payload type at %s: %s", path, type(payload).__name__)
        return _copy_default(default)
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def update_json_payload(
    path: Path,
    *,
    default: Any,
    expected_type: type | tuple[type, ...] = dict,
    updater: Callable[[Any], Any],
    logger: Any | None = None,
) -> Any:
    with locked_json_path(path):
        payload = load_json_payload(path, default=default, expected_type=expected_type, logger=logger)
        updated = updater(payload)
        write_json_atomic(path, updated)
        return updated


def _copy_default(default: Any) -> Any:
    return copy.deepcopy(default)


def _log_warning(logger: Any | None, message: str, *args: Any) -> None:
    if logger is None:
        return
    warning = getattr(logger, "warning", None)
    if callable(warning):
        try:
            text = message % args if args else message
        except Exception:
            text = message
        warning(text)
