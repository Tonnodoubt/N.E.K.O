from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any
from urllib import request

from PIL import Image


PLAYER_ORIENTATIONS = {
    "self": "bottom",
    "left_opponent": "left",
    "top_opponent": "top",
    "right_opponent": "right",
}
ENV_COMMAND = "MAHJONG_COMPANION_DISCARD_RECOGNIZER_CMD"
ENV_ENDPOINT = "MAHJONG_COMPANION_DISCARD_RECOGNIZER_URL"
ENV_TIMEOUT = "MAHJONG_COMPANION_DISCARD_RECOGNIZER_TIMEOUT_SEC"


def load_external_discard_result(
    image_path: Path,
    image: Image.Image,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    command = os.environ.get(ENV_COMMAND, "").strip()
    endpoint = os.environ.get(ENV_ENDPOINT, "").strip()
    if not command and not endpoint:
        return {}, {"external_discard_recognizer_enabled": False}

    timeout = _timeout_seconds()
    try:
        payload = (
            _run_command(command, image_path=image_path, image=image, timeout=timeout)
            if command
            else _post_endpoint(endpoint, image_path=image_path, image=image, timeout=timeout)
        )
        piles = normalize_external_discard_payload(payload)
        return piles, {
            "external_discard_recognizer_enabled": True,
            "external_discard_recognizer_source": "command" if command else "http",
            "external_discard_recognizer_count": sum(len(items) for items in piles.values()),
        }
    except Exception as exc:
        return {}, {
            "external_discard_recognizer_enabled": True,
            "external_discard_recognizer_error": str(exc),
        }


def normalize_external_discard_payload(payload: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {}

    raw_items: list[dict[str, Any]] = []
    raw_piles = payload.get("discard_piles")
    if isinstance(raw_piles, dict):
        for player, pile in raw_piles.items():
            if not isinstance(pile, list):
                continue
            for item in pile:
                if isinstance(item, dict):
                    merged = dict(item)
                    merged.setdefault("player", str(player))
                    raw_items.append(merged)

    detections = payload.get("detections")
    if isinstance(detections, list):
        raw_items.extend(item for item in detections if isinstance(item, dict))

    piles: dict[str, list[dict[str, Any]]] = {}
    for item in raw_items:
        normalized = _normalize_item(item)
        if normalized is None:
            continue
        piles.setdefault(normalized["player"], []).append(normalized)

    for pile in piles.values():
        pile.sort(key=lambda item: int(item["turn_index"]))
    return piles


def _run_command(command: str, *, image_path: Path, image: Image.Image, timeout: float) -> Any:
    formatted = command.format(
        image_path=str(image_path),
        width=image.width,
        height=image.height,
    )
    completed = subprocess.run(
        shlex.split(formatted, posix=os.name != "nt"),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return json.loads(completed.stdout)


def _post_endpoint(endpoint: str, *, image_path: Path, image: Image.Image, timeout: float) -> Any:
    body = json.dumps(
        {
            "image_path": str(image_path),
            "width": image.width,
            "height": image.height,
        },
        ensure_ascii=True,
    ).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_item(item: dict[str, Any]) -> dict[str, Any] | None:
    player = str(item.get("player", "")).strip()
    tile = str(item.get("tile", item.get("candidate_tile", ""))).strip()
    if player not in PLAYER_ORIENTATIONS or not tile:
        return None

    turn_index = _positive_int(item.get("turn_index"))
    if turn_index is None:
        return None

    quad = _normalize_quad(item.get("quad"))
    bbox = _normalize_bbox(item.get("bbox"))
    if bbox is None and quad is not None:
        bbox = _bbox_from_quad(quad)

    normalized: dict[str, Any] = {
        "tile": tile,
        "player": player,
        "turn_index": turn_index,
        "confidence": _float_value(item.get("confidence"), default=1.0),
        "orientation": str(item.get("orientation", "")).strip() or PLAYER_ORIENTATIONS[player],
        "source": str(item.get("source", "")).strip() or "external_discard_recognizer",
    }
    if bbox is not None:
        normalized["bbox"] = bbox
    if quad is not None:
        normalized["quad"] = quad
    return normalized


def _normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [int(part) for part in value]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _normalize_quad(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    points: list[list[int]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        try:
            x, y = [int(part) for part in point]
        except (TypeError, ValueError):
            return None
        points.append([x, y])
    return points


def _bbox_from_quad(quad: list[list[int]]) -> list[int]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timeout_seconds() -> float:
    try:
        return max(0.1, float(os.environ.get(ENV_TIMEOUT, "1.5")))
    except (TypeError, ValueError):
        return 1.5
