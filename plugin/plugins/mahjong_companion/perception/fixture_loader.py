"""Fixture loader — reads *.tiles.json sidecar files for labelled evaluation.

All functions here are internal to the tile-parser subsystem.  The sole
public entry point is :func:`_load_fixture`; the rest are helpers called
from :func:`tile_parser._from_fixture`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_fixture(image_path: Path) -> dict[str, Any] | None:
    candidates = [
        image_path.with_name(f"{image_path.stem}-tiles.json"),
        image_path.with_suffix(".tiles.json"),
        image_path.with_suffix(".label.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("failed to load tile fixture %s: %s", candidate, exc)
            continue
        if isinstance(payload, dict):
            return payload
    return None


def raw_detections_from_label(
    fixture: dict[str, Any],
    hand_tiles: list[str],
) -> list[dict[str, Any]] | None:
    layout = fixture.get("layout")
    if not isinstance(layout, dict):
        return None
    hand_slots = layout.get("hand_slots")
    if not isinstance(hand_slots, list):
        return None

    raw: list[dict[str, Any]] = []
    for index, slot in enumerate(hand_slots):
        if not isinstance(slot, dict):
            continue
        tile = str(slot.get("tile", "")).strip()
        if not tile and index < len(hand_tiles):
            tile = hand_tiles[index]
        raw.append({
            "slot_id": str(slot.get("slot_id", f"hand_{index + 1}")),
            "group": "hand",
            "candidate_tile": tile,
            "confidence": float(fixture.get("analysis_confidence", 0.86) or 0.86),
            "box": slot.get("box") if isinstance(slot.get("box"), dict) else {},
        })
    return raw


def normalize_tile_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_group_list(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    groups: list[list[str]] = []
    for item in value:
        if not isinstance(item, list):
            continue
        group = normalize_tile_list(item)
        if group:
            groups.append(group)
    return groups


def normalize_discard_piles(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return {}
    piles: dict[str, list[dict[str, Any]]] = {}
    for player, raw_items in value.items():
        player_key = str(player).strip()
        if not player_key or not isinstance(raw_items, list):
            continue
        normalized_items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            if not tile:
                continue
            payload: dict[str, Any] = {
                "tile": tile,
                "player": str(item.get("player") or player_key),
                "turn_index": _coerce_int(item.get("turn_index"), default=index + 1),
                "confidence": _coerce_float(item.get("confidence"), default=1.0),
                "orientation": str(item.get("orientation", "")).strip() or player_key,
                "source": str(item.get("source", "")).strip() or "fixture",
            }
            bbox = item.get("bbox")
            if isinstance(bbox, list | tuple) and len(bbox) == 4:
                try:
                    payload["bbox"] = [int(part) for part in bbox]
                except (TypeError, ValueError):
                    pass
            quad = item.get("quad")
            if isinstance(quad, list | tuple) and len(quad) == 4:
                try:
                    payload["quad"] = [[int(point[0]), int(point[1])] for point in quad]
                except (TypeError, ValueError, IndexError):
                    pass
            normalized_items.append(payload)
        if normalized_items:
            piles[player_key] = normalized_items
    return piles


def raw_discard_detections_from_piles(
    discard_piles: dict[str, list[dict[str, Any]]],
    *,
    analysis_confidence: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for player, pile in discard_piles.items():
        for item in pile:
            tile = str(item.get("tile", "")).strip()
            if not tile:
                continue
            detections.append({
                "slot_id": f"discard_{player}_{item.get('turn_index', len(detections) + 1)}",
                "group": "discard",
                "player": player,
                "candidate_tile": tile,
                "confidence": float(item.get("confidence", analysis_confidence) or analysis_confidence),
                "box": item.get("bbox", []),
                "quad": item.get("quad", []),
                "orientation": item.get("orientation", ""),
                "source": item.get("source", "fixture"),
            })
    return detections


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
