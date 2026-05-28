from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image


SIGNATURE_WIDTH = 16
SIGNATURE_HEIGHT = 24
INNER_BOUNDS = (0.06, 0.06, 0.94, 0.82)
DEFAULT_MAX_DISTANCE = 82.0
SUPPORTED_SIGNATURE_VERSIONS = {"rgb-inner-16x24-v1", "rgb-inner-full-16x24-v1"}
_TEMPLATE_MATRIX_CACHE: dict[tuple[Any, ...], tuple[list[str], np.ndarray]] = {}


@dataclass(frozen=True)
class TileTemplateMatch:
    tile: str
    confidence: float
    distance: float
    runner_up_tile: str = ""
    runner_up_distance: float | None = None


def classify_tile_from_templates(crop: Image.Image, payload: dict[str, Any]) -> TileTemplateMatch | None:
    if not _is_usable_template_payload(payload):
        return None
    tiles, matrix = _template_signature_matrix(payload)
    if not tiles or matrix.size == 0:
        return None
    width, height = _payload_signature_size(payload)
    query = np.frombuffer(
        extract_tile_signature(crop, inner_bounds=_payload_inner_bounds(payload), width=width, height=height),
        dtype=np.uint8,
    ).astype(np.int16)
    if query.size != matrix.shape[1]:
        return None
    deltas = matrix - query
    distances = np.sqrt(np.mean(deltas.astype(np.int32) * deltas.astype(np.int32), axis=1))
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    max_distance = float(payload.get("max_rms_distance") or DEFAULT_MAX_DISTANCE)
    if best_distance > max_distance:
        return None
    runner_index = _runner_up_index(distances, best_index, tiles=tiles)
    runner_distance = float(distances[runner_index]) if runner_index is not None else None
    return TileTemplateMatch(
        tile=tiles[best_index],
        confidence=_confidence_from_distances(best_distance, runner_distance, max_distance=max_distance),
        distance=round(best_distance, 3),
        runner_up_tile=tiles[runner_index] if runner_index is not None else "",
        runner_up_distance=round(runner_distance, 3) if runner_distance is not None else None,
    )


def extract_tile_signature(
    crop: Image.Image,
    *,
    inner_bounds: tuple[float, float, float, float] | list[float] | Any = INNER_BOUNDS,
    width: int = SIGNATURE_WIDTH,
    height: int = SIGNATURE_HEIGHT,
) -> bytes:
    left_ratio, top_ratio, right_ratio, bottom_ratio = _normalize_inner_bounds(inner_bounds)
    crop_width, crop_height = crop.size
    inner = crop.crop(
        (
            int(crop_width * left_ratio),
            int(crop_height * top_ratio),
            max(1, int(crop_width * right_ratio)),
            max(1, int(crop_height * bottom_ratio)),
        )
    )
    resized = inner.resize((max(1, int(width)), max(1, int(height)))).convert("RGB")
    return bytes(channel for pixel in resized.getdata() for channel in pixel)


def is_probably_occupied_hand_slot(slot_metrics: dict[str, Any], *, relaxed: bool = False) -> bool:
    mean_luma = _float_metric(slot_metrics, "slot_mean_luma", "mean_luma")
    bright_ratio = _float_metric(slot_metrics, "slot_bright_ratio", "bright_ratio")
    dark_ratio = _float_metric(slot_metrics, "slot_dark_ratio", "dark_ratio")
    stddev = _float_metric(slot_metrics, "slot_stddev", "stddev")
    if relaxed:
        return mean_luma >= 70.0 and stddev >= 10.0 and dark_ratio <= 0.75
    normal_tile = mean_luma >= 95.0 and bright_ratio >= 0.16 and dark_ratio <= 0.55 and stddev >= 18.0
    dimmed_tile = mean_luma >= 100.0 and dark_ratio <= 0.38 and stddev >= 30.0
    return normal_tile or dimmed_tile


def _is_usable_template_payload(payload: dict[str, Any]) -> bool:
    return (
        payload.get("version") == "mahjong-hand-template-v1"
        and payload.get("signature_version") in SUPPORTED_SIGNATURE_VERSIONS
        and isinstance(payload.get("templates"), dict)
    )


def _template_signature_matrix(payload: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    fingerprint = _payload_fingerprint(payload)
    cached = _TEMPLATE_MATRIX_CACHE.get(fingerprint)
    if cached is not None:
        return cached
    rows: list[np.ndarray] = []
    tiles: list[str] = []
    for tile, signatures in _iter_template_signatures(payload):
        for signature in signatures:
            row = np.frombuffer(signature, dtype=np.uint8)
            if row.size == _payload_signature_length(payload):
                rows.append(row)
                tiles.append(tile)
    matrix = np.vstack(rows).astype(np.int16) if rows else np.empty((0, _payload_signature_length(payload)), dtype=np.int16)
    _TEMPLATE_MATRIX_CACHE[fingerprint] = (tiles, matrix)
    return tiles, matrix


def _iter_template_signatures(payload: dict[str, Any]) -> Iterable[tuple[str, list[bytes]]]:
    templates = payload.get("templates")
    if not isinstance(templates, dict):
        return
    expected_length = _payload_signature_length(payload)
    for tile, item in sorted(templates.items()):
        if not isinstance(item, dict):
            continue
        raw_signatures = item.get("signatures")
        if not isinstance(raw_signatures, list):
            continue
        signatures = [_decode_signature(value, expected_length=expected_length) for value in raw_signatures]
        signatures = [value for value in signatures if value]
        if signatures:
            yield str(tile), signatures


def _decode_signature(value: Any, *, expected_length: int) -> bytes:
    if not isinstance(value, str) or not value:
        return b""
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, TypeError):
        return b""
    return decoded if len(decoded) == expected_length else b""


def _payload_signature_size(payload: dict[str, Any]) -> tuple[int, int]:
    return int(payload.get("width") or SIGNATURE_WIDTH), int(payload.get("height") or SIGNATURE_HEIGHT)


def _payload_signature_length(payload: dict[str, Any]) -> int:
    width, height = _payload_signature_size(payload)
    return width * height * 3


def _payload_inner_bounds(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    return _normalize_inner_bounds(payload.get("inner_bounds") or INNER_BOUNDS)


def _normalize_inner_bounds(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return INNER_BOUNDS
    left, top, right, bottom = (float(item) for item in value)
    return (
        max(0.0, min(0.95, left)),
        max(0.0, min(0.95, top)),
        max(0.05, min(1.0, right)),
        max(0.05, min(1.0, bottom)),
    )


def _payload_fingerprint(payload: dict[str, Any]) -> tuple[Any, ...]:
    templates = payload.get("templates")
    return (
        str(payload.get("signature_version") or ""),
        _payload_signature_size(payload),
        _payload_inner_bounds(payload),
        len(templates) if isinstance(templates, dict) else 0,
        int(payload.get("source_sample_count") or 0),
        int(payload.get("stored_sample_count") or 0),
    )


def _runner_up_index(distances: np.ndarray, best_index: int, *, tiles: list[str]) -> int | None:
    if distances.size <= 1:
        return None
    best_tile = tiles[best_index] if 0 <= best_index < len(tiles) else ""
    for index in np.argsort(distances):
        candidate = int(index)
        if candidate != best_index and tiles[candidate] != best_tile:
            return candidate
    return None


def _confidence_from_distances(best: float, runner: float | None, *, max_distance: float) -> float:
    base = max(0.0, min(1.0, 1.0 - best / max(max_distance, 1.0)))
    if runner is None:
        return round(base, 4)
    margin = max(0.0, min(1.0, (runner - best) / max(max_distance, 1.0)))
    return round(max(base, min(1.0, base * 0.75 + margin * 0.55)), 4)


def _float_metric(metrics: dict[str, Any], *names: str) -> float:
    for name in names:
        try:
            return float(metrics.get(name))
        except (TypeError, ValueError):
            continue
    return 0.0
