from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from PIL import Image


TEMPLATE_PAYLOAD_VERSION = "mahjong-hand-template-v1"
SIGNATURE_VERSION = "rgb-inner-16x24-v1"
FULL_TILE_SIGNATURE_VERSION = "rgb-inner-full-16x24-v1"
SIGNATURE_WIDTH = 16
SIGNATURE_HEIGHT = 24
INNER_BOUNDS = (0.06, 0.06, 0.94, 0.82)
FULL_TILE_INNER_BOUNDS = (0.04, 0.04, 0.96, 0.96)
DEFAULT_MAX_DISTANCE = 82.0
DEFAULT_MAX_SAMPLES_PER_TILE = 12
SUPPORTED_SIGNATURE_VERSIONS = {SIGNATURE_VERSION, FULL_TILE_SIGNATURE_VERSION}
_TEMPLATE_MATRIX_CACHE: dict[int, tuple[tuple[Any, ...], list[str], np.ndarray]] = {}


@dataclass(frozen=True)
class TileTemplateMatch:
    tile: str
    confidence: float
    distance: float
    runner_up_tile: str = ""
    runner_up_distance: float | None = None


def build_hand_tile_template_payload(
    samples: Iterable[tuple[str, Image.Image]],
    *,
    max_samples_per_tile: int = DEFAULT_MAX_SAMPLES_PER_TILE,
    inner_bounds: tuple[float, float, float, float] | list[float] = INNER_BOUNDS,
    signature_version: str = SIGNATURE_VERSION,
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    source_sample_count = 0
    normalized_inner_bounds = _normalize_inner_bounds(inner_bounds)
    normalized_signature_version = str(signature_version or SIGNATURE_VERSION).strip() or SIGNATURE_VERSION
    for tile, crop in samples:
        normalized = str(tile).strip()
        if not normalized:
            continue
        source_sample_count += 1
        encoded = _encode_signature(extract_tile_signature(crop, inner_bounds=normalized_inner_bounds))
        grouped.setdefault(normalized, []).append(encoded)

    if not grouped:
        return {}

    max_samples_per_tile = max(1, int(max_samples_per_tile))
    templates: dict[str, dict[str, Any]] = {}
    stored_sample_count = 0
    for tile, signatures in sorted(grouped.items()):
        stored = signatures[:max_samples_per_tile]
        stored_sample_count += len(stored)
        templates[tile] = {
            "count": len(signatures),
            "signatures": stored,
        }

    return {
        "version": TEMPLATE_PAYLOAD_VERSION,
        "signature_version": normalized_signature_version,
        "width": SIGNATURE_WIDTH,
        "height": SIGNATURE_HEIGHT,
        "inner_bounds": list(normalized_inner_bounds),
        "max_rms_distance": DEFAULT_MAX_DISTANCE,
        "max_samples_per_tile": max_samples_per_tile,
        "source_sample_count": source_sample_count,
        "stored_sample_count": stored_sample_count,
        "templates": templates,
    }


def extract_tile_signature(
    crop: Image.Image,
    *,
    inner_bounds: tuple[float, float, float, float] | list[float] | Any = INNER_BOUNDS,
    width: int = SIGNATURE_WIDTH,
    height: int = SIGNATURE_HEIGHT,
) -> bytes:
    signature_width = max(1, int(width or SIGNATURE_WIDTH))
    signature_height = max(1, int(height or SIGNATURE_HEIGHT))
    crop_width, crop_height = crop.size
    left_ratio, top_ratio, right_ratio, bottom_ratio = _normalize_inner_bounds(inner_bounds)
    inner = crop.crop(
        (
            int(crop_width * left_ratio),
            int(crop_height * top_ratio),
            max(1, int(crop_width * right_ratio)),
            max(1, int(crop_height * bottom_ratio)),
        ),
    )
    resized = inner.resize((signature_width, signature_height)).convert("RGB")
    return bytes(channel for pixel in resized.getdata() for channel in pixel)


def classify_tile_from_templates(crop: Image.Image, payload: dict[str, Any]) -> TileTemplateMatch | None:
    if not _is_usable_template_payload(payload):
        return None

    tiles, matrix = _template_signature_matrix(payload)
    if not tiles or matrix.size == 0:
        return None

    signature_width, signature_height = _payload_signature_size(payload)
    query = np.frombuffer(
        extract_tile_signature(
            crop,
            inner_bounds=_payload_inner_bounds(payload),
            width=signature_width,
            height=signature_height,
        ),
        dtype=np.uint8,
    ).astype(np.int16)
    if query.size != matrix.shape[1]:
        return None
    deltas = matrix - query
    distances = np.sqrt(np.mean(deltas.astype(np.int32) * deltas.astype(np.int32), axis=1))
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    max_distance = _float_payload_value(payload, "max_rms_distance", DEFAULT_MAX_DISTANCE)
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


def is_probably_occupied_hand_slot(slot_metrics: dict[str, Any]) -> bool:
    mean_luma = _float_metric(slot_metrics, "slot_mean_luma", "mean_luma")
    bright_ratio = _float_metric(slot_metrics, "slot_bright_ratio", "bright_ratio")
    dark_ratio = _float_metric(slot_metrics, "slot_dark_ratio", "dark_ratio")
    stddev = _float_metric(slot_metrics, "slot_stddev", "stddev")
    return mean_luma >= 95.0 and bright_ratio >= 0.16 and dark_ratio <= 0.55 and stddev >= 18.0


def _is_usable_template_payload(payload: dict[str, Any]) -> bool:
    return (
        payload.get("version") == TEMPLATE_PAYLOAD_VERSION
        and payload.get("signature_version") in SUPPORTED_SIGNATURE_VERSIONS
        and isinstance(payload.get("templates"), dict)
    )


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


def _template_signature_matrix(payload: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    fingerprint = _payload_fingerprint(payload)
    cached = _TEMPLATE_MATRIX_CACHE.get(fingerprint)
    if cached is not None:
        return cached[0], cached[1]

    tiles: list[str] = []
    rows: list[np.ndarray] = []
    for tile, signatures in _iter_template_signatures(payload):
        for signature in signatures:
            row = np.frombuffer(signature, dtype=np.uint8)
            if row.size != _payload_signature_length(payload):
                continue
            tiles.append(tile)
            rows.append(row)

    matrix = (
        np.vstack(rows).astype(np.int16)
        if rows
        else np.empty((0, SIGNATURE_WIDTH * SIGNATURE_HEIGHT * 3), dtype=np.int16)
    )
    _TEMPLATE_MATRIX_CACHE[fingerprint] = (tiles, matrix)
    return tiles, matrix


def _payload_fingerprint(payload: dict[str, Any]) -> tuple[Any, ...]:
    templates = payload.get("templates")
    template_count = len(templates) if isinstance(templates, dict) else 0
    return (
        str(payload.get("signature_version", "")),
        _payload_signature_size(payload),
        tuple(_payload_inner_bounds(payload)),
        template_count,
        int(payload.get("source_sample_count", 0) or 0),
        int(payload.get("stored_sample_count", 0) or 0),
    )


def _runner_up_index(distances: np.ndarray, best_index: int, *, tiles: list[str]) -> int | None:
    if distances.size <= 1:
        return None
    best_tile = tiles[best_index] if 0 <= best_index < len(tiles) else ""
    ordered = np.argsort(distances)
    for index in ordered:
        clean_index = int(index)
        if clean_index == best_index:
            continue
        if 0 <= clean_index < len(tiles) and tiles[clean_index] == best_tile:
            continue
        return clean_index
    return None


def _rms_distance(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        return math.inf
    total = 0
    for left_value, right_value in zip(left, right, strict=True):
        delta = int(left_value) - int(right_value)
        total += delta * delta
    return math.sqrt(total / len(left))


def _confidence_from_distances(best: float, runner_up: float | None, *, max_distance: float) -> float:
    if best <= 0.001:
        return 0.99
    distance_score = max(0.0, min(1.0, 1.0 - best / max(1.0, max_distance)))
    margin_score = 0.0
    if runner_up is not None and math.isfinite(runner_up):
        margin_score = max(0.0, min(1.0, (runner_up - best) / max(1.0, runner_up)))
    return round(max(0.01, min(0.99, distance_score * 0.72 + margin_score * 0.28)), 3)


def _encode_signature(signature: bytes) -> str:
    return base64.b64encode(signature).decode("ascii")


def _decode_signature(value: Any, *, expected_length: int | None = None) -> bytes:
    if not isinstance(value, str) or not value:
        return b""
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, OSError):
        return b""
    expected_length = expected_length or SIGNATURE_WIDTH * SIGNATURE_HEIGHT * 3
    if len(decoded) != expected_length:
        return b""
    return decoded


def _float_payload_value(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _payload_signature_size(payload: dict[str, Any]) -> tuple[int, int]:
    width = _int_payload_value(payload, "width", SIGNATURE_WIDTH)
    height = _int_payload_value(payload, "height", SIGNATURE_HEIGHT)
    return max(1, width), max(1, height)


def _payload_signature_length(payload: dict[str, Any]) -> int:
    width, height = _payload_signature_size(payload)
    return width * height * 3


def _payload_inner_bounds(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    return _normalize_inner_bounds(payload.get("inner_bounds", INNER_BOUNDS))


def _normalize_inner_bounds(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return INNER_BOUNDS
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError):
        return INNER_BOUNDS
    left = max(0.0, min(0.95, left))
    top = max(0.0, min(0.95, top))
    right = max(left + 0.01, min(1.0, right))
    bottom = max(top + 0.01, min(1.0, bottom))
    return (left, top, right, bottom)


def _int_payload_value(payload: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(payload.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _float_metric(metrics: dict[str, Any], primary: str, fallback: str) -> float:
    value = metrics.get(primary, metrics.get(fallback, 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
