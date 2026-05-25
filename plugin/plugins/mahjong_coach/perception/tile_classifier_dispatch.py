from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from .tile_templates import TileTemplateMatch, classify_tile_from_templates


_RED_FIVE_MAP = {"5m": "0m", "5p": "0p", "5s": "0s", "R5m": "0m", "R5p": "0p", "R5s": "0s"}


def classify_hand_tile(
    crop: Image.Image,
    template_payload: dict[str, Any],
) -> TileTemplateMatch | None:
    """Classify a hand tile using legacy templates by default.

    The old ONNX model was strong for discard crops but intentionally opt-in
    for hand tiles, so this new coach keeps hand recognition template-first.
    """
    result = classify_tile_from_templates(crop, template_payload)
    if result is None:
        return None
    return _apply_red_five(crop, result)


def onnx_hand_enabled() -> bool:
    value = os.environ.get("MAHJONG_COACH_ONNX_HAND_ENABLED", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def detect_red_five(crop: Image.Image, classified_tile: str) -> str | None:
    if classified_tile in {"R5m", "R5p", "R5s"}:
        return _RED_FIVE_MAP[classified_tile]
    if classified_tile not in {"5m", "5p", "5s"}:
        return None
    arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
    h, w, _ = arr.shape
    center = arr[int(h * 0.30) : int(h * 0.70), int(w * 0.30) : int(w * 0.70)]
    if center.size == 0:
        return None
    r = center[..., 0]
    g = center[..., 1]
    b = center[..., 2]
    red_mask = (r >= 140) & (r >= g * 1.3) & (r >= b * 1.3)
    red_ratio = float(red_mask.sum()) / max(center.shape[0] * center.shape[1], 1)
    return _RED_FIVE_MAP[classified_tile] if red_ratio >= 0.12 else None


def _apply_red_five(crop: Image.Image, match: TileTemplateMatch) -> TileTemplateMatch:
    red = detect_red_five(crop, match.tile)
    if red is None:
        return match
    return TileTemplateMatch(
        tile=red,
        confidence=match.confidence,
        distance=match.distance,
        runner_up_tile=match.runner_up_tile,
        runner_up_distance=match.runner_up_distance,
    )

