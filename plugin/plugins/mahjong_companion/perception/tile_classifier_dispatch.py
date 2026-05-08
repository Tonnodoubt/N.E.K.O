"""Dispatch layer for tile classification.

Selects between ONNX neural-network inference and template matching.
Resolution order: ONNX (if model artifacts exist) → template matching.

Call sites import :func:`classify_tile` instead of calling
:func:`tile_templates.classify_tile_from_templates` directly, so the
backend switch is transparent to downstream code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .vit_tile_classifier_onnx import classify_tile_crops_onnx, vit_onnx_available
from .tile_templates import (
    DEFAULT_MAX_DISTANCE,
    TileTemplateMatch,
    classify_tile_from_templates,
)

_ONNX_PROBED: bool | None = None
_ONNX_READY: bool = False
_ONNX_LOCK = __import__("threading").Lock()

# Red-five tiles that the 34-class model does not know about.
_RED_FIVE_MAP = {"5m": "0m", "5p": "0p", "5s": "0s"}


def onnx_backend_available() -> bool:
    """True if the ONNX tile classifier is ready for inference."""
    global _ONNX_PROBED, _ONNX_READY
    if _ONNX_PROBED is None:
        with _ONNX_LOCK:
            if _ONNX_PROBED is None:
                _ONNX_READY = vit_onnx_available()
                _ONNX_PROBED = True
    return _ONNX_READY


# Legacy alias kept for internal callers that haven't migrated yet.
_onnx_ready = onnx_backend_available


def classify_tile(
    crop: Image.Image,
    template_payload: dict[str, Any],
) -> TileTemplateMatch | None:
    """Classify a single tile crop. ONNX first, templates fallback."""
    if _onnx_ready():
        onnx_result = _onnx_single(crop)
        if onnx_result is not None:
            return onnx_result
    return classify_tile_from_templates(crop, template_payload)


def classify_tiles_batch(
    crops: list[Image.Image],
    template_payload: dict[str, Any],
) -> list[TileTemplateMatch | None]:
    """Classify multiple tile crops. ONNX batch path, templates per-crop fallback."""
    if not crops:
        return []
    if _onnx_ready():
        onnx_results = _onnx_batch(crops)
        if onnx_results is not None:
            return onnx_results
    return [classify_tile_from_templates(c, template_payload) for c in crops]


def _onnx_single(crop: Image.Image) -> TileTemplateMatch | None:
    results = classify_tile_crops_onnx([crop], top_k=2)
    if not results or results[0] is None:
        return None
    return _to_template_match(results[0])


def _onnx_batch(crops: list[Image.Image]) -> list[TileTemplateMatch | None] | None:
    results = classify_tile_crops_onnx(crops, top_k=2)
    if not results:
        return None
    converted = []
    for r in results:
        converted.append(_to_template_match(r) if r is not None else None)
    return converted


def _to_template_match(pred: Any) -> TileTemplateMatch:
    runner_up_tile = ""
    runner_up_distance = None
    top_k = getattr(pred, "top_k", []) or []
    if len(top_k) > 1:
        runner_up_tile = str(top_k[1].get("tile", ""))
        runner_up_distance = (1.0 - float(top_k[1].get("score", 0.0))) * DEFAULT_MAX_DISTANCE
    confidence = float(getattr(pred, "confidence", 0.0))
    return TileTemplateMatch(
        tile=str(getattr(pred, "tile", "")),
        confidence=confidence,
        distance=(1.0 - confidence) * DEFAULT_MAX_DISTANCE,
        runner_up_tile=runner_up_tile,
        runner_up_distance=runner_up_distance,
    )


def classify_hand_tile(
    crop: Image.Image,
    template_payload: dict[str, Any],
) -> TileTemplateMatch | None:
    """Like :func:`classify_tile` but with red-five colour post-processing.

    Only call this for hand tiles — discard tiles are never red fives.
    """
    result = classify_tile(crop, template_payload)
    if result is not None:
        return _apply_red_five(crop, result)
    return None


def detect_red_five(crop: Image.Image, classified_tile: str) -> str | None:
    """Return ``"0m"``, ``"0p"``, or ``"0s"`` if *crop* looks like the
    corresponding red-five variant, otherwise ``None``.

    The 34-class model cannot output red-five labels, so this color-based
    check runs as a post-classification overlay.  It inspects a small
    central window of the tile and tests for red-dominant pixels (the
    hallmark of aka-dora tiles in Mahjong Soul).
    """
    if classified_tile not in _RED_FIVE_MAP:
        return None
    arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
    h, w, _ = arr.shape
    # Inspect the central 40% of the tile – where the numeral / symbol lives.
    cx0 = int(w * 0.30)
    cx1 = int(w * 0.70)
    cy0 = int(h * 0.30)
    cy1 = int(h * 0.70)
    center = arr[cy0:cy1, cx0:cx1]
    if center.size == 0:
        return None
    r, g, b = center[..., 0], center[..., 1], center[..., 2]
    # Red-dominant: R clearly ahead of G and B, and above a brightness floor.
    red_mask = (r >= 140) & (r >= g * 1.3) & (r >= b * 1.3)
    red_ratio = float(red_mask.sum()) / max(center.shape[0] * center.shape[1], 1)
    if red_ratio >= 0.12:
        return _RED_FIVE_MAP[classified_tile]
    return None


def _apply_red_five(
    crop: Image.Image,
    match: TileTemplateMatch,
) -> TileTemplateMatch:
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
