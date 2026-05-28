from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from .tile_templates import DEFAULT_MAX_DISTANCE, TileTemplateMatch, classify_tile_from_templates
from .vit_tile_classifier_onnx import classify_tile_crops_onnx, onnx_tile_classifier_available


_RED_FIVE_MAP = {"5m": "0m", "5p": "0p", "5s": "0s", "R5m": "0m", "R5p": "0p", "R5s": "0s"}
_ONNX_PROBED: dict[str, bool] = {}


def onnx_discard_available(*, model_dir: str | os.PathLike[str] | None = None) -> bool:
    key = str(model_dir or "")
    if key not in _ONNX_PROBED:
        _ONNX_PROBED[key] = onnx_tile_classifier_available(model_dir=model_dir)
    return _ONNX_PROBED[key]


def classify_discard_tiles_batch(
    crops: list[Image.Image],
    template_payload: dict[str, Any] | None = None,
    *,
    model_dir: str | os.PathLike[str] | None = None,
) -> list[TileTemplateMatch | None]:
    if not crops:
        return []
    if onnx_discard_available(model_dir=model_dir):
        predictions = classify_tile_crops_onnx(crops, model_dir=model_dir, top_k=2)
        if predictions:
            return [_onnx_prediction_to_match(prediction) for prediction in predictions]
    payload = template_payload or {}
    return [classify_tile_from_templates(crop, payload) for crop in crops]


def classify_hand_tile(
    crop: Image.Image,
    template_payload: dict[str, Any],
    *,
    use_onnx: bool | None = None,
    fallback_to_onnx: bool = False,
) -> TileTemplateMatch | None:
    """Classify a hand tile using templates by default, ONNX only when enabled.

    The current bundled ONNX model is reliable for discard crops, but hand
    tiles still need a fine-tuned model. Keep live hand recognition template
    first until the retrained model is explicitly enabled.
    """
    if (onnx_hand_enabled() if use_onnx is None else bool(use_onnx)) and onnx_discard_available():
        onnx_match = _classify_hand_onnx(crop)
        if onnx_match is not None and onnx_match.confidence >= 0.5:
            return onnx_match
        # Low confidence from ONNX — try template as second opinion
        tmpl_result = classify_tile_from_templates(crop, template_payload)
        if tmpl_result is not None and tmpl_result.confidence >= 0.3:
            # If template agrees with ONNX, trust the higher-confidence one
            if tmpl_result.tile == (onnx_match.tile if onnx_match else ""):
                return tmpl_result if tmpl_result.confidence > onnx_match.confidence else onnx_match  # type: ignore[union-attr]
            # Disagreement — trust template if confident enough
            if tmpl_result.confidence >= 0.5:
                return _apply_red_five(crop, tmpl_result)
            return onnx_match
        return onnx_match

    result = classify_tile_from_templates(crop, template_payload)
    if fallback_to_onnx and onnx_discard_available() and (result is None or result.confidence < 0.3):
        onnx_match = _classify_hand_onnx(crop)
        if onnx_match is not None and onnx_match.confidence >= 0.75:
            return onnx_match
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


def _onnx_prediction_to_match(prediction: Any) -> TileTemplateMatch | None:
    if prediction is None:
        return None
    confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
    runner_up_tile = ""
    runner_up_distance = None
    top_k = getattr(prediction, "top_k", []) or []
    if len(top_k) > 1:
        runner_up_tile = str(top_k[1].get("tile") or "")
        runner_up_distance = (1.0 - float(top_k[1].get("score", 0.0) or 0.0)) * DEFAULT_MAX_DISTANCE
    return TileTemplateMatch(
        tile=str(getattr(prediction, "tile", "") or ""),
        confidence=confidence,
        distance=(1.0 - confidence) * DEFAULT_MAX_DISTANCE,
        runner_up_tile=runner_up_tile,
        runner_up_distance=runner_up_distance,
    )


def _classify_hand_onnx(crop: Image.Image) -> TileTemplateMatch | None:
    predictions = classify_tile_crops_onnx([crop], top_k=2)
    if not predictions or predictions[0] is None:
        return None
    match = _onnx_prediction_to_match(predictions[0])
    if match is None:
        return None
    # Apply red-five color post-processing
    red = detect_red_five(crop, match.tile)
    if red is not None:
        return TileTemplateMatch(
            tile=red,
            confidence=match.confidence,
            distance=match.distance,
            runner_up_tile=match.runner_up_tile,
            runner_up_distance=match.runner_up_distance,
        )
    return match
