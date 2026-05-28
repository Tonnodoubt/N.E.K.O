from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .roi import RoiBox, collect_region_metrics

CALL_BUTTONS = {"chi", "pon", "kan"}
WIN_BUTTONS = {"ron", "tsumo"}
BUTTON_ORDER = ("ron", "tsumo", "riichi", "kan", "pon", "chi", "skip")


def detect_action_buttons_fast(image_path: Path) -> tuple[list[str], dict[str, Any]]:
    """Very light action-window hinting.

    This is intentionally conservative and exists as a fast interrupt layer.
    Call sites can also pass observed buttons from another detector.
    """
    if not image_path.exists():
        return [], {"error": 1.0}
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        box = RoiBox(
            "bottom_action_bar",
            left=int(image.width * 0.24),
            top=int(image.height * 0.61),
            width=int(image.width * 0.52),
            height=int(image.height * 0.17),
        )
        metrics = collect_region_metrics(image, box, sample_step=5)
        template_buttons, template_meta = _detect_template_buttons(image)
    buttons, filter_meta = _filter_plausible_buttons(template_buttons)
    return buttons, {"metrics": metrics, "templates": template_meta, "button_filter": filter_meta}


def _detect_template_buttons(image: Image.Image) -> tuple[list[str], dict[str, Any]]:
    template_root = Path(__file__).resolve().parent / "templates"
    meta_path = template_root / "meta.json"
    if not meta_path.exists():
        return [], {"available": False}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], {"available": False, "error": "meta_unreadable"}
    templates = payload.get("templates")
    if not isinstance(templates, dict):
        return [], {"available": False, "error": "templates_missing"}

    search_box = (
        int(image.width * 0.18),
        int(image.height * 0.54),
        int(image.width * 0.86),
        int(image.height * 0.82),
    )
    search = image.crop(search_box).convert("RGB")
    matches: list[dict[str, Any]] = []
    for item in templates.values():
        if not isinstance(item, dict):
            continue
        button_type = str(item.get("button_type") or "").strip()
        rel_file = str(item.get("file") or "").strip()
        resolution = item.get("resolution") if isinstance(item.get("resolution"), list) else []
        if not button_type or not rel_file or len(resolution) != 2:
            continue
        template_path = template_root / rel_file
        if not template_path.exists():
            continue
        with Image.open(template_path) as opened:
            template = opened.convert("RGB")
        scale = min(image.width / max(1, int(resolution[0])), image.height / max(1, int(resolution[1])))
        if abs(scale - 1.0) > 0.05:
            template = template.resize(
                (max(8, int(template.width * scale)), max(8, int(template.height * scale))),
                Image.Resampling.BILINEAR,
            )
        score = _coarse_template_score(search, template)
        threshold = max(0.86, min(0.99, float(item.get("match_threshold") or payload.get("default_match_threshold") or 0.9)))
        matches.append(
            {
                "button_type": button_type,
                "score": round(score, 4),
                "threshold": round(threshold, 4),
                "accepted": score >= threshold,
            }
        )

    detected = [item["button_type"] for item in matches if item["score"] >= item["threshold"]]
    return detected, {"available": True, "matches": sorted(matches, key=lambda item: -item["score"])[:8]}


def _coarse_template_score(search: Image.Image, template: Image.Image) -> float:
    max_search_width = 360
    downscale = min(1.0, max_search_width / max(1, search.width))
    search_small = search.resize(
        (max(1, int(search.width * downscale)), max(1, int(search.height * downscale))),
        Image.Resampling.BILINEAR,
    )
    template_small = template.resize(
        (max(4, int(template.width * downscale)), max(4, int(template.height * downscale))),
        Image.Resampling.BILINEAR,
    )
    if template_small.width > search_small.width or template_small.height > search_small.height:
        return 0.0

    search_arr = np.asarray(search_small.convert("RGB"), dtype=np.float32)
    template_arr = np.asarray(template_small.convert("RGB"), dtype=np.float32)
    mask = _template_content_mask(template_arr)
    template_masked = template_arr[mask].reshape(-1, 3)
    template_centered = template_masked - template_masked.mean(axis=0, keepdims=True)
    template_norm = float(np.sqrt(np.sum(template_centered * template_centered)))
    if template_norm <= 1.0:
        return 0.0

    stride = max(2, int(min(template_small.width, template_small.height) / 18))
    best = 0.0
    max_y = search_arr.shape[0] - template_arr.shape[0]
    max_x = search_arr.shape[1] - template_arr.shape[1]
    for y in range(0, max_y + 1, stride):
        for x in range(0, max_x + 1, stride):
            window = search_arr[y : y + template_arr.shape[0], x : x + template_arr.shape[1]]
            window_masked = window[mask].reshape(-1, 3)
            window_centered = window_masked - window_masked.mean(axis=0, keepdims=True)
            window_norm = float(np.sqrt(np.sum(window_centered * window_centered)))
            if window_norm <= 1.0:
                continue
            correlation = float(np.sum(window_centered * template_centered) / (window_norm * template_norm))
            diff = float(np.mean(np.abs(window_masked - template_masked)))
            diff_score = 1.0 - diff / 255.0
            score = max(0.0, correlation) * 0.7 + diff_score * 0.3
            if score > best:
                best = score
    return best


def _template_content_mask(template_arr: np.ndarray) -> np.ndarray:
    gray = template_arr.mean(axis=2)
    median_color = np.median(template_arr.reshape(-1, 3), axis=0)
    color_distance = np.linalg.norm(template_arr - median_color, axis=2)
    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    mask = (color_distance > 20.0) | (np.maximum(grad_x, grad_y) > 12.0)
    if float(mask.mean()) < 0.08:
        return np.ones(gray.shape, dtype=bool)
    return mask


def _filter_plausible_buttons(buttons: list[str]) -> tuple[list[str], dict[str, Any]]:
    unique = {str(button).strip() for button in buttons if str(button).strip()}
    conflicts: list[str] = []
    if "ron" in unique and "tsumo" in unique:
        conflicts.append("ron_with_tsumo")
    if "tsumo" in unique and any(button in unique for button in CALL_BUTTONS):
        conflicts.append("tsumo_with_call")
    if "riichi" in unique and any(button in unique for button in CALL_BUTTONS):
        conflicts.append("riichi_with_call")
    if len([button for button in unique if button != "skip"]) > 4:
        conflicts.append("too_many_action_buttons")
    if conflicts:
        return [], {"input_buttons": sorted(unique), "rejected": True, "reasons": conflicts}
    ordered = [button for button in BUTTON_ORDER if button in unique]
    return ordered, {"input_buttons": sorted(unique), "rejected": False, "reasons": []}
