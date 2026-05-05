from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..tile_labels import dedupe as _dedupe
from .calibration import CalibrationProfile


@dataclass(frozen=True)
class ButtonRegion:
    button_type: str
    bbox: tuple[int, int, int, int]
    confidence: float
    template_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


def detect_actions(
    scene: str,
    metrics: dict[str, dict[str, Any]],
) -> tuple[list[str], bool, list[str]]:
    bottom = metrics["bottom_action_bar"]
    center = metrics["center_dialog"]
    hand = metrics["bottom_hand_area"]
    right = metrics["right_replay_panel"]
    buttons: list[str] = []
    notes: list[str] = []
    bottom_bar_visible = _looks_like_bottom_action_bar(bottom)
    dark_dialog = _looks_like_dark_dialog(center)
    hand_strip_visible = _looks_like_player_hand_strip(bottom, hand)

    if scene == "replay":
        notes.append("replay scene suppresses user-turn inference")
        return buttons, False, notes

    if scene not in {"menu", "lobby", "result"} and bottom_bar_visible and hand_strip_visible and _looks_like_false_skip(bottom):
        notes.append("bottom strip looks like hand tiles rather than action buttons")
    elif scene not in {"menu", "lobby", "result"} and bottom_bar_visible:
        if bottom["green_ratio"] >= 0.028:
            buttons.append("chi")
            notes.append("green accent in bottom action bar")
        if bottom["gold_ratio"] >= 0.052:
            buttons.append("riichi")
            notes.append("gold accent in bottom action bar")
        if bottom["red_ratio"] >= 0.02:
            buttons.append("ron")
            notes.append("red accent in bottom action bar")
        if bottom["orange_ratio"] >= 0.03:
            buttons.append("skip")
            notes.append("orange accent in bottom action bar")
        if (
            bottom["dark_ratio"] <= 0.52
            and bottom["colorful_ratio"] >= 0.18
            and "skip" not in buttons
        ):
            buttons.append("confirm")
            notes.append("multi-button action bar likely visible")
    elif scene in {"menu", "lobby", "result"} and bottom_bar_visible:
        notes.append("non-match scene suppresses bottom action bar inference")

    if dark_dialog:
        if "confirm" not in buttons:
            buttons.append("confirm")
        notes.append("dark center dialog likely exposes a primary confirm button")
    elif center["bright_ratio"] >= 0.34 and center["stddev"] <= 62.0:
        if "confirm" not in buttons:
            buttons.append("confirm")
        buttons.append("cancel")
        notes.append("center dialog likely exposes confirm/cancel")

    buttons = _dedupe(buttons)
    is_user_turn = False
    if scene in {"in_match", "dialog"} and buttons:
        is_user_turn = True
    elif scene == "unknown" and buttons and right["dark_ratio"] < 0.35:
        is_user_turn = True
        notes.append("button evidence suggests likely user turn despite unknown scene")
    elif not buttons:
        notes.append("insufficient action evidence")

    return buttons, is_user_turn, notes


def detect_button_regions(
    image: Image.Image,
    metrics: dict[str, dict[str, Any]],
    *,
    profile: CalibrationProfile,
    template_dir: Path,
    search_center: bool = True,
) -> list[ButtonRegion]:
    """Locate visible Mahjong Soul action buttons with template matching."""
    if not template_dir.exists():
        return []

    templates = _load_button_templates(template_dir, profile=profile, image_size=image.size)
    if not templates:
        return []

    search_boxes = _button_search_boxes(metrics, image.size, search_center=search_center)
    if not search_boxes:
        return []

    rgb_image = image.convert("RGB")
    best_by_type: dict[str, ButtonRegion] = {}
    for box in search_boxes:
        left, top, right, bottom = box
        region_gray = np.asarray(rgb_image.crop(box).convert("L"), dtype=np.float64)
        for template in templates:
            scaled_template = _scale_template_for_image(template["image"], template, image.size)
            if scaled_template.width < 4 or scaled_template.height < 4:
                continue
            match = _match_template_from_gray(region_gray, scaled_template)
            if match is None:
                continue
            score, rel_x, rel_y = match
            if score < template["match_threshold"]:
                continue
            bbox = (
                left + rel_x,
                top + rel_y,
                left + rel_x + scaled_template.width,
                top + rel_y + scaled_template.height,
            )
            candidate = ButtonRegion(
                button_type=template["button_type"],
                bbox=bbox,
                confidence=round(float(score), 4),
                template_id=template["template_id"],
            )
            existing = best_by_type.get(candidate.button_type)
            if existing is None or candidate.confidence > existing.confidence:
                best_by_type[candidate.button_type] = candidate

    return sorted(best_by_type.values(), key=lambda item: (-item.confidence, item.button_type))


def _looks_like_bottom_action_bar(bottom: dict[str, Any]) -> bool:
    return (
        bottom["orange_ratio"] >= 0.03
        or bottom["gold_ratio"] >= 0.04
        or bottom["red_ratio"] >= 0.015
        or (
            bottom["dark_ratio"] <= 0.58
            and bottom["colorful_ratio"] >= 0.14
            and (
                bottom["orange_ratio"] >= 0.015
                or bottom["gold_ratio"] >= 0.02
                or bottom["green_ratio"] >= 0.02
            )
        )
    )


def _looks_like_dark_dialog(center: dict[str, Any]) -> bool:
    return (
        center["dark_ratio"] >= 0.55
        and center["stddev"] <= 48.0
        and center["colorful_ratio"] >= 0.16
        and (center["gold_ratio"] >= 0.012 or center["orange_ratio"] >= 0.012)
    )


def _looks_like_player_hand_strip(
    bottom: dict[str, Any],
    hand: dict[str, Any],
) -> bool:
    return (
        hand["bright_ratio"] >= 0.12
        and hand["colorful_ratio"] >= 0.55
        and abs(bottom["dark_ratio"] - hand["dark_ratio"]) <= 0.08
        and abs(bottom["colorful_ratio"] - hand["colorful_ratio"]) <= 0.12
    )


def _looks_like_false_skip(bottom: dict[str, Any]) -> bool:
    return (
        bottom["orange_ratio"] < 0.05
        and bottom["gold_ratio"] < 0.01
        and bottom["red_ratio"] < 0.01
        and bottom["green_ratio"] < 0.01
    )




def _load_button_templates(
    template_dir: Path,
    *,
    profile: CalibrationProfile,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in _load_button_templates_cached(
            str(template_dir.resolve()),
            int(profile.screen_width or 0),
            int(profile.screen_height or 0),
            int(image_size[0]),
            int(image_size[1]),
        )
    ]


@lru_cache(maxsize=16)
def _load_button_templates_cached(
    template_dir_value: str,
    profile_width: int,
    profile_height: int,
    image_width: int,
    image_height: int,
) -> tuple[dict[str, Any], ...]:
    template_dir = Path(template_dir_value)
    meta = _load_template_meta(template_dir / "meta.json")
    entries = meta.get("templates") if isinstance(meta.get("templates"), dict) else {}
    default_threshold = float(meta.get("default_match_threshold", 0.85) or 0.85)
    image_size = (image_width, image_height)
    image_resolution = f"{image_width}x{image_height}"
    profile_resolution = f"{profile_width}x{profile_height}"

    raw_templates: list[dict[str, Any]] = []
    if isinstance(entries, dict) and entries:
        for template_id, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            file_value = str(payload.get("file", "")).strip()
            if not file_value:
                continue
            resolution = payload.get("resolution") or [image_size[0], image_size[1]]
            raw_templates.append({
                "template_id": str(template_id),
                "button_type": str(payload.get("button_type") or template_id).strip(),
                "path": template_dir / file_value,
                "resolution": _template_resolution(resolution, image_size),
                "match_threshold": float(payload.get("match_threshold", default_threshold) or default_threshold),
            })
    else:
        for directory_name in {image_resolution, profile_resolution}:
            directory = template_dir / directory_name
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.png")):
                raw_templates.append({
                    "template_id": path.stem,
                    "button_type": path.stem,
                    "path": path,
                    "resolution": _template_resolution(directory_name, image_size),
                    "match_threshold": default_threshold,
                })

    loaded: list[dict[str, Any]] = []
    for raw in raw_templates:
        path = raw["path"]
        if not isinstance(path, Path) or not path.exists():
            continue
        button_type = str(raw["button_type"]).strip()
        if not button_type:
            continue
        try:
            with Image.open(path) as opened:
                template_image = opened.convert("RGB")
        except OSError:
            continue
        loaded.append({
            **raw,
            "button_type": button_type,
            "image": template_image,
        })
    return tuple(loaded)


def _load_template_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _template_resolution(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str) and "x" in value:
        left, right = value.lower().split("x", 1)
        if left.strip().isdigit() and right.strip().isdigit():
            return int(left), int(right)
    if isinstance(value, list | tuple) and len(value) == 2:
        width, height = value
        try:
            return max(1, int(width)), max(1, int(height))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _scale_template_for_image(
    template: Image.Image,
    template_meta: dict[str, Any],
    image_size: tuple[int, int],
) -> Image.Image:
    template_width, template_height = template_meta.get("resolution", image_size)
    scale_x = image_size[0] / max(1, int(template_width))
    scale_y = image_size[1] / max(1, int(template_height))
    if math.isclose(scale_x, 1.0, abs_tol=0.03) and math.isclose(scale_y, 1.0, abs_tol=0.03):
        return template
    width = max(1, int(round(template.width * scale_x)))
    height = max(1, int(round(template.height * scale_y)))
    return template.resize((width, height), Image.Resampling.LANCZOS)


def _button_search_boxes(
    metrics: dict[str, dict[str, Any]],
    image_size: tuple[int, int],
    *,
    search_center: bool = True,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    bottom = metrics.get("bottom_action_bar")
    hand = metrics.get("bottom_hand_area")
    center = metrics.get("center_dialog")
    if isinstance(bottom, dict) and (
        _looks_like_bottom_action_bar(bottom)
        or _looks_like_table_backed_button_region(bottom, hand)
    ):
        bottom_box = _metric_box(bottom)
        if bottom_box is not None:
            boxes.append(_expand_box(bottom_box, image_size, top_ratio=0.80, x_ratio=0.12, bottom_ratio=0.20))
    if search_center and isinstance(center, dict) and (
        _looks_like_dark_dialog(center)
        or (center.get("bright_ratio", 0.0) >= 0.30 and center.get("stddev", 999.0) <= 70.0)
    ):
        center_box = _metric_box(center)
        if center_box is not None:
            boxes.append(_expand_box(center_box, image_size, top_ratio=0.30, x_ratio=0.12, bottom_ratio=0.30))
    return _dedupe_boxes(boxes)


def _looks_like_table_backed_button_region(
    bottom: dict[str, Any],
    hand: dict[str, Any] | None,
) -> bool:
    if (
        bottom.get("colorful_ratio", 0.0) < 0.70
        or bottom.get("bright_ratio", 1.0) > 0.06
        or bottom.get("dark_ratio", 1.0) > 0.12
        or bottom.get("mean_luma", 999.0) > 90.0
    ):
        return False
    if not isinstance(hand, dict):
        return True
    return abs(float(hand.get("colorful_ratio", 0.0) or 0.0) - float(bottom.get("colorful_ratio", 0.0) or 0.0)) <= 0.25


def _metric_box(metric: dict[str, Any]) -> tuple[int, int, int, int] | None:
    raw = metric.get("box")
    if not isinstance(raw, dict):
        return None
    try:
        left = int(raw.get("left", 0) or 0)
        top = int(raw.get("top", 0) or 0)
        width = int(raw.get("width", 0) or 0)
        height = int(raw.get("height", 0) or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return left, top, left + width, top + height


def _expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    top_ratio: float,
    x_ratio: float,
    bottom_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    return (
        max(0, left - int(width * x_ratio)),
        max(0, top - int(height * top_ratio)),
        min(image_size[0], right + int(width * x_ratio)),
        min(image_size[1], bottom + int(height * bottom_ratio)),
    )


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    ordered: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        ordered.append(box)
    return ordered


def _match_template(
    region: Image.Image,
    template: Image.Image,
) -> tuple[float, int, int] | None:
    if region.width < template.width or region.height < template.height:
        return None

    cv2_match = _match_template_cv2(region, template)
    if cv2_match is not None:
        return cv2_match
    return _match_template_numpy(region, template)


def _match_template_from_gray(
    region_gray: np.ndarray,
    template: Image.Image,
) -> tuple[float, int, int] | None:
    if region_gray.shape[0] < template.height or region_gray.shape[1] < template.width:
        return None

    cv2_match = _match_template_cv2_gray(region_gray, template)
    if cv2_match is not None:
        return cv2_match
    return _match_template_numpy_gray(region_gray, template)


def _match_template_cv2(
    region: Image.Image,
    template: Image.Image,
) -> tuple[float, int, int] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    region_arr = np.asarray(region.convert("L"))
    template_arr = np.asarray(template.convert("L"))
    result = cv2.matchTemplate(region_arr, template_arr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_val), int(max_loc[0]), int(max_loc[1])


def _match_template_cv2_gray(
    region_gray: np.ndarray,
    template: Image.Image,
) -> tuple[float, int, int] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    region_arr = region_gray.astype(np.uint8) if region_gray.dtype != np.uint8 else region_gray
    template_arr = np.asarray(template.convert("L"))
    result = cv2.matchTemplate(region_arr, template_arr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_val), int(max_loc[0]), int(max_loc[1])


def _match_template_numpy(
    region: Image.Image,
    template: Image.Image,
) -> tuple[float, int, int] | None:
    region_gray = np.asarray(region.convert("L"), dtype=np.float64)
    template_gray = np.asarray(template.convert("L"), dtype=np.float64)
    if region_gray.shape[0] < template_gray.shape[0] or region_gray.shape[1] < template_gray.shape[1]:
        return None

    scores = _normalized_cross_correlation(region_gray, template_gray)
    if scores.size == 0:
        return None
    flat_index = int(np.nanargmax(scores))
    y, x = np.unravel_index(flat_index, scores.shape)
    score = float(scores[y, x])
    if not math.isfinite(score):
        return None
    return score, int(x), int(y)


def _match_template_numpy_gray(
    region_gray: np.ndarray,
    template: Image.Image,
) -> tuple[float, int, int] | None:
    template_gray = np.asarray(template.convert("L"), dtype=np.float64)
    if region_gray.shape[0] < template_gray.shape[0] or region_gray.shape[1] < template_gray.shape[1]:
        return None

    scores = _normalized_cross_correlation(region_gray, template_gray)
    if scores.size == 0:
        return None
    flat_index = int(np.nanargmax(scores))
    y, x = np.unravel_index(flat_index, scores.shape)
    score = float(scores[y, x])
    if not math.isfinite(score):
        return None
    return score, int(x), int(y)


def _normalized_cross_correlation(
    image: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    template_centered = template - float(template.mean())
    template_norm = math.sqrt(float(np.sum(template_centered * template_centered)))
    if template_norm <= 1e-9:
        return np.empty((0, 0), dtype=np.float64)

    th, tw = template.shape
    ih, iw = image.shape
    shape = (ih + th - 1, iw + tw - 1)
    image_fft = np.fft.rfft2(image, shape)
    template_fft = np.fft.rfft2(np.flipud(np.fliplr(template_centered)), shape)
    cross = np.fft.irfft2(image_fft * template_fft, shape).real
    valid_cross = cross[th - 1:ih, tw - 1:iw]

    sums = _window_sum(image, th, tw)
    sums_sq = _window_sum(image * image, th, tw)
    area = float(th * tw)
    variance = np.maximum(sums_sq - (sums * sums) / area, 0.0)
    denom = np.sqrt(variance) * template_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = valid_cross / denom
    scores[denom <= 1e-9] = -np.inf
    return scores


def _window_sum(image: np.ndarray, height: int, width: int) -> np.ndarray:
    padded = np.pad(image, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )
