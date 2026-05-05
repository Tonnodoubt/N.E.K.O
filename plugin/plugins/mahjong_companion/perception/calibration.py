from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from ..storage import load_json_payload, locked_json_path, write_json_atomic


logger = logging.getLogger(__name__)

CALIBRATION_LABEL_SCHEMA = "mahjong-calibration-label-v1"
CALIBRATION_PROFILE_VERSION = "v0.3-calibration"

_resolution_cache: dict[str, tuple[float, CalibrationProfile]] = {}


@dataclass
class CalibrationOffsets:
    x_px: int = 0
    y_px: int = 0
    width_px: int = 0
    height_px: int = 0
    gap_px: int = 0
    draw_gap_px: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationProfile:
    profile_id: str = "default"
    version: str = "v8-preview"
    source: str = "builtin"
    enabled: bool = False
    screen_width: int = 0
    screen_height: int = 0
    confidence: float = 0.0
    hand_offsets: CalibrationOffsets = field(default_factory=CalibrationOffsets)
    meld_offsets: CalibrationOffsets = field(default_factory=CalibrationOffsets)
    dora_offsets: CalibrationOffsets = field(default_factory=CalibrationOffsets)
    hand_tile_templates: dict[str, Any] = field(default_factory=dict)
    discard_tile_templates: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hand_offsets"] = self.hand_offsets.to_dict()
        payload["meld_offsets"] = self.meld_offsets.to_dict()
        payload["dora_offsets"] = self.dora_offsets.to_dict()
        return payload


def build_default_calibration_profile(width: int, height: int) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id=f"default-{width}x{height}",
        version="v8-preview",
        source="builtin",
        enabled=False,
        screen_width=max(0, int(width)),
        screen_height=max(0, int(height)),
        confidence=0.18,
        notes=[
            "using builtin fallback calibration",
            "tile-level parsing should stay in degraded mode until a tuned profile is provided",
        ],
    )


def load_calibration_profile(path: Path) -> CalibrationProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration profile payload is not a JSON object")
    return CalibrationProfile(
        profile_id=str(payload.get("profile_id", path.stem)).strip() or path.stem,
        version=str(payload.get("version", "v8-preview")).strip() or "v8-preview",
        source=str(payload.get("source", str(path))).strip() or str(path),
        enabled=bool(payload.get("enabled", True)),
        screen_width=int(payload.get("screen_width", 0) or 0),
        screen_height=int(payload.get("screen_height", 0) or 0),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        hand_offsets=_load_offsets(payload.get("hand_offsets")),
        meld_offsets=_load_offsets(payload.get("meld_offsets")),
        dora_offsets=_load_offsets(payload.get("dora_offsets")),
        hand_tile_templates=_load_template_payload(payload.get("hand_tile_templates")),
        discard_tile_templates=_load_template_payload(payload.get("discard_tile_templates")),
        notes=_load_notes(payload.get("notes")),
    )


def save_calibration_profile(profile: CalibrationProfile, path: Path) -> None:
    with locked_json_path(path):
        write_json_atomic(path, profile.to_dict())


def resolve_calibration_profile(
    width: int,
    height: int,
    *,
    calibration_dir: Path | None = None,
) -> CalibrationProfile:
    cache_key = f"{width}x{height}:{calibration_dir}"

    if calibration_dir is not None and calibration_dir.exists():
        try:
            dir_mtime = calibration_dir.stat().st_mtime
        except OSError:
            dir_mtime = 0.0
        cached = _resolution_cache.get(cache_key)
        if cached is not None and cached[0] >= dir_mtime:
            return cached[1]

        resolved = _resolve_profile_from_dir(width, height, calibration_dir)
        if resolved is not None:
            _resolution_cache[cache_key] = (dir_mtime, resolved)
            return resolved

    return build_default_calibration_profile(width, height)


def label_sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".label.json")


def write_calibration_label(path: Path, payload: dict[str, Any]) -> None:
    with locked_json_path(path):
        write_json_atomic(path, payload)


def iter_calibration_label_paths(root: Path) -> list[Path]:
    if root.is_file() and root.name.endswith(".label.json"):
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.label.json") if path.is_file())


def train_calibration_profile(
    label_root: Path,
    *,
    client_version: str = "unknown",
    min_samples: int = 20,
    profile_id: str | None = None,
) -> CalibrationProfile:
    label_paths = iter_calibration_label_paths(label_root)
    return train_calibration_profile_from_paths(
        label_paths,
        label_root=label_root,
        client_version=client_version,
        min_samples=min_samples,
        profile_id=profile_id,
    )


def train_calibration_profile_from_paths(
    label_paths: list[Path],
    *,
    label_root: Path | None = None,
    client_version: str = "unknown",
    min_samples: int = 20,
    profile_id: str | None = None,
) -> CalibrationProfile:
    label_root = label_root or _common_label_root(label_paths)
    if not label_paths:
        raise ValueError(f"no calibration labels found under {label_root}")

    records = [_load_calibration_label(path) for path in sorted(label_paths)]
    records = [record for record in records if record is not None]
    if not records:
        raise ValueError(f"no usable calibration labels found under {label_root}")

    resolutions = {(record["width"], record["height"]) for record in records}
    if len(resolutions) != 1:
        formatted = ", ".join(f"{width}x{height}" for width, height in sorted(resolutions))
        raise ValueError(f"calibration labels must use one resolution per profile, got: {formatted}")
    width, height = next(iter(resolutions))

    x_offsets: list[int] = []
    y_offsets: list[int] = []
    width_offsets: list[int] = []
    height_offsets: list[int] = []
    gap_offsets: list[int] = []
    draw_gap_offsets: list[int] = []
    annotated_slots = 0

    from .hand_layout import build_hand_layout

    default_layout = build_hand_layout(width, height, calibration=build_default_calibration_profile(width, height))
    default_slots = {slot.slot_id: slot.box for slot in default_layout["hand"]}

    for record in records:
        hand_offsets = record.get("hand_offsets")
        hand_slots = record["hand_slots"]
        annotated_slots += len(hand_slots)
        if record.get("has_hand_offsets") and isinstance(hand_offsets, CalibrationOffsets):
            x_offsets.append(hand_offsets.x_px)
            y_offsets.append(hand_offsets.y_px)
            width_offsets.append(hand_offsets.width_px)
            height_offsets.append(hand_offsets.height_px)
            gap_offsets.append(hand_offsets.gap_px)
            draw_gap_offsets.append(hand_offsets.draw_gap_px)
            continue

        observed_left_by_index: dict[int, int] = {}
        for slot in hand_slots:
            slot_id = slot.get("slot_id")
            box = slot.get("box")
            if not isinstance(slot_id, str) or not isinstance(box, dict):
                continue
            default_box = default_slots.get(slot_id)
            if default_box is None:
                continue
            index = _hand_slot_index(slot_id)
            if index is not None:
                observed_left_by_index[index] = int(box.get("left", 0) or 0)
            if slot_id == "hand_1":
                x_offsets.append(int(box.get("left", 0) or 0) - default_box.left)
                y_offsets.append(int(box.get("top", 0) or 0) - default_box.top)
            width_offsets.append(int(box.get("width", 0) or 0) - default_box.width)
            height_offsets.append(int(box.get("height", 0) or 0) - default_box.height)
        if 1 in observed_left_by_index and 2 in observed_left_by_index:
            default_step = default_slots["hand_2"].left - default_slots["hand_1"].left
            observed_step = observed_left_by_index[2] - observed_left_by_index[1]
            gap_offsets.append(observed_step - default_step)
        if 13 in observed_left_by_index and 14 in observed_left_by_index:
            default_step = default_slots["hand_14"].left - default_slots["hand_13"].left
            observed_step = observed_left_by_index[14] - observed_left_by_index[13]
            draw_gap_offsets.append(observed_step - default_step - _median_int(gap_offsets))

    sample_count = len(records)
    safe_client_version = _safe_slug(client_version or "unknown")
    profile_id = profile_id or f"majsoul-pc-{safe_client_version}-{width}x{height}"
    min_samples = max(1, int(min_samples))
    enabled = sample_count >= min_samples
    confidence = min(
        0.95,
        0.35
        + min(sample_count, min_samples) / min_samples * 0.45
        + min(annotated_slots, min_samples * 14) / (min_samples * 14) * 0.15,
    )
    hand_tile_templates = _build_hand_tile_templates(records)
    discard_tile_templates = _build_discard_tile_templates(records)
    notes = [
        f"trained_from_labels={sample_count}",
        f"annotated_hand_slots={annotated_slots}",
        f"min_samples={min_samples}",
        "raw calibration frames remain local and are not tracked by git",
    ]
    if hand_tile_templates:
        notes.append(f"hand_tile_template_samples={hand_tile_templates.get('stored_sample_count', 0)}")
    if discard_tile_templates:
        notes.append(f"discard_tile_template_samples={discard_tile_templates.get('stored_sample_count', 0)}")

    return CalibrationProfile(
        profile_id=profile_id,
        version=CALIBRATION_PROFILE_VERSION,
        source=str(label_root),
        enabled=enabled,
        screen_width=width,
        screen_height=height,
        confidence=round(confidence, 3),
        hand_offsets=CalibrationOffsets(
            x_px=_median_int(x_offsets),
            y_px=_median_int(y_offsets),
            width_px=_median_int(width_offsets),
            height_px=_median_int(height_offsets),
            gap_px=_median_int(gap_offsets),
            draw_gap_px=_median_int(draw_gap_offsets),
        ),
        hand_tile_templates=hand_tile_templates,
        discard_tile_templates=discard_tile_templates,
        notes=notes,
    )


def _load_offsets(value: Any) -> CalibrationOffsets:
    if not isinstance(value, dict):
        return CalibrationOffsets()
    return CalibrationOffsets(
        x_px=int(value.get("x_px", 0) or 0),
        y_px=int(value.get("y_px", 0) or 0),
        width_px=int(value.get("width_px", 0) or 0),
        height_px=int(value.get("height_px", 0) or 0),
        gap_px=int(value.get("gap_px", 0) or 0),
        draw_gap_px=int(value.get("draw_gap_px", 0) or 0),
    )


def _load_notes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _load_template_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_profile_from_dir(width: int, height: int, calibration_dir: Path) -> CalibrationProfile | None:
    profiles: list[CalibrationProfile] = []
    for candidate in _profile_candidates(width, height, calibration_dir):
        try:
            profile = load_calibration_profile(candidate)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("failed to load calibration profile %s: %s", candidate, exc)
            continue
        if profile.screen_width not in {0, width} or profile.screen_height not in {0, height}:
            continue
        profiles.append(profile)
    if not profiles:
        return _resolve_scaled_profile_from_dir(width, height, calibration_dir)
    profiles.sort(key=lambda item: (item.enabled, item.confidence, item.profile_id), reverse=True)
    return _merge_specialized_profile_templates(profiles[0], profiles)


def _merge_specialized_profile_templates(
    primary: CalibrationProfile,
    profiles: list[CalibrationProfile],
) -> CalibrationProfile:
    hand_profile = _best_profile_for_template_domain(profiles, domain="hand")
    discard_profile = _best_profile_for_template_domain(profiles, domain="discard")
    if hand_profile is primary and discard_profile is primary:
        return primary

    merged = copy.deepcopy(primary)
    merged_notes = list(merged.notes)
    if hand_profile is not None and hand_profile is not primary:
        merged.hand_tile_templates = copy.deepcopy(hand_profile.hand_tile_templates)
        merged_notes.append(f"merged_hand_templates_from_profile={hand_profile.profile_id}")
    if discard_profile is not None and discard_profile is not primary:
        merged.discard_tile_templates = copy.deepcopy(discard_profile.discard_tile_templates)
        merged_notes.append(f"merged_discard_templates_from_profile={discard_profile.profile_id}")
    if merged_notes != merged.notes:
        merged.profile_id = f"{primary.profile_id}-merged"
        merged.source = f"{primary.source}; specialized_template_merge"
        merged.notes = merged_notes
    return merged


def _best_profile_for_template_domain(
    profiles: list[CalibrationProfile],
    *,
    domain: str,
) -> CalibrationProfile | None:
    candidates = [
        profile
        for profile in profiles
        if _template_payload_for_domain(profile, domain)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda profile: _template_domain_score(profile, domain=domain))


def _template_payload_for_domain(profile: CalibrationProfile, domain: str) -> dict[str, Any]:
    if domain == "discard":
        return profile.discard_tile_templates
    return profile.hand_tile_templates


def _template_domain_score(profile: CalibrationProfile, *, domain: str) -> tuple[Any, ...]:
    payload = _template_payload_for_domain(profile, domain)
    text = _profile_template_text(profile, payload)
    source_is_vit = "vit_labeled" in text or "vit_template_training" in text
    discard_specialized = "vit-discard" in text or "discard_only" in text
    stored_samples = _coerce_template_int(payload.get("stored_sample_count"), default=0)
    source_samples = _coerce_template_int(payload.get("source_sample_count"), default=0)
    if domain == "discard":
        return (
            profile.enabled,
            source_is_vit or discard_specialized,
            stored_samples,
            source_samples,
            profile.confidence,
            profile.profile_id,
        )
    return (
        profile.enabled,
        not discard_specialized,
        not source_is_vit,
        profile.confidence,
        stored_samples,
        profile.profile_id,
    )


def _profile_template_text(profile: CalibrationProfile, payload: dict[str, Any]) -> str:
    return " ".join(
        [
            profile.profile_id,
            profile.source,
            str(payload.get("source", "")),
            " ".join(profile.notes),
        ],
    ).lower()


def _coerce_template_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _profile_candidates(width: int, height: int, calibration_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    search_dirs = [calibration_dir]
    profiles_dir = calibration_dir / "profiles"
    if profiles_dir != calibration_dir:
        search_dirs.append(profiles_dir)

    for directory in search_dirs:
        candidates.append(directory / f"{width}x{height}.json")
        candidates.extend(sorted(directory.glob(f"*-{width}x{height}.json")))
        candidates.append(directory / "default.json")

    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        existing.append(candidate)
    return existing


def _resolve_scaled_profile_from_dir(width: int, height: int, calibration_dir: Path) -> CalibrationProfile | None:
    candidates: list[tuple[float, int, CalibrationProfile]] = []
    for path in _all_profile_paths(calibration_dir):
        try:
            profile = load_calibration_profile(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("failed to load calibration profile %s: %s", path, exc)
            continue
        if not _can_scale_profile(profile, width=width, height=height):
            continue
        aspect_delta = abs((profile.screen_width / profile.screen_height) - (width / height))
        area_delta = abs((profile.screen_width * profile.screen_height) - (width * height))
        candidates.append((aspect_delta, area_delta, profile))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], not item[2].enabled, -item[2].confidence, item[2].profile_id))
    return _scale_profile(candidates[0][2], width=width, height=height)


def _all_profile_paths(calibration_dir: Path) -> list[Path]:
    search_dirs = [calibration_dir]
    profiles_dir = calibration_dir / "profiles"
    if profiles_dir != calibration_dir:
        search_dirs.append(profiles_dir)
    seen: set[Path] = set()
    paths: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen or path.name.startswith("."):
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def _can_scale_profile(profile: CalibrationProfile, *, width: int, height: int) -> bool:
    if not profile.enabled:
        return False
    if profile.screen_width <= 0 or profile.screen_height <= 0 or width <= 0 or height <= 0:
        return False
    if profile.screen_width == width and profile.screen_height == height:
        return False
    aspect_delta = abs((profile.screen_width / profile.screen_height) - (width / height))
    if aspect_delta > 0.02:
        return False
    return bool(profile.hand_tile_templates or profile.discard_tile_templates)


def _scale_profile(profile: CalibrationProfile, *, width: int, height: int) -> CalibrationProfile:
    scale_x = width / profile.screen_width
    scale_y = height / profile.screen_height
    notes = list(profile.notes)
    notes.append(f"scaled_from_profile={profile.profile_id}")
    notes.append(f"scaled_from_resolution={profile.screen_width}x{profile.screen_height}")
    return CalibrationProfile(
        profile_id=f"{profile.profile_id}-scaled-{width}x{height}",
        version=profile.version,
        source=f"{profile.source} (scaled to {width}x{height})",
        enabled=profile.enabled,
        screen_width=width,
        screen_height=height,
        confidence=round(max(0.0, min(0.95, profile.confidence * 0.82)), 3),
        hand_offsets=_scale_offsets(profile.hand_offsets, scale_x=scale_x, scale_y=scale_y),
        meld_offsets=_scale_offsets(profile.meld_offsets, scale_x=scale_x, scale_y=scale_y),
        dora_offsets=_scale_offsets(profile.dora_offsets, scale_x=scale_x, scale_y=scale_y),
        hand_tile_templates=copy.deepcopy(profile.hand_tile_templates),
        discard_tile_templates=copy.deepcopy(profile.discard_tile_templates),
        notes=notes,
    )


def _scale_offsets(offsets: CalibrationOffsets, *, scale_x: float, scale_y: float) -> CalibrationOffsets:
    return CalibrationOffsets(
        x_px=round(offsets.x_px * scale_x),
        y_px=round(offsets.y_px * scale_y),
        width_px=round(offsets.width_px * scale_x),
        height_px=round(offsets.height_px * scale_y),
        gap_px=round(offsets.gap_px * scale_x),
        draw_gap_px=round(offsets.draw_gap_px * scale_x),
    )


def _load_calibration_label(path: Path) -> dict[str, Any] | None:
    payload = load_json_payload(path, default={}, expected_type=dict, logger=logger)
    if not payload:
        return None

    image_path = _resolve_label_image_path(path, payload)
    width, height = _coerce_label_size(payload)
    if width <= 0 or height <= 0:
        width, height = _image_size(image_path)
    if width <= 0 or height <= 0:
        return None

    layout = payload.get("layout")
    if not isinstance(layout, dict):
        layout = {}
    hand_slots = layout.get("hand_slots")
    if not isinstance(hand_slots, list):
        hand_slots = payload.get("hand_slots")
    if not isinstance(hand_slots, list):
        hand_slots = []

    raw_hand_offsets = layout.get("hand_offsets") or payload.get("hand_offsets")
    hand_offsets = _load_offsets(raw_hand_offsets)
    return {
        "path": path,
        "image_path": image_path,
        "width": width,
        "height": height,
        "hand_slots": [slot for slot in hand_slots if isinstance(slot, dict)],
        "discard_items": _discard_items_from_label(payload),
        "hand_offsets": hand_offsets,
        "has_hand_offsets": isinstance(raw_hand_offsets, dict),
    }


def _build_hand_tile_templates(records: list[dict[str, Any]]) -> dict[str, Any]:
    from PIL import Image

    from .tile_templates import build_hand_tile_template_payload

    samples: list[tuple[str, Image.Image]] = []
    for record in records:
        image_path = record.get("image_path")
        if not isinstance(image_path, Path) or not image_path.exists():
            continue
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                for slot in record.get("hand_slots", []):
                    sample = _template_sample_from_slot(image, slot)
                    if sample is not None:
                        samples.append(sample)
        except OSError as exc:
            logger.warning("failed to load calibration frame %s: %s", image_path, exc)
            continue
    return build_hand_tile_template_payload(samples)


def _build_discard_tile_templates(records: list[dict[str, Any]]) -> dict[str, Any]:
    from PIL import Image

    from .tile_templates import build_hand_tile_template_payload

    samples: list[tuple[str, Image.Image]] = []
    for record in records:
        image_path = record.get("image_path")
        if not isinstance(image_path, Path) or not image_path.exists():
            continue
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                for item in record.get("discard_items", []):
                    sample = _template_sample_from_discard_item(image, item)
                    if sample is not None:
                        samples.append(sample)
        except OSError as exc:
            logger.warning("failed to load calibration frame %s: %s", image_path, exc)
            continue
    return build_hand_tile_template_payload(samples)


def _template_sample_from_slot(image: Any, slot: Any) -> tuple[str, Any] | None:
    if not isinstance(slot, dict):
        return None
    tile = str(slot.get("tile", "")).strip()
    box = slot.get("box")
    if not tile or not isinstance(box, dict):
        return None
    left = int(box.get("left", 0) or 0)
    top = int(box.get("top", 0) or 0)
    width = int(box.get("width", 0) or 0)
    height = int(box.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return None
    crop = image.crop((left, top, left + width, top + height))
    return (tile, crop.copy())


def _template_sample_from_discard_item(image: Any, item: Any) -> tuple[str, Any] | None:
    from .discard_parser import crop_discard_quad, normalize_discard_crop

    if not isinstance(item, dict):
        return None
    tile = str(item.get("tile", "")).strip()
    bbox = _bbox_from_payload(item)
    if not tile or bbox is None:
        return None
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return None
    orientation = str(item.get("orientation", "")).strip()
    quad = _quad_from_payload(item)
    if quad is not None:
        crop = crop_discard_quad(
            image,
            quad,
            output_size=(right - left, bottom - top),
            orientation=orientation,
        )
    else:
        crop = image.crop((left, top, right, bottom))
        crop = normalize_discard_crop(crop, orientation)
    return (tile, crop.copy())


def _discard_items_from_label(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_piles = payload.get("discard_piles")
    if not isinstance(raw_piles, dict):
        layout = payload.get("layout")
        raw_piles = layout.get("discard_piles") if isinstance(layout, dict) else None
    if not isinstance(raw_piles, dict):
        return []

    items: list[dict[str, Any]] = []
    for player, raw_items in raw_piles.items():
        player_key = str(player).strip()
        if not player_key or not isinstance(raw_items, list):
            continue
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            tile = str(item.get("tile", "")).strip()
            bbox = _bbox_from_payload(item)
            if not tile or bbox is None:
                continue
            items.append(
                _with_optional_quad(
                    {
                        "tile": tile,
                        "player": str(item.get("player") or player_key),
                        "turn_index": int(item.get("turn_index", index + 1) or index + 1),
                        "bbox": list(bbox),
                        "orientation": str(item.get("orientation", "")).strip()
                        or _orientation_from_player(player_key),
                    },
                    item,
                )
            )
    return items


def _bbox_from_payload(payload: dict[str, Any]) -> tuple[int, int, int, int] | None:
    bbox = payload.get("bbox")
    if isinstance(bbox, list | tuple) and len(bbox) == 4:
        try:
            left, top, right, bottom = [int(value) for value in bbox]
        except (TypeError, ValueError):
            return None
        return left, top, right, bottom

    box = payload.get("box")
    if isinstance(box, dict):
        try:
            left = int(box.get("left", 0) or 0)
            top = int(box.get("top", 0) or 0)
            width = int(box.get("width", 0) or 0)
            height = int(box.get("height", 0) or 0)
        except (TypeError, ValueError):
            return None
        return left, top, left + width, top + height
    return None


def _quad_from_payload(
    payload: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    quad = payload.get("quad")
    if not isinstance(quad, list | tuple) or len(quad) != 4:
        return None
    points: list[tuple[int, int]] = []
    for point in quad:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        try:
            x, y = [int(value) for value in point]
        except (TypeError, ValueError):
            return None
        points.append((x, y))
    return (points[0], points[1], points[2], points[3])


def _with_optional_quad(payload: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    quad = _quad_from_payload(source)
    if quad is not None:
        payload["quad"] = [[x, y] for x, y in quad]
    return payload


def _orientation_from_player(player: str) -> str:
    if player == "self":
        return "bottom"
    if player == "top_opponent":
        return "top"
    if player == "left_opponent":
        return "left"
    if player == "right_opponent":
        return "right"
    return player


def _resolve_label_image_path(label_path: Path, payload: dict[str, Any]) -> Path:
    image = payload.get("image")
    if isinstance(image, dict):
        raw_path = image.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path)
            if candidate.is_absolute():
                return candidate
            relative = label_path.parent / candidate
            if relative.exists():
                return relative
    stem = label_path.name.removesuffix(".label.json")
    for suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = label_path.with_name(f"{stem}{suffix}")
        if candidate.exists():
            return candidate
    return label_path.with_name(f"{stem}.png")


def _coerce_label_size(payload: dict[str, Any]) -> tuple[int, int]:
    image = payload.get("image")
    if isinstance(image, dict):
        width = int(image.get("width", 0) or image.get("screen_width", 0) or 0)
        height = int(image.get("height", 0) or image.get("screen_height", 0) or 0)
        if width > 0 and height > 0:
            return width, height
        resolution = str(image.get("resolution", "")).strip()
        parsed = _parse_resolution(resolution)
        if parsed is not None:
            return parsed

    image_size = payload.get("image_size")
    if isinstance(image_size, dict):
        width = int(image_size.get("width", 0) or 0)
        height = int(image_size.get("height", 0) or 0)
        if width > 0 and height > 0:
            return width, height

    width = int(payload.get("screen_width", 0) or payload.get("width", 0) or 0)
    height = int(payload.get("screen_height", 0) or payload.get("height", 0) or 0)
    if width > 0 and height > 0:
        return width, height

    return (0, 0)


def _image_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    try:
        from PIL import Image

        with Image.open(path) as opened:
            return opened.size
    except OSError:
        return (0, 0)


def _parse_resolution(value: str) -> tuple[int, int] | None:
    normalized = value.lower().replace("*", "x").replace("×", "x")
    if "x" not in normalized:
        return None
    left, right = normalized.split("x", 1)
    try:
        width = int(left.strip())
        height = int(right.strip())
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(round(median(values)))


def _hand_slot_index(slot_id: str) -> int | None:
    if not slot_id.startswith("hand_"):
        return None
    raw = slot_id.removeprefix("hand_")
    if not raw.isdigit():
        return None
    return int(raw)


def _safe_slug(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("-")
    return "".join(cleaned).strip("-") or "unknown"


def _common_label_root(label_paths: list[Path]) -> Path:
    if not label_paths:
        return Path(".")
    resolved = [path.parent.resolve() for path in label_paths]
    common = Path(*Path(resolved[0]).parts[:1])
    try:
        import os

        return Path(os.path.commonpath([str(path) for path in resolved]))
    except (OSError, ValueError):
        return common
