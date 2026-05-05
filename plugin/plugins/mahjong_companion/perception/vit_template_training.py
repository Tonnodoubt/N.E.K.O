from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import copy

from PIL import Image

from ..storage import write_json_atomic
from .calibration import CalibrationProfile, _scale_profile, load_calibration_profile, save_calibration_profile
from .discard_layout import build_discard_layout
from .discard_parser import _collect_discard_slot_metrics, crop_discard_slot, is_probably_occupied_discard_slot
from .discard_quad_finder import refine_discard_slot_quad
from .tile_templates import build_hand_tile_template_payload
from .vit_tile_classifier import classify_tile_crops, vit_device_from_config, vit_model_from_config, vit_top_k_from_config


DEFAULT_MIN_CONFIDENCE = 0.85
DEFAULT_MIN_MARGIN = 0.20
DEFAULT_MAX_SAMPLES_PER_TILE = 12


@dataclass
class VitTemplateTrainingReport:
    crop_roots: list[str] = field(default_factory=list)
    frame_count: int = 0
    frame_sizes: list[str] = field(default_factory=list)
    total_crops: int = 0
    accepted_crops: int = 0
    rejected_crops: int = 0
    accepted_by_tile: dict[str, int] = field(default_factory=dict)
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    template_stored_sample_count: int = 0
    template_source_sample_count: int = 0
    output_profile_path: str = ""
    output_report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_roots": list(self.crop_roots),
            "frame_count": self.frame_count,
            "frame_sizes": list(self.frame_sizes),
            "total_crops": self.total_crops,
            "accepted_crops": self.accepted_crops,
            "rejected_crops": self.rejected_crops,
            "accepted_by_tile": dict(self.accepted_by_tile),
            "rejected_reasons": dict(self.rejected_reasons),
            "template_stored_sample_count": self.template_stored_sample_count,
            "template_source_sample_count": self.template_source_sample_count,
            "output_profile_path": self.output_profile_path,
            "output_report_path": self.output_report_path,
        }


def train_profile_templates_from_vit_crops(
    crop_roots: list[Path],
    *,
    base_profile_path: Path,
    output_profile_path: Path,
    output_report_path: Path | None = None,
    target: str = "discard",
    classifier_config: dict[str, Any] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_samples_per_tile: int = DEFAULT_MAX_SAMPLES_PER_TILE,
    batch_size: int = 64,
) -> VitTemplateTrainingReport:
    target = _normalize_target(target)
    crop_paths = _collect_crop_paths(crop_roots)
    report = VitTemplateTrainingReport(
        crop_roots=[str(path) for path in crop_roots],
        total_crops=len(crop_paths),
    )
    if not crop_paths:
        _write_report(report, output_report_path)
        return report

    samples: list[tuple[str, Image.Image]] = []
    accepted_by_tile: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    for batch_paths in _chunks(crop_paths, max(1, int(batch_size))):
        images = [_load_crop(path) for path in batch_paths]
        predictions = classify_tile_crops(
            images,
            model=vit_model_from_config(classifier_config),
            device=vit_device_from_config(classifier_config),
            top_k=vit_top_k_from_config(classifier_config),
        )
        for path, image, prediction in zip(batch_paths, images, predictions, strict=True):
            reason = _rejection_reason(prediction, min_confidence=min_confidence, min_margin=min_margin)
            if reason:
                rejected_reasons[reason] += 1
                continue
            assert prediction is not None
            samples.append((prediction.tile, image.copy()))
            accepted_by_tile[prediction.tile] += 1
            _ = path

    new_payload = build_hand_tile_template_payload(
        samples,
        max_samples_per_tile=max(1, int(max_samples_per_tile)),
    )
    base_profile = load_calibration_profile(base_profile_path)
    trained_profile = _profile_with_templates(
        base_profile,
        target=target,
        new_payload=new_payload,
        max_samples_per_tile=max(1, int(max_samples_per_tile)),
    )
    save_calibration_profile(trained_profile, output_profile_path)

    report.accepted_crops = len(samples)
    report.rejected_crops = report.total_crops - report.accepted_crops
    report.accepted_by_tile = dict(sorted(accepted_by_tile.items()))
    report.rejected_reasons = dict(sorted(rejected_reasons.items()))
    report.template_stored_sample_count = int(new_payload.get("stored_sample_count", 0) or 0)
    report.template_source_sample_count = int(new_payload.get("source_sample_count", 0) or 0)
    report.output_profile_path = str(output_profile_path)
    _write_report(report, output_report_path)
    return report


def train_profile_discard_templates_from_vit_frames(
    frame_paths: list[Path],
    *,
    base_profile_path: Path,
    output_profile_path: Path,
    output_report_path: Path | None = None,
    classifier_config: dict[str, Any] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_samples_per_tile: int = DEFAULT_MAX_SAMPLES_PER_TILE,
    batch_size: int = 64,
) -> VitTemplateTrainingReport:
    crop_images: list[tuple[Path, Image.Image]] = []
    frame_sizes: set[tuple[int, int]] = set()
    for frame_path in frame_paths:
        if not frame_path.exists():
            continue
        with Image.open(frame_path) as opened:
            image = opened.convert("RGB")
        frame_sizes.add(image.size)
        layout = build_discard_layout(*image.size)
        for slots in layout.values():
            for slot in slots:
                metrics = _collect_discard_slot_metrics(image, slot)
                occupied = is_probably_occupied_discard_slot(metrics) or refine_discard_slot_quad(image, slot) is not None
                if not occupied:
                    continue
                crop_images.append((frame_path, crop_discard_slot(image, slot, refine=True)))

    if len(frame_sizes) > 1:
        formatted = ", ".join(f"{width}x{height}" for width, height in sorted(frame_sizes))
        raise ValueError(f"runtime frame training requires one resolution per profile, got: {formatted}")
    profile_resolution = next(iter(frame_sizes), None)
    return _train_profile_from_labeled_crops(
        crop_images,
        crop_roots=[path.parent for path in frame_paths],
        frame_count=len(frame_paths),
        frame_sizes=[f"{width}x{height}" for width, height in sorted(frame_sizes)],
        base_profile_path=base_profile_path,
        output_profile_path=output_profile_path,
        output_report_path=output_report_path,
        target="discard",
        profile_resolution=profile_resolution,
        classifier_config=classifier_config,
        min_confidence=min_confidence,
        min_margin=min_margin,
        max_samples_per_tile=max_samples_per_tile,
        batch_size=batch_size,
    )


def discover_crop_roots(root: Path) -> list[Path]:
    if root.is_dir() and root.name.endswith("_crops"):
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*_crops") if path.is_dir())


def _collect_crop_paths(crop_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in crop_roots:
        if not root.exists():
            continue
        paths.extend(sorted(path for path in root.glob("*.png") if path.is_file()))
    return paths


def _train_profile_from_labeled_crops(
    crop_images: list[tuple[Path, Image.Image]],
    *,
    crop_roots: list[Path],
    frame_count: int = 0,
    frame_sizes: list[str] | None = None,
    base_profile_path: Path,
    output_profile_path: Path,
    output_report_path: Path | None,
    target: str,
    profile_resolution: tuple[int, int] | None = None,
    classifier_config: dict[str, Any] | None,
    min_confidence: float,
    min_margin: float,
    max_samples_per_tile: int,
    batch_size: int,
) -> VitTemplateTrainingReport:
    target = _normalize_target(target)
    report = VitTemplateTrainingReport(
        crop_roots=[str(path) for path in crop_roots],
        frame_count=frame_count,
        frame_sizes=list(frame_sizes or []),
        total_crops=len(crop_images),
    )
    samples: list[tuple[str, Image.Image]] = []
    accepted_by_tile: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    for batch_items in _chunks_any(crop_images, max(1, int(batch_size))):
        images = [image for _path, image in batch_items]
        predictions = classify_tile_crops(
            images,
            model=vit_model_from_config(classifier_config),
            device=vit_device_from_config(classifier_config),
            top_k=vit_top_k_from_config(classifier_config),
        )
        for (_path, image), prediction in zip(batch_items, predictions, strict=True):
            reason = _rejection_reason(prediction, min_confidence=min_confidence, min_margin=min_margin)
            if reason:
                rejected_reasons[reason] += 1
                continue
            assert prediction is not None
            samples.append((prediction.tile, image.copy()))
            accepted_by_tile[prediction.tile] += 1

    new_payload = build_hand_tile_template_payload(
        samples,
        max_samples_per_tile=max(1, int(max_samples_per_tile)),
    )
    base_profile = _profile_for_training_resolution(
        load_calibration_profile(base_profile_path),
        profile_resolution=profile_resolution,
    )
    trained_profile = _profile_with_templates(
        base_profile,
        target=target,
        new_payload=new_payload,
        max_samples_per_tile=max(1, int(max_samples_per_tile)),
    )
    save_calibration_profile(trained_profile, output_profile_path)

    report.accepted_crops = len(samples)
    report.rejected_crops = report.total_crops - report.accepted_crops
    report.accepted_by_tile = dict(sorted(accepted_by_tile.items()))
    report.rejected_reasons = dict(sorted(rejected_reasons.items()))
    report.template_stored_sample_count = int(new_payload.get("stored_sample_count", 0) or 0)
    report.template_source_sample_count = int(new_payload.get("source_sample_count", 0) or 0)
    report.output_profile_path = str(output_profile_path)
    _write_report(report, output_report_path)
    return report


def _profile_with_templates(
    base_profile: CalibrationProfile,
    *,
    target: str,
    new_payload: dict[str, Any],
    max_samples_per_tile: int,
) -> CalibrationProfile:
    hand_payload = copy.deepcopy(base_profile.hand_tile_templates)
    discard_payload = copy.deepcopy(base_profile.discard_tile_templates)
    if target == "hand":
        hand_payload = _merge_template_payloads(new_payload, hand_payload, max_samples_per_tile=max_samples_per_tile)
    else:
        discard_payload = _merge_template_payloads(new_payload, discard_payload, max_samples_per_tile=max_samples_per_tile)

    notes = [note for note in base_profile.notes if not str(note).startswith("vit_template_training_")]
    notes.extend(
        [
            f"vit_template_training_target={target}",
            f"vit_template_training_new_samples={int(new_payload.get('source_sample_count', 0) or 0)}",
        ]
    )
    return CalibrationProfile(
        profile_id=f"{base_profile.profile_id}-vit-{target}",
        version=base_profile.version,
        source=f"{base_profile.source}; vit_template_training",
        enabled=base_profile.enabled,
        screen_width=base_profile.screen_width,
        screen_height=base_profile.screen_height,
        confidence=max(base_profile.confidence, 0.96),
        hand_offsets=base_profile.hand_offsets,
        meld_offsets=base_profile.meld_offsets,
        dora_offsets=base_profile.dora_offsets,
        hand_tile_templates=hand_payload,
        discard_tile_templates=discard_payload,
        notes=notes,
    )


def _profile_for_training_resolution(
    base_profile: CalibrationProfile,
    *,
    profile_resolution: tuple[int, int] | None,
) -> CalibrationProfile:
    if profile_resolution is None:
        return base_profile
    width, height = profile_resolution
    if base_profile.screen_width == width and base_profile.screen_height == height:
        return base_profile
    if base_profile.screen_width > 0 and base_profile.screen_height > 0 and width > 0 and height > 0:
        aspect_delta = abs((base_profile.screen_width / base_profile.screen_height) - (width / height))
        if aspect_delta <= 0.02:
            return _scale_profile(base_profile, width=width, height=height)
    adjusted = copy.deepcopy(base_profile)
    adjusted.profile_id = f"{base_profile.profile_id}-runtime-{width}x{height}"
    adjusted.source = f"{base_profile.source} (runtime training resolution {width}x{height})"
    adjusted.screen_width = width
    adjusted.screen_height = height
    adjusted.confidence = round(max(0.0, min(0.95, base_profile.confidence * 0.75)), 3)
    adjusted.notes = [
        *base_profile.notes,
        f"runtime_training_resolution={width}x{height}",
    ]
    return adjusted


def _merge_template_payloads(
    preferred_payload: dict[str, Any],
    fallback_payload: dict[str, Any],
    *,
    max_samples_per_tile: int,
) -> dict[str, Any]:
    if not preferred_payload:
        return copy.deepcopy(fallback_payload)
    if not fallback_payload:
        return copy.deepcopy(preferred_payload)

    merged = copy.deepcopy(fallback_payload)
    merged["max_samples_per_tile"] = max_samples_per_tile
    merged_templates: dict[str, dict[str, Any]] = {}
    tile_keys = set(_templates(preferred_payload)) | set(_templates(fallback_payload))
    for tile in sorted(tile_keys):
        preferred_item = _templates(preferred_payload).get(tile, {})
        fallback_item = _templates(fallback_payload).get(tile, {})
        signatures = [
            *_signature_list(preferred_item.get("signatures")),
            *_signature_list(fallback_item.get("signatures")),
        ][:max_samples_per_tile]
        if not signatures:
            continue
        merged_templates[tile] = {
            **fallback_item,
            "count": _coerce_int(preferred_item.get("count"), default=len(_signature_list(preferred_item.get("signatures"))))
            + _coerce_int(fallback_item.get("count"), default=len(_signature_list(fallback_item.get("signatures")))),
            "signatures": signatures,
        }

    merged["templates"] = merged_templates
    merged["source_sample_count"] = _coerce_int(preferred_payload.get("source_sample_count"), default=0) + _coerce_int(
        fallback_payload.get("source_sample_count"),
        default=0,
    )
    merged["stored_sample_count"] = sum(
        len(_signature_list(item.get("signatures")))
        for item in merged_templates.values()
        if isinstance(item, dict)
    )
    merged["source"] = "vit_template_training_merge"
    return merged


def _rejection_reason(prediction: Any, *, min_confidence: float, min_margin: float) -> str:
    if prediction is None:
        return "no_prediction"
    if float(prediction.confidence or 0.0) < min_confidence:
        return "low_confidence"
    top_k = prediction.top_k if isinstance(prediction.top_k, list) else []
    if len(top_k) >= 2:
        first = _coerce_float(top_k[0].get("score"), default=0.0)
        second = _coerce_float(top_k[1].get("score"), default=0.0)
        if first - second < min_margin:
            return "low_margin"
    return ""


def _load_crop(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _write_report(report: VitTemplateTrainingReport, output_report_path: Path | None) -> None:
    if output_report_path is None:
        return
    report.output_report_path = str(output_report_path)
    write_json_atomic(output_report_path, report.to_dict())


def _normalize_target(target: str) -> str:
    value = str(target or "discard").strip().lower()
    if value not in {"discard", "hand"}:
        raise ValueError("target must be 'discard' or 'hand'")
    return value


def _chunks(items: list[Path], size: int) -> list[list[Path]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _chunks_any(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _templates(payload: dict[str, Any]) -> dict[str, Any]:
    templates = payload.get("templates")
    return templates if isinstance(templates, dict) else {}


def _signature_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


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
