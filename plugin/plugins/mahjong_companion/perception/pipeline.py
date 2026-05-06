from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..contracts import PerceivedGameState
from .action_detector import detect_actions, detect_button_regions
from .calibration import CalibrationProfile, resolve_calibration_profile
from .roi import build_default_rois, collect_region_metrics
from .scene_classifier import classify_scene
from .tile_parser import enrich_perceived_state_with_tiles

SCENE_PROMOTION_BUTTONS = {"chi", "pon", "kan", "riichi", "ron", "tsumo", "skip"}


def analyze_image_path(
    image_path: Path,
    *,
    calibration_dir: Path | None = None,
    template_dir: Path | None = None,
    fixture_mode: str = "auto",
) -> tuple[PerceivedGameState, dict[str, Any]]:
    if not image_path.exists():
        raise FileNotFoundError("image not found: %s" % image_path)

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        rois = build_default_rois(width, height)
        metrics: dict[str, dict[str, Any]] = {
            "full_frame": collect_region_metrics(image, None),
        }
        for name, roi in rois.items():
            metrics[name] = collect_region_metrics(image, roi)

    scene, confidence, scene_notes, roi_hits = classify_scene(metrics)
    buttons, is_user_turn, action_notes = detect_actions(scene, metrics)
    notes = scene_notes + action_notes
    profile = resolve_calibration_profile(width, height, calibration_dir=calibration_dir)
    button_regions = detect_button_regions(
        image,
        metrics,
        profile=profile,
        template_dir=template_dir or Path(__file__).resolve().parent / "templates",
    )
    if button_regions:
        matched_button_types = {region.button_type for region in button_regions}
        for region in button_regions:
            if region.button_type not in buttons:
                buttons.append(region.button_type)
        if scene in {"in_match", "dialog", "unknown"}:
            is_user_turn = True
        if scene not in {"in_match", "dialog", "replay"} and matched_button_types & SCENE_PROMOTION_BUTTONS:
            scene = "in_match"
            confidence = max(confidence, 0.74)
            is_user_turn = True
            notes.append("action button template evidence promotes scene to in_match")
        notes.append(f"button template matches: {len(button_regions)}")

    perceived = PerceivedGameState(
        scene=scene,
        confidence=confidence,
        is_user_turn=is_user_turn,
        buttons=buttons,
        notes=notes,
        roi_hits=roi_hits,
        button_regions=[region.to_dict() for region in button_regions],
    )
    perceived = enrich_perceived_state_with_tiles(
        perceived,
        image_path,
        image,
        metrics=metrics,
        calibration_dir=calibration_dir,
        fixture_mode=fixture_mode,
    )
    debug_payload = {
        "image_path": str(image_path),
        "image_size": {"width": width, "height": height},
        "roi_boxes": {name: roi.to_dict() for name, roi in rois.items()},
        "roi_metrics": metrics,
        "button_regions_count": len(button_regions),
        "button_regions_max_confidence": max(
            (region.confidence for region in button_regions),
            default=0.0,
        ),
    }
    return perceived, debug_payload


def analyze_action_buttons_fast(
    image_path: Path,
    *,
    calibration_dir: Path | None = None,
    template_dir: Path | None = None,
) -> tuple[PerceivedGameState, dict[str, Any]]:
    if not image_path.exists():
        raise FileNotFoundError("image not found: %s" % image_path)

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        rois = build_default_rois(width, height)
        metrics: dict[str, dict[str, Any]] = {
            "full_frame": collect_region_metrics(image, None, sample_step=18),
            "center_dialog": collect_region_metrics(image, rois["center_dialog"], sample_step=10),
            "bottom_action_bar": collect_region_metrics(image, rois["bottom_action_bar"], sample_step=8),
            "bottom_hand_area": collect_region_metrics(image, rois["bottom_hand_area"], sample_step=12),
            "right_replay_panel": collect_region_metrics(image, rois["right_replay_panel"], sample_step=18),
        }
        metrics["top_banner"] = {
            "box": rois["top_banner"].to_dict(),
            "gold_ratio": 0.0,
            "orange_ratio": 0.0,
            "bright_ratio": 0.0,
            "dark_ratio": 0.0,
            "mean_luma": 0.0,
            "stddev": 999.0,
            "colorful_ratio": 0.0,
        }

        button_regions = detect_button_regions(
            image,
            metrics,
            profile=CalibrationProfile(
                profile_id=f"fast-button-{width}x{height}",
                enabled=True,
                screen_width=width,
                screen_height=height,
            ),
            template_dir=template_dir or Path(__file__).resolve().parent / "templates",
            search_center=False,
        )

    buttons = [region.button_type for region in button_regions]
    scene = "in_match" if buttons else "unknown"
    perceived = PerceivedGameState(
        scene=scene,
        confidence=0.8 if buttons else 0.0,
        is_user_turn=bool(buttons),
        buttons=buttons,
        notes=["fast action button scan"] if buttons else ["fast action button scan: no buttons"],
        roi_hits={"bottom_action_bar": bool(buttons)},
        button_regions=[region.to_dict() for region in button_regions],
    )
    return perceived, {
        "image_path": str(image_path),
        "image_size": {"width": width, "height": height},
        "button_regions_count": len(button_regions),
        "button_regions_max_confidence": max(
            (region.confidence for region in button_regions),
            default=0.0,
        ),
        "source": "action_buttons_fast",
    }
