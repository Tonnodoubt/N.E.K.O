from __future__ import annotations

from .action_detector import ButtonRegion, detect_button_regions
from .calibration import (
    CalibrationProfile,
    build_default_calibration_profile,
    label_sidecar_path,
    load_calibration_profile,
    resolve_calibration_profile,
    save_calibration_profile,
    train_calibration_profile,
    write_calibration_label,
)
from .discard_layout import DiscardSlot, build_discard_layout
from .discard_parser import DiscardParseResult, parse_discards_from_image
from .hand_layout import TileSlot, build_hand_layout
from .pipeline import analyze_action_buttons_fast, analyze_image_path
from .tile_parser import TileParseResult, enrich_perceived_state_with_tiles, parse_tiles_from_image
from .vit_template_training import (
    VitTemplateTrainingReport,
    train_profile_discard_templates_from_vit_frames,
    train_profile_templates_from_vit_crops,
)
from .vit_tile_classifier import VitTileClassifierUnavailable, VitTilePrediction, classify_tile_crops

__all__ = [
    "CalibrationProfile",
    "ButtonRegion",
    "DiscardParseResult",
    "DiscardSlot",
    "TileParseResult",
    "TileSlot",
    "VitTileClassifierUnavailable",
    "VitTilePrediction",
    "VitTemplateTrainingReport",
    "analyze_image_path",
    "analyze_action_buttons_fast",
    "build_default_calibration_profile",
    "build_discard_layout",
    "build_hand_layout",
    "classify_tile_crops",
    "detect_button_regions",
    "enrich_perceived_state_with_tiles",
    "label_sidecar_path",
    "load_calibration_profile",
    "parse_discards_from_image",
    "parse_tiles_from_image",
    "resolve_calibration_profile",
    "save_calibration_profile",
    "train_calibration_profile",
    "train_profile_discard_templates_from_vit_frames",
    "train_profile_templates_from_vit_crops",
    "write_calibration_label",
]
