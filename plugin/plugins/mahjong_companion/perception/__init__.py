from __future__ import annotations

from .action_detector import ButtonRegion, detect_button_regions
from .bottom_hand_detector import BottomHandDetection, BottomHandSlot, detect_bottom_hand_tiles
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
from .model_river_adapter import (
    ModelRiverCandidate,
    ModelRiverConfig,
    ModelRiverDetection,
    parse_model_river_from_json,
)
from .pipeline import analyze_action_buttons_fast, analyze_image_path
from .river_detector_v2 import (
    MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE,
    RiverDetectionResult,
    RiverTileCandidate,
    crop_river_candidate,
    detect_river_tiles_v2,
    expand_candidate_quad_for_classification,
    river_candidate_classification_rejection_reason,
    river_candidate_looks_blank,
)
from .tile_parser import TileParseResult, enrich_perceived_state_with_tiles, parse_tiles_from_image

__all__ = [
    "CalibrationProfile",
    "ButtonRegion",
    "BottomHandDetection",
    "BottomHandSlot",
    "DiscardParseResult",
    "DiscardSlot",
    "RiverDetectionResult",
    "RiverTileCandidate",
    "ModelRiverCandidate",
    "ModelRiverConfig",
    "ModelRiverDetection",
    "MIN_RIVER_TILE_CLASSIFICATION_CONFIDENCE",
    "TileParseResult",
    "TileSlot",
    "analyze_image_path",
    "analyze_action_buttons_fast",
    "build_default_calibration_profile",
    "build_discard_layout",
    "build_hand_layout",
    "detect_button_regions",
    "detect_bottom_hand_tiles",
    "detect_river_tiles_v2",
    "crop_river_candidate",
    "expand_candidate_quad_for_classification",
    "river_candidate_classification_rejection_reason",
    "river_candidate_looks_blank",
    "enrich_perceived_state_with_tiles",
    "label_sidecar_path",
    "load_calibration_profile",
    "parse_discards_from_image",
    "parse_model_river_from_json",
    "parse_tiles_from_image",
    "resolve_calibration_profile",
    "save_calibration_profile",
    "train_calibration_profile",
    "write_calibration_label",
]
