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
from .river_detector_v2 import (
    RiverDetectionResult,
    RiverTileCandidate,
    crop_river_candidate,
    detect_river_tiles_v2,
    expand_candidate_quad_for_classification,
)
from .tile_parser import TileParseResult, enrich_perceived_state_with_tiles, parse_tiles_from_image

__all__ = [
    "CalibrationProfile",
    "ButtonRegion",
    "DiscardParseResult",
    "DiscardSlot",
    "RiverDetectionResult",
    "RiverTileCandidate",
    "TileParseResult",
    "TileSlot",
    "analyze_image_path",
    "analyze_action_buttons_fast",
    "build_default_calibration_profile",
    "build_discard_layout",
    "build_hand_layout",
    "detect_button_regions",
    "detect_river_tiles_v2",
    "crop_river_candidate",
    "expand_candidate_quad_for_classification",
    "enrich_perceived_state_with_tiles",
    "label_sidecar_path",
    "load_calibration_profile",
    "parse_discards_from_image",
    "parse_tiles_from_image",
    "resolve_calibration_profile",
    "save_calibration_profile",
    "train_calibration_profile",
    "write_calibration_label",
]
