from __future__ import annotations

import sys

sys.path.insert(0, "plugin/plugins/mahjong_companion")

from PIL import Image, ImageDraw
import numpy as np

from plugin.plugins.mahjong_companion.perception.river_detector_v2 import (
    RIVER_PLAYERS,
    RiverDetectorParams,
    RiverRoi,
    RiverTileCandidate,
    _calibrate_right_river_template,
    _looks_like_overlap_tail_only,
    _perspective_quad,
    _renumber_by_player,
    _side_player_grid_additions,
    _stabilize_lower_side_visible_quads,
    build_river_rois,
    crop_river_candidate,
    expand_candidate_quad_for_classification,
    detect_river_tiles_v2,
    river_candidate_classification_rejection_reason,
    river_candidate_looks_blank,
)
from plugin.plugins.mahjong_companion.scripts.experiment_sam_river_masks import build_point_prompts, build_prompt_boxes, mask_to_quad


def test_build_river_rois_keeps_all_players_inside_image():
    rois = build_river_rois(2560, 1440)
    assert [roi.player for roi in rois] == [
        "top_opponent",
        "left_opponent",
        "right_opponent",
        "self",
    ]
    for roi in rois:
        assert 0 <= roi.left < roi.right <= 2560
        assert 0 <= roi.top < roi.bottom <= 1440


def test_detect_river_tiles_ignores_bottom_hand_area():
    image = Image.new("RGB", (1920, 1080), (35, 70, 110))
    draw = ImageDraw.Draw(image)

    # Paint three discard-like tiles in self river ROI.
    for left in (760, 825, 890):
        draw.rounded_rectangle((left, 560, left + 52, 632), radius=5, fill=(218, 214, 198))
        draw.rectangle((left + 8, 580, left + 44, 610), fill=(30, 30, 30))

    # Paint hand-like tiles below the river ROIs; these must not be counted.
    for left in (420, 490, 560, 630):
        draw.rounded_rectangle((left, 930, left + 78, 1060), radius=7, fill=(222, 222, 214))

    result = detect_river_tiles_v2(image)

    assert len(result.by_player["self"]) == 3
    assert all(candidate.bbox[1] < 800 for candidate in result.by_player["self"])
    for player in RIVER_PLAYERS:
        if player != "self":
            assert result.by_player[player] == []


def test_self_action_panel_tiles_are_not_counted_as_river():
    image = Image.new("RGB", (1920, 1080), (35, 70, 110))
    draw = ImageDraw.Draw(image)

    # Real self river tiles sit above the action-selection panel.
    for left in (760, 825, 890):
        draw.rounded_rectangle((left, 560, left + 52, 632), radius=5, fill=(218, 214, 198))
        draw.rectangle((left + 8, 580, left + 44, 610), fill=(30, 30, 30))

    # Mahjong Soul's call/kan selection panel can show selectable hand tiles
    # inside the broad self river ROI; these must not become discard candidates.
    draw.rounded_rectangle((670, 715, 1265, 860), radius=12, fill=(116, 50, 54))
    for left in (760, 850, 940, 1030):
        draw.rounded_rectangle((left, 730, left + 70, 830), radius=5, fill=(218, 214, 198))
        draw.rectangle((left + 10, 760, left + 60, 805), fill=(30, 30, 30))

    result = detect_river_tiles_v2(image)

    assert len(result.by_player["self"]) == 3
    assert all(candidate.center[1] < 1080 * 0.64 for candidate in result.by_player["self"])


def test_opponent_candidates_use_perspective_quads():
    image = Image.new("RGB", (1920, 1080), (35, 70, 110))
    draw = ImageDraw.Draw(image)

    draw.polygon([(560, 430), (550, 500), (618, 500), (628, 430)], fill=(218, 214, 198))
    draw.rectangle((578, 455, 608, 480), fill=(30, 30, 30))

    result = detect_river_tiles_v2(image)
    candidate = result.by_player["left_opponent"][0]

    assert candidate.quad != (
        (candidate.bbox[0], candidate.bbox[1]),
        (candidate.bbox[0], candidate.bbox[3]),
        (candidate.bbox[2], candidate.bbox[3]),
        (candidate.bbox[2], candidate.bbox[1]),
    )
    assert candidate.to_dict()["quad_order"] == "upper_left,lower_left,lower_right,upper_right"
    crop = crop_river_candidate(image, candidate)
    assert crop.width > 0
    assert crop.height > 0


def test_classification_crop_expands_side_completion_quad():
    candidate = RiverTileCandidate(
        player="right_opponent",
        order_index=15,
        bbox=(1728, 651, 1842, 721),
        quad=((1752, 649), (1756, 723), (1847, 725), (1847, 646)),
        center=(1785, 686),
        confidence=0.86,
        source="river_detector_v2_completion",
    )

    expanded = expand_candidate_quad_for_classification(candidate)

    original_width = max(x for x, _y in candidate.quad) - min(x for x, _y in candidate.quad)
    expanded_width = max(x for x, _y in expanded) - min(x for x, _y in expanded)
    original_height = max(y for _x, y in candidate.quad) - min(y for _x, y in candidate.quad)
    expanded_height = max(y for _x, y in expanded) - min(y for _x, y in expanded)
    assert expanded_width > original_width
    assert expanded_height > original_height
    assert min(x for x, _y in expanded) <= candidate.bbox[0]
    assert max(x for x, _y in expanded) >= candidate.bbox[2]


def test_blank_river_candidate_is_flagged_unknown():
    image = Image.new("RGB", (220, 220), (35, 70, 110))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 60, 140, 150), radius=6, fill=(137, 188, 214))
    candidate = RiverTileCandidate(
        player="self",
        order_index=1,
        bbox=(60, 60, 140, 150),
        quad=((60, 60), (60, 150), (140, 150), (140, 60)),
        center=(100, 105),
        confidence=0.8,
    )

    assert river_candidate_looks_blank(image, candidate)


def test_text_river_candidate_is_not_flagged_blank():
    image = Image.new("RGB", (220, 220), (35, 70, 110))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 60, 140, 150), radius=6, fill=(218, 214, 198))
    draw.rectangle((78, 86, 122, 124), fill=(170, 30, 30))
    candidate = RiverTileCandidate(
        player="self",
        order_index=1,
        bbox=(60, 60, 140, 150),
        quad=((60, 60), (60, 150), (140, 150), (140, 60)),
        center=(100, 105),
        confidence=0.8,
    )

    assert not river_candidate_looks_blank(image, candidate)


def test_very_low_confidence_side_candidate_is_not_classified():
    image = Image.new("RGB", (220, 220), (35, 70, 110))
    candidate = RiverTileCandidate(
        player="left_opponent",
        order_index=1,
        bbox=(60, 60, 140, 150),
        quad=((60, 60), (60, 150), (140, 150), (140, 60)),
        center=(100, 105),
        confidence=0.44,
    )

    assert river_candidate_classification_rejection_reason(image, candidate) == "low_side_candidate_confidence"


def test_moderate_confidence_side_candidate_can_be_classified():
    image = Image.new("RGB", (220, 220), (35, 70, 110))
    candidate = RiverTileCandidate(
        player="left_opponent",
        order_index=1,
        bbox=(60, 60, 140, 150),
        quad=((60, 60), (60, 150), (140, 150), (140, 60)),
        center=(100, 105),
        confidence=0.51,
    )

    assert river_candidate_classification_rejection_reason(image, candidate) == ""


def test_side_completion_extends_after_reference_fill():
    image = Image.new("RGB", (360, 360), (35, 70, 110))
    draw = ImageDraw.Draw(image)
    roi = RiverRoi("left_opponent", 40, 40, 320, 320, "row_major")
    candidates: list[RiverTileCandidate] = []
    for row_index, y in enumerate((60, 105, 150, 195, 240)):
        xs = (100, 190) if row_index == 0 else (190,)
        for x in xs:
            bbox = (x - 40, y - 28, x + 40, y + 28)
            draw.rectangle(bbox, fill=(218, 214, 198))
            candidates.append(
                RiverTileCandidate(
                    player="left_opponent",
                    order_index=len(candidates) + 1,
                    bbox=bbox,
                    quad=((bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])),
                    center=(x, y),
                    confidence=0.8,
                )
            )
    draw.rectangle((60, 257, 140, 313), fill=(218, 214, 198))
    draw.rectangle((150, 257, 230, 313), fill=(218, 214, 198))

    additions = _side_player_grid_additions(
        candidates,
        player="left_opponent",
        roi=roi,
        arr=np.asarray(image),
        image_area=360 * 360,
        params=RiverDetectorParams(),
    )

    assert any(candidate.center[1] > 260 for candidate in additions)


def test_side_completion_fills_vertical_single_column_gap():
    image = Image.new("RGB", (360, 360), (35, 70, 110))
    draw = ImageDraw.Draw(image)
    roi = RiverRoi("right_opponent", 40, 40, 320, 320, "row_major")
    candidates: list[RiverTileCandidate] = []
    for y in (70, 170, 220, 270, 315):
        bbox = (130, y - 32, 230, y + 32)
        draw.rectangle(bbox, fill=(218, 214, 198))
        candidates.append(
            RiverTileCandidate(
                player="right_opponent",
                order_index=len(candidates) + 1,
                bbox=bbox,
                quad=((bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])),
                center=(180, y),
                confidence=0.8,
            )
        )
    draw.rectangle((130, 88, 230, 152), fill=(218, 214, 198))

    additions = _side_player_grid_additions(
        candidates,
        player="right_opponent",
        roi=roi,
        arr=np.asarray(image),
        image_area=360 * 360,
        params=RiverDetectorParams(),
    )

    assert any(105 <= candidate.center[1] <= 135 for candidate in additions)


def test_overlap_tail_completion_gate_keeps_real_lower_tile():
    occupied = [
        RiverTileCandidate(
            player="left_opponent",
            order_index=10,
            bbox=(50, 50, 150, 120),
            quad=((50, 50), (50, 120), (150, 120), (150, 50)),
            center=(100, 85),
            confidence=0.8,
        )
    ]

    false_tail = Image.new("RGB", (220, 220), (35, 70, 110))
    draw = ImageDraw.Draw(false_tail)
    draw.rectangle((50, 100, 150, 132), fill=(218, 214, 198))
    assert _looks_like_overlap_tail_only(np.asarray(false_tail), (50, 100, 150, 170), occupied)

    real_lower = Image.new("RGB", (220, 220), (35, 70, 110))
    draw = ImageDraw.Draw(real_lower)
    draw.rectangle((50, 100, 150, 132), fill=(218, 214, 198))
    draw.rectangle((50, 140, 150, 170), fill=(220, 150, 45))
    assert not _looks_like_overlap_tail_only(np.asarray(real_lower), (50, 100, 150, 170), occupied)


def test_renumber_by_player_preserves_source_and_quad():
    roi = RiverRoi("right_opponent", 100, 100, 500, 500, "row_major")
    candidate = RiverTileCandidate(
        player="right_opponent",
        order_index=0,
        bbox=(140, 140, 240, 210),
        quad=((155, 140), (140, 210), (225, 210), (240, 140)),
        center=(190, 175),
        confidence=0.7,
        source="river_detector_v2_completion",
    )

    renumbered = _renumber_by_player([candidate], [roi])

    assert renumbered[0].order_index == 1
    assert renumbered[0].source == "river_detector_v2_completion"
    assert renumbered[0].quad == candidate.quad


def test_lower_right_visible_quad_clamps_to_its_detection_box():
    roi = RiverRoi("right_opponent", 100, 100, 500, 500, "row_major")
    candidate = RiverTileCandidate(
        player="right_opponent",
        order_index=15,
        bbox=(220, 340, 330, 420),
        quad=((214, 336), (216, 424), (336, 426), (338, 334)),
        center=(275, 380),
        confidence=0.86,
    )

    stabilized = _stabilize_lower_side_visible_quads([candidate], [roi])[0]

    assert stabilized.quad == ((220, 340), (220, 420), (330, 420), (330, 340))


def test_right_river_template_keeps_upper_contours_and_stabilizes_lower_rows():
    roi = RiverRoi("right_opponent", 100, 100, 600, 600, "row_major")
    candidates: list[RiverTileCandidate] = []
    for row_index, y in enumerate((130, 180, 230, 285, 340, 395)):
        columns = (160, 260) if row_index < 3 else (160, 260, 360)
        for x in columns:
            bbox = (x, y, x + 100, y + 70)
            candidates.append(
                RiverTileCandidate(
                    player="right_opponent",
                    order_index=len(candidates) + 1,
                    bbox=bbox,
                    quad=((x + 12, y + 2), (x + 4, y + 65), (x + 96, y + 62), (x + 90, y + 5)),
                    center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                    confidence=0.8,
                )
            )

    calibrated = _calibrate_right_river_template(candidates, [roi])

    assert calibrated[0].quad == candidates[0].quad
    assert calibrated[-1].quad == _perspective_quad("right_opponent", calibrated[-1].bbox)
    assert calibrated[-1].quad != candidates[-1].quad


def test_lower_right_visible_quad_keeps_left_edge_from_collapsing():
    roi = RiverRoi("right_opponent", 100, 100, 500, 500, "row_major")
    candidate = RiverTileCandidate(
        player="right_opponent",
        order_index=15,
        bbox=(220, 340, 330, 420),
        quad=((250, 340), (254, 420), (330, 420), (330, 340)),
        center=(275, 380),
        confidence=0.86,
    )

    stabilized = _stabilize_lower_side_visible_quads([candidate], [roi])[0]

    assert stabilized.quad[0][0] == 233
    assert stabilized.quad[1][0] == 233


def test_upper_right_visible_quad_keeps_fitted_contour():
    roi = RiverRoi("right_opponent", 100, 100, 500, 500, "row_major")
    candidate = RiverTileCandidate(
        player="right_opponent",
        order_index=1,
        bbox=(220, 120, 330, 200),
        quad=((214, 116), (216, 204), (336, 206), (338, 114)),
        center=(275, 160),
        confidence=0.86,
    )

    stabilized = _stabilize_lower_side_visible_quads([candidate], [roi])[0]

    assert stabilized.quad == candidate.quad


def test_sam_mask_to_quad_fits_rotated_tile_mask():
    mask = np.zeros((160, 160), dtype=np.uint8)
    points = np.asarray([(55, 30), (42, 120), (116, 126), (128, 38)], dtype=np.int32)
    import cv2

    cv2.fillConvexPoly(mask, points, 1)

    quad = mask_to_quad(mask, fallback_bbox=(40, 30, 130, 130))

    assert quad is not None
    assert min(x for x, _y in quad) <= 56
    assert max(x for x, _y in quad) >= 116
    assert min(y for _x, y in quad) <= 38
    assert max(y for _x, y in quad) >= 120


def test_adaptive_sam_prompt_boxes_keep_detector_bbox_inside():
    candidates = [
        RiverTileCandidate(
            player="right_opponent",
            order_index=index,
            bbox=(left, 100, left + 112, 170),
            quad=((left + 5, 104), (left + 5, 166), (left + 108, 164), (left + 104, 102)),
            center=(left + 56, 135),
            confidence=0.8,
        )
        for index, left in enumerate((150, 245, 340), start=1)
    ]

    boxes = build_prompt_boxes(candidates, (600, 400), padding=14, mode="adaptive")

    for candidate, box in zip(candidates, boxes, strict=True):
        assert box[0] <= candidate.bbox[0]
        assert box[1] <= candidate.bbox[1]
        assert box[2] >= candidate.bbox[2]
        assert box[3] >= candidate.bbox[3]
    assert boxes[0][0] < candidates[0].bbox[0]
    assert boxes[2][2] > candidates[2].bbox[2]


def test_box_points_sam_prompts_include_neighbor_negative_points():
    candidates = [
        RiverTileCandidate(
            player="right_opponent",
            order_index=index,
            bbox=(left, top, left + 100, top + 70),
            quad=((left + 8, top + 4), (left + 4, top + 66), (left + 96, top + 64), (left + 92, top + 2)),
            center=(left + 50, top + 35),
            confidence=0.8,
        )
        for index, (left, top) in enumerate(((150, 100), (245, 100), (150, 160)), start=1)
    ]

    prompts = build_point_prompts(candidates, mode="box-points")

    assert prompts[0]["positive"]
    assert candidates[1].center in prompts[0]["negative"]
    assert candidates[2].center in prompts[0]["negative"]
