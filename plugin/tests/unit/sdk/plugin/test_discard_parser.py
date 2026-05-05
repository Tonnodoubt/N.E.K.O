from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.perception.calibration import CalibrationProfile, save_calibration_profile
from plugin.plugins.mahjong_companion.perception.discard_layout import DiscardSlot, build_discard_layout
from plugin.plugins.mahjong_companion.perception.discard_parser import (
    is_probably_occupied_discard_slot,
    parse_discards_from_image,
)
from plugin.plugins.mahjong_companion.perception.hand_layout import TileSlot, build_hand_layout
from plugin.plugins.mahjong_companion.perception.roi import collect_region_metrics
from plugin.plugins.mahjong_companion.perception.riichi_detector import detect_riichi_players
from plugin.plugins.mahjong_companion.perception.tile_parser import (
    _match_rejection_reason,
    enrich_perceived_state_with_tiles,
    parse_tiles_from_image,
)
from plugin.plugins.mahjong_companion.perception.tile_templates import build_hand_tile_template_payload, classify_tile_from_templates


def test_parse_discards_from_image_detects_bottom_and_side_tiles() -> None:
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    layout = build_discard_layout(*image.size)
    red_tile = _tile_image((210, 52, 58))
    green_tile = _tile_image((50, 150, 84))
    payload = build_hand_tile_template_payload([("1m", red_tile), ("2s", green_tile)])

    _paste_upright_tile(image, layout["self"][0], red_tile)
    _paste_right_oriented_tile(image, layout["right_opponent"][0], green_tile)

    parsed = parse_discards_from_image(image, payload, layout=layout)

    assert parsed.discard_piles["self"][0]["tile"] == "1m"
    assert parsed.discard_piles["self"][0]["turn_index"] == 1
    assert parsed.discard_piles["right_opponent"][0]["tile"] == "2s"
    assert parsed.discard_piles["right_opponent"][0]["orientation"] == "right"
    assert parsed.visible_tiles == ["1m", "2s"]
    assert parsed.analysis_hints["recognized_discard_tile_count"] == 2
    assert all(item["group"] == "discard" for item in parsed.raw_detections)


def test_parse_discards_from_image_assigns_refined_side_quad_to_best_slot() -> None:
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    layout = build_discard_layout(*image.size)
    tile = _tile_image((210, 52, 58))
    payload = build_hand_tile_template_payload([("5z", tile)])
    target_slot = layout["left_opponent"][9]

    _paste_left_oriented_tile(image, target_slot, tile)

    parsed = parse_discards_from_image(image, payload, layout=layout)

    left_pile = parsed.discard_piles["left_opponent"]
    assert [item["turn_index"] for item in left_pile] == [10]
    rejected_neighbor = next(
        item
        for item in parsed.raw_detections
        if item["slot_id"] == "discard_left_opponent_04"
        and item.get("rejected_refinement_owner_slot_id") == "discard_left_opponent_10"
    )
    assert rejected_neighbor["candidate_tile"] == "5z"


def test_parse_discards_from_image_can_emit_empty_debug_slots() -> None:
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    payload = build_hand_tile_template_payload([("1m", _tile_image((210, 52, 58)))])

    parsed = parse_discards_from_image(image, payload, include_empty_detections=True)

    assert parsed.discard_piles == {}
    assert parsed.visible_tiles == []
    assert len(parsed.raw_detections) == 72
    assert parsed.raw_detections[0]["occupied"] is False


def test_discard_occupancy_rejects_plain_table_and_accepts_tile_crop() -> None:
    empty = Image.new("RGB", (58, 70), color=(28, 58, 104))
    tile = _tile_image((210, 52, 58))

    assert not is_probably_occupied_discard_slot(collect_region_metrics(empty, None))
    assert is_probably_occupied_discard_slot(collect_region_metrics(tile, None))


def test_template_runner_up_uses_nearest_different_tile() -> None:
    seven_pin = _tile_image((210, 52, 58))
    seven_pin_variant = _tile_image((212, 54, 58))
    six_pin = _tile_image((218, 58, 58))
    payload = build_hand_tile_template_payload(
        [
            ("7p", seven_pin),
            ("7p", seven_pin_variant),
            ("6p", six_pin),
        ]
    )

    match = classify_tile_from_templates(seven_pin, payload)

    assert match is not None
    assert match.tile == "7p"
    assert match.runner_up_tile == "6p"


def test_parse_discards_rejects_ambiguous_pin_pair() -> None:
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    layout = build_discard_layout(*image.size)
    seven_pin = _tile_image((210, 52, 58))
    six_pin = _tile_image((212, 54, 58))
    payload = build_hand_tile_template_payload([("7p", seven_pin), ("6p", six_pin)])

    _paste_upright_tile(image, layout["self"][0], seven_pin)

    parsed = parse_discards_from_image(image, payload, layout=layout)

    assert parsed.discard_piles == {}
    rejected = next(
        item
        for item in parsed.raw_detections
        if item["slot_id"] == "discard_self_01"
    )
    assert rejected["candidate_tile"] == "7p"
    assert rejected["rejection_reason"] == "ambiguous_discard_template_pair"


def test_hand_parser_can_use_discard_template_for_missing_pin_tile(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    calibration_dir = tmp_path / "calibration"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    seven_pin = _tile_image((180, 54, 132))
    six_pin = _tile_image((190, 64, 132))
    five_pin = _tile_image((200, 74, 132))

    _paste_upright_tile(image, build_hand_layout(*image.size)["hand"][0], seven_pin)
    image.save(image_path)
    save_calibration_profile(
        CalibrationProfile(
            profile_id="test-1280x720",
            enabled=True,
            screen_width=1280,
            screen_height=720,
            confidence=0.9,
            hand_tile_templates=build_hand_tile_template_payload([("5p", five_pin), ("6p", six_pin)]),
            discard_tile_templates=build_hand_tile_template_payload([("7p", seven_pin)]),
        ),
        calibration_dir / "profiles" / "test-1280x720.json",
    )

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )

    assert parsed.hand_tiles == ["7p"]
    assert parsed.analysis_hints["recognized_hand_tile_count"] == 1


def test_hand_rejection_marks_adjacent_pin_pairs_ambiguous() -> None:
    class FakeMatch:
        tile = "5p"
        runner_up_tile = "6p"
        distance = 10.0
        runner_up_distance = 11.0
        confidence = 0.70

    assert _match_rejection_reason(FakeMatch()) == "ambiguous_5p_6p"

    FakeMatch.tile = "6p"
    FakeMatch.runner_up_tile = "7p"
    FakeMatch.confidence = 0.66

    assert _match_rejection_reason(FakeMatch()) == "ambiguous_6p_7p"


def test_visual_riichi_stick_detector_identifies_players() -> None:
    image = Image.new("RGB", (1920, 1080), color=(28, 58, 104))

    _draw_horizontal_riichi_stick(image, (900, 322, 1015, 340))
    _draw_horizontal_riichi_stick(image, (890, 516, 1030, 534))
    _draw_vertical_riichi_stick(image, (820, 350, 838, 465))
    _draw_vertical_riichi_stick(image, (1088, 350, 1106, 465))

    players, detections = detect_riichi_players(image)

    assert players == ["top_opponent", "self", "left_opponent", "right_opponent"]
    assert len(detections) == 4
    assert all(item["source"] == "riichi_stick_detector" for item in detections)


def test_parse_tiles_from_image_includes_visual_riichi_players(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    calibration_dir = tmp_path / "calibration"
    image = Image.new("RGB", (1920, 1080), color=(28, 58, 104))
    tile = _tile_image((210, 52, 58))

    _paste_upright_tile(image, build_hand_layout(*image.size)["hand"][0], tile)
    _draw_horizontal_riichi_stick(image, (900, 322, 1015, 340))
    image.save(image_path)
    save_calibration_profile(
        CalibrationProfile(
            profile_id="test-1920x1080",
            enabled=True,
            screen_width=1920,
            screen_height=1080,
            confidence=0.9,
            hand_tile_templates=build_hand_tile_template_payload([("1m", tile)]),
        ),
        calibration_dir / "profiles" / "test-1920x1080.json",
    )

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )

    assert parsed.riichi_players == ["top_opponent"]
    assert parsed.analysis_hints["riichi_stick_count"] == 1
    assert parsed.raw_detections[-1]["group"] == "riichi_stick"


def test_parse_tiles_from_image_includes_template_discard_piles(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    calibration_dir = tmp_path / "calibration"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    red_tile = _tile_image((210, 52, 58))
    payload = build_hand_tile_template_payload([("1m", red_tile)])

    _paste_upright_tile(image, build_hand_layout(*image.size)["hand"][0], red_tile)
    discard_layout = build_discard_layout(*image.size)
    _paste_upright_tile(image, discard_layout["self"][0], red_tile)
    _paste_right_oriented_tile(image, discard_layout["right_opponent"][0], red_tile)
    image.save(image_path)
    save_calibration_profile(
        CalibrationProfile(
            profile_id="test-1280x720",
            enabled=True,
            screen_width=1280,
            screen_height=720,
            confidence=0.9,
            hand_tile_templates=payload,
        ),
        calibration_dir / "profiles" / "test-1280x720.json",
    )

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )

    assert parsed.hand_tiles == ["1m"]
    assert parsed.discard_piles["self"][0]["tile"] == "1m"
    assert parsed.discard_piles["right_opponent"][0]["tile"] == "1m"
    assert parsed.visible_tiles == ["1m", "1m"]
    assert parsed.analysis_hints["recognized_discard_tile_count"] == 2

    enriched = enrich_perceived_state_with_tiles(
        PerceivedGameState(scene="in_match", confidence=0.9, riichi_players=["right_opponent"]),
        image_path,
        image,
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )
    assert enriched.riichi_players == ["right_opponent"]
    assert enriched.known_genbutsu_tiles == ["1m"]
    assert enriched.analysis_hints["known_genbutsu_tiles"] == ["1m"]


def test_parse_tiles_from_image_prefers_discard_tile_templates(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    calibration_dir = tmp_path / "calibration"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    hand_tile = _tile_image((54, 92, 205))
    discard_tile = _tile_image((210, 52, 58))

    _paste_upright_tile(image, build_hand_layout(*image.size)["hand"][0], hand_tile)
    _paste_upright_tile(image, build_discard_layout(*image.size)["self"][0], discard_tile)
    image.save(image_path)
    save_calibration_profile(
        CalibrationProfile(
            profile_id="test-1280x720",
            enabled=True,
            screen_width=1280,
            screen_height=720,
            confidence=0.9,
            hand_tile_templates=build_hand_tile_template_payload([("2p", hand_tile)]),
            discard_tile_templates=build_hand_tile_template_payload([("1m", discard_tile)]),
        ),
        calibration_dir / "profiles" / "test-1280x720.json",
    )

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )

    assert parsed.hand_tiles == ["2p"]
    assert parsed.discard_piles["self"][0]["tile"] == "1m"
    assert parsed.analysis_hints["discard_template_source"] == "discard_and_hand_tile_templates"


def test_parse_tiles_from_image_uses_hand_templates_for_non_discard_template_tiles(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    calibration_dir = tmp_path / "calibration"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    hand_tile = _tile_image((54, 92, 205))
    discard_tile = _tile_image((210, 52, 58))
    side_tile = _tile_image((50, 150, 84))

    _paste_upright_tile(image, build_hand_layout(*image.size)["hand"][0], hand_tile)
    _paste_upright_tile(image, build_discard_layout(*image.size)["self"][0], discard_tile)
    _paste_right_oriented_tile(image, build_discard_layout(*image.size)["right_opponent"][0], side_tile)
    image.save(image_path)
    save_calibration_profile(
        CalibrationProfile(
            profile_id="test-1280x720",
            enabled=True,
            screen_width=1280,
            screen_height=720,
            confidence=0.9,
            hand_tile_templates=build_hand_tile_template_payload([("2p", hand_tile), ("2s", side_tile)]),
            discard_tile_templates=build_hand_tile_template_payload([("1m", discard_tile)]),
        ),
        calibration_dir / "profiles" / "test-1280x720.json",
    )

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )

    assert parsed.discard_piles["self"][0]["tile"] == "1m"
    assert parsed.discard_piles["right_opponent"][0]["tile"] == "2s"
    assert parsed.analysis_hints["discard_template_source"] == "discard_and_hand_tile_templates"


def test_parse_tiles_from_image_can_use_external_discard_recognizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "frame.png"
    calibration_dir = tmp_path / "calibration"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    hand_tile = _tile_image((54, 92, 205))

    _paste_upright_tile(image, build_hand_layout(*image.size)["hand"][0], hand_tile)
    image.save(image_path)
    save_calibration_profile(
        CalibrationProfile(
            profile_id="test-1280x720",
            enabled=True,
            screen_width=1280,
            screen_height=720,
            confidence=0.9,
            hand_tile_templates=build_hand_tile_template_payload([("2p", hand_tile)]),
        ),
        calibration_dir / "profiles" / "test-1280x720.json",
    )

    recognizer = tmp_path / "recognizer.py"
    recognizer.write_text(
        "import json\n"
        "print(json.dumps({'detections': ["
        "{'player': 'right_opponent', 'turn_index': 4, 'tile': '7s', "
        "'confidence': 0.91, 'quad': [[10, 20], [12, 80], [90, 84], [88, 18]]}"
        "]}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MAHJONG_COMPANION_DISCARD_RECOGNIZER_CMD",
        f"{sys.executable} {recognizer} --image {{image_path}}",
    )

    parsed = parse_tiles_from_image(
        image_path,
        image,
        scene="in_match",
        metrics={"bottom_hand_area": {"colorful_ratio": 0.58}},
        calibration_dir=calibration_dir,
        fixture_mode="disabled",
    )

    assert parsed.hand_tiles == ["2p"]
    assert parsed.discard_piles["right_opponent"][0]["tile"] == "7s"
    assert parsed.discard_piles["right_opponent"][0]["quad"] == [[10, 20], [12, 80], [90, 84], [88, 18]]
    assert parsed.analysis_hints["discard_parser_source"] == "external_discard_recognizer"
    assert parsed.analysis_hints["external_discard_recognizer_count"] == 1


def _tile_image(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (39, 47), color=(238, 236, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 32, 33), fill=color)
    draw.rectangle((6, 36, 32, 41), fill=(218, 138, 28))
    return image


def _paste_upright_tile(image: Image.Image, slot: TileSlot | DiscardSlot, tile: Image.Image) -> None:
    box = slot.box
    image.paste(tile.resize((box.width, box.height)), (box.left, box.top))


def _paste_right_oriented_tile(image: Image.Image, slot: DiscardSlot, tile: Image.Image) -> None:
    box = slot.box
    rotated = tile.rotate(270, expand=True).resize((box.width, box.height))
    image.paste(rotated, (box.left, box.top))


def _paste_left_oriented_tile(image: Image.Image, slot: DiscardSlot, tile: Image.Image) -> None:
    box = slot.box
    rotated = tile.rotate(90, expand=True).resize((box.width, box.height))
    image.paste(rotated, (box.left, box.top))


def _draw_horizontal_riichi_stick(image: Image.Image, bbox: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, fill=(236, 236, 232))
    left, top, right, bottom = bbox
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill=(228, 18, 18))


def _draw_vertical_riichi_stick(image: Image.Image, bbox: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, fill=(236, 236, 232))
    left, top, right, bottom = bbox
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill=(228, 18, 18))
