from __future__ import annotations

from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout


PLAYERS = {"self", "left_opponent", "top_opponent", "right_opponent"}


def test_discard_layout_builds_four_player_rivers() -> None:
    layout = build_discard_layout(1920, 1080)

    assert set(layout) == PLAYERS
    assert all(len(slots) == 18 for slots in layout.values())
    assert len({slot.slot_id for slots in layout.values() for slot in slots}) == 72

    for player, slots in layout.items():
        assert [slot.turn_index for slot in slots] == list(range(1, 19))
        assert all(slot.player == player for slot in slots)


def test_discard_layout_uses_stable_1920x1080_anchor_boxes() -> None:
    layout = build_discard_layout(1920, 1080)

    assert layout["self"][0].to_dict()["bbox"] == [762, 542, 820, 612]
    assert layout["self"][-1].to_dict()["bbox"] == [1082, 682, 1140, 752]
    assert layout["top_opponent"][0].to_dict()["bbox"] == [802, 242, 860, 312]
    assert layout["top_opponent"][6].to_dict()["bbox"] == [802, 172, 860, 242]
    assert layout["left_opponent"][0].to_dict()["bbox"] == [624, 290, 708, 348]
    assert layout["left_opponent"][5].to_dict()["bbox"] == [624, 600, 708, 658]
    assert layout["right_opponent"][0].to_dict()["bbox"] == [1148, 290, 1232, 348]
    assert layout["right_opponent"][5].to_dict()["bbox"] == [1148, 600, 1232, 658]
    assert layout["right_opponent"][0].to_dict()["quad"] == [
        [1148, 290],
        [1148, 348],
        [1232, 348],
        [1232, 290],
    ]

    assert layout["self"][0].orientation == "bottom"
    assert layout["top_opponent"][0].orientation == "top"
    assert layout["left_opponent"][0].orientation == "left"
    assert layout["right_opponent"][0].orientation == "right"


def test_discard_layout_boxes_stay_inside_screen_bounds() -> None:
    width = 1920
    height = 1080
    layout = build_discard_layout(width, height)

    for slots in layout.values():
        for slot in slots:
            assert 0 <= slot.box.left < width
            assert 0 <= slot.box.top < height
            assert 0 < slot.box.right <= width
            assert 0 < slot.box.bottom <= height


def test_discard_layout_scales_to_smaller_16_9_capture() -> None:
    layout = build_discard_layout(1280, 720)
    first_self = layout["self"][0]
    first_right = layout["right_opponent"][0]

    assert first_self.to_dict()["bbox"] == [508, 361, 547, 408]
    assert first_right.box.left > 1280 // 2
    assert layout["top_opponent"][0].box.top < 720 // 2
    assert layout["self"][0].box.top > 720 // 2
