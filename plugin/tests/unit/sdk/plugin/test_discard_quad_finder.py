from __future__ import annotations

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.perception.discard_quad_finder import refine_discard_slot_quad
from plugin.plugins.mahjong_companion.perception.discard_parser import crop_discard_slot


def test_refine_discard_slot_quad_fits_visible_slanted_tile_face() -> None:
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    slot = build_discard_layout(*image.size)["self"][0]
    draw = ImageDraw.Draw(image)
    polygon = [
        (slot.box.left - 5, slot.box.top - 4),
        (slot.box.left + 1, slot.box.bottom + 6),
        (slot.box.right + 7, slot.box.bottom - 2),
        (slot.box.right + 1, slot.box.top - 9),
    ]
    draw.polygon(polygon, fill=(238, 236, 220))
    draw.rectangle(
        (
            slot.box.left + 9,
            slot.box.top + 10,
            slot.box.right - 9,
            slot.box.bottom - 16,
        ),
        fill=(210, 52, 58),
    )

    refinement = refine_discard_slot_quad(image, slot)

    assert refinement is not None
    assert refinement.bbox[0] < slot.box.left
    assert refinement.bbox[3] > slot.box.bottom
    assert refinement.quad != slot.corners
    assert refinement.confidence >= 0.34

    crop = crop_discard_slot(image, slot)
    assert crop.size[0] > 0
    assert crop.size[1] > 0
