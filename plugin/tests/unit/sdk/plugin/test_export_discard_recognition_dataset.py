from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.perception.discard_layout import build_discard_layout
from plugin.plugins.mahjong_companion.scripts.export_discard_recognition_dataset import (
    export_discard_recognition_dataset,
    main as export_discard_recognition_dataset_main,
)


def test_export_discard_recognition_dataset_writes_jsonl_and_copies_images(tmp_path: Path) -> None:
    case_dir = tmp_path / "eval" / "discard_recognition" / "1280x720" / "basic"
    image_path, label_path = _write_label_case(case_dir, label_scope="partial")
    output_dir = tmp_path / "dataset"

    result = export_discard_recognition_dataset(label_root=label_path, output_dir=output_dir)

    assert result["ok"] is True
    assert result["image_count"] == 1
    assert result["detection_count"] == 1
    assert result["label_scope_counts"] == {"full": 0, "partial": 1}
    assert Path(result["annotations_path"]).exists()
    record = json.loads(Path(result["annotations_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert record["label_scope"] == "partial"
    assert record["source_image_path"] == str(image_path)
    assert record["detections"][0]["tile"] == "1m"
    assert record["detections"][0]["player"] == "self"
    assert record["detections"][0]["quad"]
    assert (output_dir / record["image"]).exists()


def test_export_discard_recognition_dataset_can_refine_label_quads(tmp_path: Path) -> None:
    case_dir = tmp_path / "eval" / "discard_recognition" / "1280x720" / "slanted"
    _, label_path = _write_label_case(case_dir, slanted=True)
    output_dir = tmp_path / "dataset"

    result = export_discard_recognition_dataset(
        label_root=label_path,
        output_dir=output_dir,
        refine_quads=True,
    )

    record = json.loads(Path(result["annotations_path"]).read_text(encoding="utf-8").splitlines()[0])
    detection = record["detections"][0]
    assert detection["quad_source"] == "refined_tile_surface"
    assert detection["quad_confidence"] >= 0.34
    assert detection["quad"] != _slot_quad(1280, 720)


def test_export_discard_recognition_dataset_cli(tmp_path: Path, capsys) -> None:
    case_dir = tmp_path / "eval" / "discard_recognition" / "1280x720" / "basic"
    _, label_path = _write_label_case(case_dir)
    output_dir = tmp_path / "dataset"

    exit_code = export_discard_recognition_dataset_main(
        [
            "--label-root",
            str(label_path),
            "--output-dir",
            str(output_dir),
            "--pretty",
        ],
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["detection_count"] == 1


def _write_label_case(case_dir: Path, *, slanted: bool = False, label_scope: str = "full") -> tuple[Path, Path]:
    case_dir.mkdir(parents=True)
    image_path = case_dir / "frame.png"
    image = Image.new("RGB", (1280, 720), color=(28, 58, 104))
    slot = build_discard_layout(*image.size)["self"][0]
    draw = ImageDraw.Draw(image)
    if slanted:
        draw.polygon(
            [
                (slot.box.left - 5, slot.box.top - 4),
                (slot.box.left + 1, slot.box.bottom + 6),
                (slot.box.right + 7, slot.box.bottom - 2),
                (slot.box.right + 1, slot.box.top - 9),
            ],
            fill=(238, 236, 220),
        )
    else:
        draw.rectangle((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom), fill=(238, 236, 220))
    draw.rectangle(
        (
            slot.box.left + 9,
            slot.box.top + 10,
            slot.box.right - 9,
            slot.box.bottom - 16,
        ),
        fill=(210, 52, 58),
    )
    image.save(image_path)

    label_path = case_dir / "frame.label.json"
    label_path.write_text(
        json.dumps(
            {
                "case_id": case_dir.name,
                "label_scope": label_scope,
                "image": {"path": "frame.png"},
                "scene": "in_match",
                "discard_piles": {
                    "self": [
                        {
                            "tile": "1m",
                            "turn_index": 1,
                            "bbox": slot.bbox,
                            "quad": [[x, y] for x, y in slot.corners],
                            "orientation": "bottom",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return image_path, label_path


def _slot_quad(width: int, height: int) -> list[list[int]]:
    slot = build_discard_layout(width, height)["self"][0]
    return [[x, y] for x, y in slot.corners]
