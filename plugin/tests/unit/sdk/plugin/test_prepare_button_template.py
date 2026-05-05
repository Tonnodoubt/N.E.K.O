from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.mahjong_companion.scripts.prepare_button_template import (
    main as prepare_button_template_main,
    prepare_button_template,
)


def _write_source_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1920, 1080), (30, 65, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((800, 740, 980, 820), fill=(15, 180, 190))
    draw.rectangle((1010, 740, 1200, 820), fill=(90, 80, 75))
    image.save(path)


def test_prepare_button_template_writes_template_meta_and_fixture(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    template_dir = tmp_path / "templates"
    eval_dir = tmp_path / "eval"
    _write_source_image(source)

    result = prepare_button_template(
        image_path=source,
        button_type="pon",
        bbox=(800, 740, 980, 820),
        template_dir=template_dir,
        match_threshold=0.91,
        padding=5,
        write_fixture=True,
        eval_dir=eval_dir,
        fixture_case_id="pon-skip-source",
        fixture_buttons=[("skip", (1010, 740, 1200, 820))],
    )

    assert result["ok"] is True
    assert result["template_bbox"] == [795, 735, 985, 825]
    template_path = Path(result["template_path"])
    assert template_path.exists()
    assert Image.open(template_path).size == (190, 90)

    meta = json.loads((template_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["templates"]["pon"]["button_type"] == "pon"
    assert meta["templates"]["pon"]["file"] == "1920x1080/pon.png"
    assert meta["templates"]["pon"]["resolution"] == [1920, 1080]
    assert meta["templates"]["pon"]["match_threshold"] == 0.91

    label_path = eval_dir / "button_localization" / "1920x1080" / "pon-skip-source" / "frame.label.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))
    assert label["image"]["path"] == "frame.png"
    assert label["buttons"] == [
        {"button_type": "pon", "bbox": [795, 735, 985, 825]},
        {"button_type": "skip", "bbox": [1010, 740, 1200, 820]},
    ]


def test_prepare_button_template_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    template_dir = tmp_path / "templates"
    _write_source_image(source)
    prepare_button_template(
        image_path=source,
        button_type="pon",
        bbox=(800, 740, 980, 820),
        template_dir=template_dir,
    )

    with pytest.raises(FileExistsError):
        prepare_button_template(
            image_path=source,
            button_type="pon",
            bbox=(800, 740, 980, 820),
            template_dir=template_dir,
        )


def test_prepare_button_template_cli_writes_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source.png"
    template_dir = tmp_path / "templates"
    _write_source_image(source)

    exit_code = prepare_button_template_main(
        [
            "--image",
            str(source),
            "--button-type",
            "riichi",
            "--bbox",
            "800,740,980,820",
            "--template-dir",
            str(template_dir),
            "--match-threshold",
            "0.92",
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["button_type"] == "riichi"
    assert payload["template_path"].endswith("templates/1920x1080/riichi.png")
    assert (template_dir / "meta.json").exists()
