from __future__ import annotations

import json

from PIL import Image

from plugin.plugins.mahjong_companion.perception.external_discard_recognizer import (
    ENV_MODEL_JSON_DIR,
    ENV_MODEL_MANUAL_LABELS,
    load_external_discard_result,
)


def test_load_external_discard_result_reads_model_river_json_dir(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (320, 240), (35, 70, 110))
    image.save(image_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "frame.json").write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "x": 155,
                        "y": 145,
                        "width": 50,
                        "height": 50,
                        "confidence": 0.8,
                        "class": "8D",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_MODEL_JSON_DIR, str(cache))

    piles, hints = load_external_discard_result(image_path, image)

    assert hints["external_discard_recognizer_source"] == "model_river_adapter"
    assert hints["discard_parser_source"] == "model_river_adapter"
    assert piles["self"][0]["tile"] == "8p"


def test_load_external_discard_result_passes_manual_labels(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    image = Image.new("RGB", (320, 240), (35, 70, 110))
    cache = tmp_path / "cache"
    cache.mkdir()
    labels = tmp_path / "labels.json"
    labels.write_text("[]", encoding="utf-8")
    seen = {}

    def fake_parse(_image, _image_path, *, config):
        seen["manual_labels_path"] = config.manual_labels_path

        class _Result:
            discard_piles = {}
            analysis_hints = {"discard_parser_source": "model_river_adapter"}

        return _Result()

    monkeypatch.setenv(ENV_MODEL_JSON_DIR, str(cache))
    monkeypatch.setenv(ENV_MODEL_MANUAL_LABELS, str(labels))
    monkeypatch.setattr(
        "plugin.plugins.mahjong_companion.perception.external_discard_recognizer.parse_model_river_from_json",
        fake_parse,
    )

    load_external_discard_result(image_path, image)

    assert seen["manual_labels_path"] == labels
