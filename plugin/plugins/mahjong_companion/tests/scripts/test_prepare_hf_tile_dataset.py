from __future__ import annotations

from pathlib import Path

from PIL import Image

from plugin.plugins.mahjong_companion.scripts.prepare_hf_tile_dataset import (
    normalize_hf_label,
    prepare_dataset,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 16), color).save(path)


def test_normalize_hf_label_maps_suits_and_honors() -> None:
    assert normalize_hf_label("1n") == "1m"
    assert normalize_hf_label("9b") == "9s"
    assert normalize_hf_label("5p") == "5p"
    assert normalize_hf_label("ew") == "1z"
    assert normalize_hf_label("gd") == "6z"
    assert normalize_hf_label("rd") == "7z"
    assert normalize_hf_label("flower") == ""


def test_prepare_dataset_relabels_imagefolder(tmp_path: Path) -> None:
    source = tmp_path / "hf"
    _write_image(source / "dataset" / "train" / "1n" / "a.png", (200, 180, 160))
    _write_image(source / "dataset" / "train" / "1b" / "a.png", (20, 120, 80))
    _write_image(source / "dataset" / "test" / "ew" / "a.png", (80, 80, 80))

    output = tmp_path / "out"
    labels = prepare_dataset(source, output, overwrite=True)

    assert labels == ["1m", "1s", "1z"]
    assert (output / "train" / "1m" / "00000.png").exists()
    assert (output / "train" / "1s" / "00000.png").exists()
    assert (output / "val" / "1z" / "00000.png").exists()
    assert (output / "labels.txt").read_text(encoding="utf-8") == "1m\n1s\n1z"
