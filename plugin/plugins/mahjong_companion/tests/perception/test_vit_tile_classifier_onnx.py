"""Tests for the ONNX backend of the ViT tile classifier.

Coverage:

* API surface — missing-model behaviour, env-var resolution.
* Preprocessor JSON parsing — accepts the shapes produced by
  ``scripts/export_vit_to_onnx.py``.
* Smoke test (skipped without artifacts) — runs a single forward when
  ``MAHJONG_COMPANION_VIT_ONNX_DIR`` (or the default
  ``data/models/vit_tile_classifier``) contains a real export.
* Consistency test (skipped without both) — when the transformers
  backend and the ONNX backend are both available, top-1 predictions
  must agree on a fixed set of crops.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from plugin.plugins.mahjong_companion.perception import vit_tile_classifier_onnx as onnx_backend
from plugin.plugins.mahjong_companion.perception.vit_tile_classifier_onnx import (
    VitOnnxClassifierUnavailable,
    _load_preprocessor,
    classify_tile_crops_onnx,
    vit_onnx_available,
)


def _real_export_dir() -> Path | None:
    candidate_env = onnx_backend.os.environ.get("MAHJONG_COMPANION_VIT_ONNX_DIR")
    if candidate_env:
        path = Path(candidate_env).expanduser().resolve()
        return path if (path / "model.onnx").exists() else None
    plugin_root = Path(onnx_backend.__file__).resolve().parent.parent
    default_path = (plugin_root / onnx_backend.DEFAULT_MODEL_SUBDIR).resolve()
    return default_path if (default_path / "model.onnx").exists() else None


def _onnxruntime_installed() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _transformers_installed() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.unit
def test_classify_raises_when_model_dir_missing(tmp_path: Path) -> None:
    missing_dir = tmp_path / "no_such"
    crops = [Image.new("RGB", (32, 32), (128, 128, 128))]
    with pytest.raises(VitOnnxClassifierUnavailable):
        classify_tile_crops_onnx(crops, model_dir=missing_dir)


@pytest.mark.unit
def test_vit_onnx_available_returns_false_when_missing(tmp_path: Path) -> None:
    assert vit_onnx_available(model_dir=tmp_path / "no_such") is False


@pytest.mark.unit
def test_classify_returns_empty_list_for_empty_input(tmp_path: Path) -> None:
    # Empty input must short-circuit before model loading, so a missing
    # model dir must not raise here.
    assert classify_tile_crops_onnx([], model_dir=tmp_path / "no_such") == []


@pytest.mark.unit
def test_load_preprocessor_dict_size(tmp_path: Path) -> None:
    payload = {
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": {"height": 224, "width": 224},
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "do_normalize": True,
    }
    target = tmp_path / "preprocessor.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    pre = _load_preprocessor(target)
    assert pre.height == 224
    assert pre.width == 224
    assert pre.mean == (0.5, 0.5, 0.5)
    assert pre.std == (0.5, 0.5, 0.5)
    assert pre.do_rescale is True
    assert pre.do_normalize is True


@pytest.mark.unit
def test_load_preprocessor_shortest_edge_fallback(tmp_path: Path) -> None:
    payload = {
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "size": {"shortest_edge": 192},
    }
    target = tmp_path / "preprocessor.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    pre = _load_preprocessor(target)
    assert pre.height == 192
    assert pre.width == 192


@pytest.mark.unit
def test_load_preprocessor_rejects_invalid_mean(tmp_path: Path) -> None:
    payload = {"image_mean": [0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "size": 224}
    target = tmp_path / "preprocessor.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VitOnnxClassifierUnavailable):
        _load_preprocessor(target)


# ---------------------------------------------------------------------------
# Smoke / consistency tests (skipped without real artifacts)


@pytest.mark.unit
def test_real_export_smoke() -> None:
    export_dir = _real_export_dir()
    if export_dir is None:
        pytest.skip("No ONNX export found; run scripts/export_vit_to_onnx.py first.")
    if not _onnxruntime_installed():
        pytest.skip("onnxruntime not installed in this environment.")
    crops = [Image.new("RGB", (60, 80), (200, 180, 160))]
    predictions = classify_tile_crops_onnx(crops, model_dir=export_dir, top_k=3)
    assert len(predictions) == 1
    pred = predictions[0]
    # Lightweight exports may include an ``empty`` class.  A synthetic colour
    # blob can legitimately land there, which normalizes to ``None``.
    if pred is not None:
        assert pred.tile
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.top_k) <= 3


@pytest.mark.unit
def test_backend_consistency_against_transformers() -> None:
    export_dir = _real_export_dir()
    if export_dir is None:
        pytest.skip("No ONNX export found.")
    if not _onnxruntime_installed():
        pytest.skip("onnxruntime not installed.")
    if not _transformers_installed():
        pytest.skip("transformers not installed; cross-backend check disabled.")
    from plugin.plugins.mahjong_companion.perception import vit_tile_classifier as transformers_backend
    from plugin.plugins.mahjong_companion.perception.vit_tile_classifier import (
        VitTileClassifierUnavailable,
    )

    crops = [
        Image.new("RGB", (60, 80), color)
        for color in ((220, 200, 180), (180, 60, 40), (40, 80, 200))
    ]
    onnx_predictions = classify_tile_crops_onnx(crops, model_dir=export_dir, top_k=1)
    try:
        transformer_predictions = transformers_backend.classify_tile_crops(crops, top_k=1)
    except VitTileClassifierUnavailable:
        pytest.skip("transformers backend could not load the HF model.")

    matches = 0
    total = 0
    for onnx_pred, hf_pred in zip(onnx_predictions, transformer_predictions, strict=True):
        if onnx_pred is None or hf_pred is None:
            continue
        total += 1
        if onnx_pred.tile == hf_pred.tile:
            matches += 1
    if total == 0:
        pytest.skip("No prediction overlap to compare.")
    # Allow at most one disagreement on a fixed colour-blob set; tighten
    # later once we run on real tile crops in an evaluation harness.
    assert matches >= total - 1, (
        f"ONNX vs transformers disagreement: {matches}/{total} matches; "
        f"onnx={[p.tile if p else None for p in onnx_predictions]} "
        f"hf={[p.tile if p else None for p in transformer_predictions]}"
    )
