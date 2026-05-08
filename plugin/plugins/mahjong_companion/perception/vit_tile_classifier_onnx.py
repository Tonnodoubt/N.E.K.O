"""ONNX runtime path for the ViT mahjong tile classifier.

Mirrors the public surface of :mod:`vit_tile_classifier` (transformers
backend) so call sites can switch backends without behaviour changes.
This module deliberately depends only on ``onnxruntime``, ``numpy`` and
``Pillow`` so it stays usable on machines without ``torch`` /
``transformers`` installed.

Model artifacts (produced by ``scripts/export_vit_to_onnx.py``)::

    <model_dir>/
        model.onnx
        preprocessor.json
        labels.json
        metadata.json   # optional, informational only

Resolution order for ``model_dir``:

    1. ``MAHJONG_COMPANION_VIT_ONNX_DIR`` env var
    2. ``data/models/vit_tile_classifier`` under the plugin root

If the directory or required files are missing, the loader raises
:class:`VitOnnxClassifierUnavailable` so callers can fall back to the
transformers backend (or template matching).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .vit_tile_classifier import VitTilePrediction, _LABEL_MAP  # noqa: F401  (reuse label map)
from ..tile_labels import normalize_tile


DEFAULT_TOP_K = 3
DEFAULT_MODEL_SUBDIR = Path("data") / "models" / "vit_tile_classifier"
REQUIRED_FILES = ("model.onnx", "preprocessor.json", "labels.json")
ENV_PROVIDERS = "MAHJONG_COMPANION_VIT_ONNX_PROVIDERS"


class VitOnnxClassifierUnavailable(RuntimeError):
    """Raised when the ONNX backend cannot be loaded."""


@dataclass(frozen=True)
class _Preprocessor:
    height: int
    width: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rescale_factor: float
    do_rescale: bool
    do_normalize: bool


@dataclass(frozen=True)
class _LoadedModel:
    session: Any
    input_name: str
    preprocessor: _Preprocessor
    labels: dict[int, str]


_MODEL_CACHE: dict[Path, _LoadedModel] = {}
_MODEL_FAILURES: dict[Path, str] = {}
_MODEL_LOCK = Lock()


def classify_tile_crops_onnx(
    crops: list[Image.Image],
    *,
    model_dir: str | os.PathLike[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[VitTilePrediction | None]:
    """Run the ONNX backend on a list of PIL crops.

    Mirrors :func:`vit_tile_classifier.classify_tile_crops`. Returns
    ``None`` for any crop that fails inference.
    """
    if not crops:
        return []
    resolved = _resolve_model_dir(model_dir)
    loaded = _load_model(resolved)
    clean_top_k = max(1, int(top_k or DEFAULT_TOP_K))
    images = [crop.convert("RGB") for crop in crops]
    batch = np.stack([_preprocess(image, loaded.preprocessor) for image in images], axis=0)
    raw = loaded.session.run(None, {loaded.input_name: batch})
    if not raw:
        return [None] * len(images)
    logits = np.asarray(raw[0], dtype=np.float32)
    probs = _softmax(logits)
    return [_prediction_from_probs(row, loaded.labels, top_k=clean_top_k) for row in probs]


def vit_onnx_available(*, model_dir: str | os.PathLike[str] | None = None) -> bool:
    """Cheap probe so callers can decide between ONNX and transformer paths."""
    try:
        resolved = _resolve_model_dir(model_dir)
        _load_model(resolved)
    except VitOnnxClassifierUnavailable:
        return False
    return True


def fetch_model_from_url(
    base_url: str,
    *,
    model_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Download required artifacts from ``<base_url>/<filename>``.

    Used at first-run when the local model dir is empty. Caller is
    responsible for choosing a trusted host (e.g. an HF release URL).
    """
    target = _resolve_model_dir(model_dir, ensure_exists=False)
    target.mkdir(parents=True, exist_ok=True)
    base = base_url.rstrip("/")
    for filename in REQUIRED_FILES:
        dest = target / filename
        if dest.exists() and not overwrite:
            continue
        url = f"{base}/{filename}"
        try:
            with urlopen(url) as response, dest.open("wb") as out:
                out.write(response.read())
        except URLError as exc:
            raise VitOnnxClassifierUnavailable(f"Failed to download {url}: {exc}") from exc
    return target


# ---------------------------------------------------------------------------
# Internal helpers


def resolve_onnx_providers(
    *,
    available: tuple[str, ...] | None = None,
    platform: str | None = None,
    env_value: str | None = None,
) -> tuple[str, ...]:
    """Pick the ONNX execution provider list to hand to ``InferenceSession``.

    Resolution order:

      1. The env var ``MAHJONG_COMPANION_VIT_ONNX_PROVIDERS`` (comma-
         separated). Takes precedence so users can pin behaviour even on
         platforms with poor auto-detection. Empty entries are dropped.
      2. A platform-specific default — CoreML on macOS, DirectML on
         Windows, CUDA on Linux — followed by ``CPUExecutionProvider``
         as a guaranteed fallback.

    If ``available`` is provided (callers usually pass
    ``ort.get_available_providers()``), entries not listed there are
    filtered out. ``CPUExecutionProvider`` is always appended last, so
    even an empty result still yields a valid list.
    """
    raw = env_value if env_value is not None else os.environ.get(ENV_PROVIDERS, "")
    cleaned = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
    if not cleaned:
        cleaned = list(_default_providers_for_platform(platform or sys.platform))

    if available is not None:
        available_set = {item for item in available if item}
        filtered = [item for item in cleaned if item in available_set]
        cleaned = filtered or []

    cpu = "CPUExecutionProvider"
    if cpu not in cleaned:
        cleaned.append(cpu)
    return tuple(cleaned)


def _default_providers_for_platform(platform: str) -> tuple[str, ...]:
    p = platform.lower()
    if p == "darwin":
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    if p.startswith("win"):
        return ("DmlExecutionProvider", "CPUExecutionProvider")
    return ("CUDAExecutionProvider", "CPUExecutionProvider")


def _resolve_model_dir(
    explicit: str | os.PathLike[str] | None,
    *,
    ensure_exists: bool = True,
) -> Path:
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    else:
        env_dir = os.environ.get("MAHJONG_COMPANION_VIT_ONNX_DIR")
        if env_dir:
            candidate = Path(env_dir).expanduser().resolve()
        else:
            plugin_root = Path(__file__).resolve().parent.parent
            candidate = (plugin_root / DEFAULT_MODEL_SUBDIR).resolve()
    if ensure_exists and not candidate.exists():
        raise VitOnnxClassifierUnavailable(
            f"ONNX model dir not found: {candidate}. "
            "Run scripts/export_vit_to_onnx.py or set "
            "MAHJONG_COMPANION_VIT_ONNX_DIR."
        )
    return candidate


def _load_model(model_dir: Path) -> _LoadedModel:
    failure = _MODEL_FAILURES.get(model_dir)
    if failure:
        raise VitOnnxClassifierUnavailable(failure)
    cached = _MODEL_CACHE.get(model_dir)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_dir)
        if cached is not None:
            return cached
        try:
            loaded = _build_loaded_model(model_dir)
        except VitOnnxClassifierUnavailable as exc:
            _MODEL_FAILURES[model_dir] = str(exc)
            raise
        _MODEL_CACHE[model_dir] = loaded
        return loaded


def _build_loaded_model(model_dir: Path) -> _LoadedModel:
    for filename in REQUIRED_FILES:
        if not (model_dir / filename).exists():
            raise VitOnnxClassifierUnavailable(
                f"Missing required file in {model_dir}: {filename}"
            )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise VitOnnxClassifierUnavailable(f"onnxruntime not installed: {exc}") from exc
    providers = resolve_onnx_providers(available=tuple(ort.get_available_providers()))
    try:
        session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=list(providers),
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        raise VitOnnxClassifierUnavailable(f"Failed to load ONNX session: {exc}") from exc
    inputs = session.get_inputs()
    if not inputs:
        raise VitOnnxClassifierUnavailable("ONNX session has no inputs")
    preprocessor = _load_preprocessor(model_dir / "preprocessor.json")
    labels = _load_labels(model_dir / "labels.json")
    return _LoadedModel(
        session=session,
        input_name=inputs[0].name,
        preprocessor=preprocessor,
        labels=labels,
    )


def _load_preprocessor(path: Path) -> _Preprocessor:
    payload = json.loads(path.read_text(encoding="utf-8"))
    size = payload.get("size") or {}
    if isinstance(size, dict):
        height = int(size.get("height") or size.get("shortest_edge") or 224)
        width = int(size.get("width") or size.get("shortest_edge") or height)
    elif isinstance(size, int):
        height = width = int(size)
    else:
        height = width = 224
    mean = tuple(float(v) for v in payload.get("image_mean") or (0.5, 0.5, 0.5))
    std = tuple(float(v) for v in payload.get("image_std") or (0.5, 0.5, 0.5))
    if len(mean) != 3 or len(std) != 3:
        raise VitOnnxClassifierUnavailable(f"Invalid mean/std in {path}")
    rescale_factor = float(payload.get("rescale_factor") or (1.0 / 255.0))
    do_rescale = bool(payload.get("do_rescale", True))
    do_normalize = bool(payload.get("do_normalize", True))
    return _Preprocessor(
        height=height,
        width=width,
        mean=(mean[0], mean[1], mean[2]),
        std=(std[0], std[1], std[2]),
        rescale_factor=rescale_factor,
        do_rescale=do_rescale,
        do_normalize=do_normalize,
    )


def _load_labels(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VitOnnxClassifierUnavailable(f"labels.json must be a mapping, got {type(payload)}")
    cleaned: dict[int, str] = {}
    for key, value in payload.items():
        try:
            cleaned[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    if not cleaned:
        raise VitOnnxClassifierUnavailable(f"No usable labels in {path}")
    return cleaned


def _preprocess(image: Image.Image, preprocessor: _Preprocessor) -> np.ndarray:
    resized = image.resize((preprocessor.width, preprocessor.height), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if preprocessor.do_rescale:
        array = array * preprocessor.rescale_factor
    if preprocessor.do_normalize:
        mean = np.asarray(preprocessor.mean, dtype=np.float32)
        std = np.asarray(preprocessor.std, dtype=np.float32)
        array = (array - mean) / std
    # HWC -> CHW
    return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _prediction_from_probs(
    probs: np.ndarray,
    labels: dict[int, str],
    *,
    top_k: int,
) -> VitTilePrediction | None:
    if probs.size == 0:
        return None
    order = np.argsort(probs)[::-1][:top_k]
    top_entries: list[dict[str, Any]] = []
    for idx in order:
        label = labels.get(int(idx))
        if label is None:
            continue
        tile = _tile_from_label(label)
        top_entries.append(
            {
                "label": label,
                "tile": tile,
                "score": round(float(probs[idx]), 4),
            }
        )
    if not top_entries:
        return None
    top = top_entries[0]
    tile = str(top.get("tile", ""))
    if not tile:
        return None
    return VitTilePrediction(
        tile=tile,
        label=str(top.get("label", "")),
        confidence=float(top.get("score", 0.0) or 0.0),
        top_k=top_entries,
    )


def _tile_from_label(label: str) -> str:
    lower = str(label or "").strip().lower()
    mapped = _LABEL_MAP.get(lower, lower)
    return normalize_tile(mapped)
