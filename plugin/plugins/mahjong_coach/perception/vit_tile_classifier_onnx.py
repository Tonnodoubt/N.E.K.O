from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_SUBDIR = Path("data") / "models" / "vit_tile_classifier"
DEFAULT_TOP_K = 3
ENV_MODEL_DIR = "MAHJONG_COACH_TILE_ONNX_DIR"
ENV_PROVIDERS = "MAHJONG_COACH_TILE_ONNX_PROVIDERS"
REQUIRED_FILES = ("model.onnx", "preprocessor.json", "labels.json")


class OnnxTileClassifierUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class OnnxTilePrediction:
    tile: str
    label: str
    confidence: float
    top_k: list[dict[str, Any]]


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
) -> list[OnnxTilePrediction | None]:
    if not crops:
        return []
    loaded = _load_model(_resolve_model_dir(model_dir))
    images = [crop.convert("RGB") for crop in crops]
    batch = np.stack([_preprocess(image, loaded.preprocessor) for image in images], axis=0)
    raw = loaded.session.run(None, {loaded.input_name: batch})
    if not raw:
        return [None] * len(crops)
    logits = np.asarray(raw[0], dtype=np.float32)
    probabilities = _softmax(logits)
    clean_top_k = max(1, int(top_k or DEFAULT_TOP_K))
    return [_prediction_from_probs(row, loaded.labels, top_k=clean_top_k) for row in probabilities]


def onnx_tile_classifier_available(*, model_dir: str | os.PathLike[str] | None = None) -> bool:
    try:
        _load_model(_resolve_model_dir(model_dir))
    except OnnxTileClassifierUnavailable:
        return False
    return True


def resolve_onnx_providers(
    *,
    available: tuple[str, ...] | None = None,
    platform: str | None = None,
    env_value: str | None = None,
) -> tuple[str, ...]:
    raw = env_value if env_value is not None else os.environ.get(ENV_PROVIDERS, "")
    providers = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
    if not providers:
        providers = list(_default_providers_for_platform(platform or sys.platform))

    if available is not None:
        available_set = {item for item in available if item}
        providers = [item for item in providers if item in available_set]

    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return tuple(providers)


def _default_providers_for_platform(platform: str) -> tuple[str, ...]:
    lowered = platform.lower()
    if lowered == "darwin":
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    if lowered.startswith("win"):
        return ("DmlExecutionProvider", "CPUExecutionProvider")
    return ("CUDAExecutionProvider", "CPUExecutionProvider")


def _resolve_model_dir(explicit: str | os.PathLike[str] | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_dir = os.environ.get(ENV_MODEL_DIR)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    plugin_root = Path(__file__).resolve().parent.parent
    return (plugin_root / DEFAULT_MODEL_SUBDIR).resolve()


def _load_model(model_dir: Path) -> _LoadedModel:
    failure = _MODEL_FAILURES.get(model_dir)
    if failure:
        raise OnnxTileClassifierUnavailable(failure)
    cached = _MODEL_CACHE.get(model_dir)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_dir)
        if cached is not None:
            return cached
        try:
            loaded = _build_loaded_model(model_dir)
        except OnnxTileClassifierUnavailable as exc:
            _MODEL_FAILURES[model_dir] = str(exc)
            raise
        _MODEL_CACHE[model_dir] = loaded
        return loaded


def _build_loaded_model(model_dir: Path) -> _LoadedModel:
    for filename in REQUIRED_FILES:
        if not (model_dir / filename).exists():
            raise OnnxTileClassifierUnavailable(f"Missing required ONNX artifact: {model_dir / filename}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise OnnxTileClassifierUnavailable(f"onnxruntime not installed: {exc}") from exc

    providers = resolve_onnx_providers(available=tuple(ort.get_available_providers()))
    try:
        session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=list(providers))
    except Exception as exc:
        raise OnnxTileClassifierUnavailable(f"Failed to load ONNX session: {exc}") from exc

    inputs = session.get_inputs()
    if not inputs:
        raise OnnxTileClassifierUnavailable("ONNX session has no inputs")
    return _LoadedModel(
        session=session,
        input_name=inputs[0].name,
        preprocessor=_load_preprocessor(model_dir / "preprocessor.json"),
        labels=_load_labels(model_dir / "labels.json"),
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
    mean = tuple(float(item) for item in payload.get("image_mean") or (0.5, 0.5, 0.5))
    std = tuple(float(item) for item in payload.get("image_std") or (0.5, 0.5, 0.5))
    if len(mean) != 3 or len(std) != 3:
        raise OnnxTileClassifierUnavailable(f"Invalid preprocessor mean/std: {path}")
    return _Preprocessor(
        height=height,
        width=width,
        mean=(mean[0], mean[1], mean[2]),
        std=(std[0], std[1], std[2]),
        rescale_factor=float(payload.get("rescale_factor") or (1.0 / 255.0)),
        do_rescale=bool(payload.get("do_rescale", True)),
        do_normalize=bool(payload.get("do_normalize", True)),
    )


def _load_labels(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OnnxTileClassifierUnavailable(f"labels.json must be a mapping: {path}")
    labels: dict[int, str] = {}
    for key, value in payload.items():
        try:
            labels[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    if not labels:
        raise OnnxTileClassifierUnavailable(f"No usable labels in {path}")
    return labels


def _preprocess(image: Image.Image, preprocessor: _Preprocessor) -> np.ndarray:
    resized = image.resize((preprocessor.width, preprocessor.height), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if preprocessor.do_rescale:
        array = array * preprocessor.rescale_factor
    if preprocessor.do_normalize:
        mean = np.asarray(preprocessor.mean, dtype=np.float32)
        std = np.asarray(preprocessor.std, dtype=np.float32)
        array = (array - mean) / std
    return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _prediction_from_probs(
    probabilities: np.ndarray,
    labels: dict[int, str],
    *,
    top_k: int,
) -> OnnxTilePrediction | None:
    if probabilities.size == 0:
        return None
    top_entries: list[dict[str, Any]] = []
    for index in np.argsort(probabilities)[::-1][:top_k]:
        label = labels.get(int(index))
        if label is None:
            continue
        tile = _tile_from_label(label)
        if not tile:
            continue
        top_entries.append({"label": label, "tile": tile, "score": round(float(probabilities[index]), 4)})
    if not top_entries:
        return None
    top = top_entries[0]
    return OnnxTilePrediction(
        tile=str(top["tile"]),
        label=str(top["label"]),
        confidence=float(top["score"]),
        top_k=top_entries,
    )


def _tile_from_label(label: str) -> str:
    value = str(label or "").strip().lower()
    if value == "empty":
        return "empty"
    if len(value) == 2 and value[0] in "123456789" and value[1] in "mpsz":
        return value
    return ""
