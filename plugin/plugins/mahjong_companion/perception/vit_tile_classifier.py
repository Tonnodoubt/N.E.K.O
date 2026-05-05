from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Lock
from typing import Any

from PIL import Image

from ..tile_labels import normalize_tile


DEFAULT_VIT_MODEL = "krmin/mahjong_soul_vision"
DEFAULT_TOP_K = 3

_LABEL_MAP = {
    **{f"{index}n": f"{index}m" for index in range(1, 10)},
    **{f"{index}m": f"{index}m" for index in range(1, 10)},
    **{f"{index}p": f"{index}p" for index in range(1, 10)},
    **{f"{index}b": f"{index}s" for index in range(1, 10)},
    **{f"{index}s": f"{index}s" for index in range(1, 10)},
    "ew": "1z",
    "east": "1z",
    "sw": "2z",
    "south": "2z",
    "ww": "3z",
    "west": "3z",
    "nw": "4z",
    "north": "4z",
    "wd": "5z",
    "white": "5z",
    "gd": "6z",
    "green": "6z",
    "rd": "7z",
    "red": "7z",
}

_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}
_PIPELINE_FAILURES: dict[tuple[str, str], str] = {}
_PIPELINE_LOCK = Lock()
_ACCELERATOR_AVAILABLE: bool | None = None


class VitTileClassifierUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class VitTilePrediction:
    tile: str
    label: str
    confidence: float
    top_k: list[dict[str, Any]]

    def to_detection_fields(self) -> dict[str, Any]:
        return {
            "candidate_tile": self.tile,
            "confidence": self.confidence,
            "vit_label": self.label,
            "vit_top_k": list(self.top_k),
        }


def classify_tile_crops(
    crops: list[Image.Image],
    *,
    model: str = DEFAULT_VIT_MODEL,
    device: Any = -1,
    top_k: int = DEFAULT_TOP_K,
) -> list[VitTilePrediction | None]:
    if not crops:
        return []

    clean_model = str(model or DEFAULT_VIT_MODEL).strip() or DEFAULT_VIT_MODEL
    clean_device = _coerce_device(device)
    pipe = _load_pipeline(clean_model, clean_device)
    clean_top_k = max(1, int(top_k or DEFAULT_TOP_K))
    images = [crop.convert("RGB") for crop in crops]
    raw_outputs = pipe(images, top_k=clean_top_k)
    outputs = _normalize_pipeline_outputs(raw_outputs, expected_count=len(images))
    return [_prediction_from_output(output) for output in outputs]


def vit_classifier_enabled(config: dict[str, Any] | None, *, area: str) -> bool:
    if isinstance(config, dict) and _truthy(config.get("force_disabled", False)):
        return False
    env_enabled = os.environ.get("MAHJONG_COMPANION_VIT_ENABLED")
    if env_enabled is not None:
        return _truthy(env_enabled)
    if not isinstance(config, dict):
        return False

    backend = str(config.get("backend", "vit")).strip().lower()
    if backend not in {"auto", "vit", "mahjong_soul_vision", "mahjong-soul-vision"}:
        return False
    if not _truthy(config.get("enabled", True)):
        return False
    if _truthy(config.get("require_accelerator", False)) and not _accelerator_available():
        return False

    area_key = f"{area}_enabled"
    if area_key in config and not _truthy(config.get(area_key)):
        return False
    return True


def vit_model_from_config(config: dict[str, Any] | None) -> str:
    env_model = os.environ.get("MAHJONG_COMPANION_VIT_MODEL")
    if env_model:
        return env_model.strip() or DEFAULT_VIT_MODEL
    if isinstance(config, dict):
        return str(config.get("model", DEFAULT_VIT_MODEL)).strip() or DEFAULT_VIT_MODEL
    return DEFAULT_VIT_MODEL


def vit_device_from_config(config: dict[str, Any] | None) -> Any:
    env_device = os.environ.get("MAHJONG_COMPANION_VIT_DEVICE")
    if env_device is not None:
        return _coerce_device(env_device)
    if isinstance(config, dict):
        return _coerce_device(config.get("device", -1))
    return -1


def vit_top_k_from_config(config: dict[str, Any] | None) -> int:
    if isinstance(config, dict):
        return max(1, _coerce_int(config.get("top_k"), default=DEFAULT_TOP_K))
    return DEFAULT_TOP_K


def _load_pipeline(model: str, device: Any) -> Any:
    key = (model, str(device))
    failure = _PIPELINE_FAILURES.get(key)
    if failure:
        raise VitTileClassifierUnavailable(failure)

    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    with _PIPELINE_LOCK:
        cached = _PIPELINE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from transformers import pipeline

            loaded = pipeline("image-classification", model=model, device=device)
        except Exception as exc:  # pragma: no cover - exercised only with optional deps/model
            message = f"ViT tile classifier unavailable: {exc}"
            _PIPELINE_FAILURES[key] = message
            raise VitTileClassifierUnavailable(message) from exc
        _PIPELINE_CACHE[key] = loaded
        return loaded


def _prediction_from_output(output: Any) -> VitTilePrediction | None:
    if not isinstance(output, list) or not output:
        return None
    clean_top_k: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        tile = _tile_from_label(label)
        score = _coerce_float(item.get("score"), default=0.0)
        clean_top_k.append(
            {
                "label": label,
                "tile": tile,
                "score": round(score, 4),
            }
        )
    if not clean_top_k:
        return None
    top = clean_top_k[0]
    tile = str(top.get("tile", ""))
    if not tile:
        return None
    return VitTilePrediction(
        tile=tile,
        label=str(top.get("label", "")),
        confidence=float(top.get("score", 0.0) or 0.0),
        top_k=clean_top_k,
    )


def _tile_from_label(label: str) -> str:
    lower = str(label or "").strip().lower()
    mapped = _LABEL_MAP.get(lower, lower)
    return normalize_tile(mapped)


def _normalize_pipeline_outputs(raw_outputs: Any, *, expected_count: int) -> list[Any]:
    if expected_count == 1 and isinstance(raw_outputs, list) and raw_outputs and isinstance(raw_outputs[0], dict):
        return [raw_outputs]
    if isinstance(raw_outputs, list):
        return raw_outputs[:expected_count]
    return []


def _coerce_device(value: Any) -> Any:
    if value is None:
        return -1
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return -1
    if text.lower() == "auto":
        return 0 if _cuda_available() else -1
    try:
        return int(text)
    except ValueError:
        return text


def _accelerator_available() -> bool:
    global _ACCELERATOR_AVAILABLE
    if _ACCELERATOR_AVAILABLE is None:
        _ACCELERATOR_AVAILABLE = _cuda_available() or _mps_available()
    return _ACCELERATOR_AVAILABLE


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no", "off", "disabled"}


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
