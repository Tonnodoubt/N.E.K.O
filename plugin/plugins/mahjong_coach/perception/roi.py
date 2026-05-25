from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RoiBox:
    name: str
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def clipped(self, image_width: int, image_height: int) -> "RoiBox":
        left = max(0, min(self.left, image_width))
        top = max(0, min(self.top, image_height))
        right = max(left, min(self.right, image_width))
        bottom = max(top, min(self.bottom, image_height))
        return RoiBox(self.name, left, top, right - left, bottom - top)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_region_metrics(image: Image.Image, box: RoiBox, *, sample_step: int = 4) -> dict[str, float]:
    clipped = box.clipped(image.width, image.height)
    if clipped.width <= 0 or clipped.height <= 0:
        return {
            "mean_luma": 0.0,
            "bright_ratio": 0.0,
            "dark_ratio": 1.0,
            "stddev": 0.0,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
            "gold_ratio": 0.0,
            "orange_ratio": 0.0,
            "colorful_ratio": 0.0,
        }
    crop = image.crop((clipped.left, clipped.top, clipped.right, clipped.bottom)).convert("RGB")
    arr = np.asarray(crop, dtype=np.int16)[:: max(1, sample_step), :: max(1, sample_step)]
    if arr.size == 0:
        return collect_region_metrics(image, RoiBox("empty", 0, 0, 0, 0), sample_step=sample_step)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    luma = (r * 0.299 + g * 0.587 + b * 0.114)
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    colorful = (max_channel - min_channel) >= 35
    red = (r >= 135) & (r >= g * 1.25) & (r >= b * 1.25)
    green = (g >= 120) & (g >= r * 1.12) & (g >= b * 1.08)
    gold = (r >= 150) & (g >= 115) & (b <= 105) & ((r - b) >= 50)
    orange = (r >= 150) & (g >= 75) & (g <= 165) & (b <= 100)
    denom = max(1, int(luma.size))
    return {
        "mean_luma": float(luma.mean()),
        "bright_ratio": float((luma >= 180).sum()) / denom,
        "dark_ratio": float((luma <= 80).sum()) / denom,
        "stddev": float(luma.std()),
        "red_ratio": float(red.sum()) / denom,
        "green_ratio": float(green.sum()) / denom,
        "gold_ratio": float(gold.sum()) / denom,
        "orange_ratio": float(orange.sum()) / denom,
        "colorful_ratio": float(colorful.sum()) / denom,
    }

