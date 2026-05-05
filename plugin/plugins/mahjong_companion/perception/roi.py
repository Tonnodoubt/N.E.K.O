from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageStat


@dataclass
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_default_rois(width: int, height: int) -> dict[str, RoiBox]:
    return {
        "top_banner": _relative_roi("top_banner", width, height, 0.05, 0.03, 0.90, 0.12),
        "center_dialog": _relative_roi("center_dialog", width, height, 0.25, 0.20, 0.50, 0.30),
        "bottom_action_bar": _relative_roi("bottom_action_bar", width, height, 0.18, 0.76, 0.64, 0.16),
        "bottom_hand_area": _relative_roi("bottom_hand_area", width, height, 0.12, 0.68, 0.76, 0.26),
        "right_replay_panel": _relative_roi("right_replay_panel", width, height, 0.78, 0.10, 0.18, 0.70),
    }


def collect_region_metrics(
    image: Image.Image,
    box: Optional[RoiBox],
    sample_step: int = 6,
) -> dict[str, Any]:
    region = image.crop((box.left, box.top, box.right, box.bottom)) if box is not None else image
    rgb_region = region.convert("RGB")
    stat = ImageStat.Stat(rgb_region)
    avg_r, avg_g, avg_b = [float(value) for value in stat.mean[:3]]
    stddev = sum(float(value) for value in stat.stddev[:3]) / 3.0

    width, height = rgb_region.size
    step = max(1, sample_step)

    arr = np.asarray(rgb_region)[::step, ::step].astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (r + g + b) / 3.0
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = max_c - min_c
    sample_count = max(1, int(brightness.size))

    mean_luma = float(ImageStat.Stat(rgb_region.convert("L")).mean[0])

    return {
        "box": box.to_dict() if box is not None else None,
        "size": {"width": width, "height": height},
        "avg_rgb": {"r": round(avg_r, 2), "g": round(avg_g, 2), "b": round(avg_b, 2)},
        "mean_luma": round(mean_luma, 2),
        "stddev": round(stddev, 2),
        "bright_ratio": round(float(np.sum(brightness >= 200)) / sample_count, 4),
        "dark_ratio": round(float(np.sum(brightness <= 70)) / sample_count, 4),
        "white_ratio": round(float(np.sum((brightness >= 225) & (saturation <= 25))) / sample_count, 4),
        "colorful_ratio": round(float(np.sum(saturation >= 55)) / sample_count, 4),
        "gold_ratio": round(float(np.sum((r >= 170) & (g >= 135) & (b <= 130) & (r >= b + 45))) / sample_count, 4),
        "orange_ratio": round(float(np.sum((r >= 185) & (g >= 90) & (g <= 190) & (b <= 125) & (r >= g) & (g >= b))) / sample_count, 4),
        "red_ratio": round(float(np.sum((r >= 175) & (g <= 110) & (b <= 120))) / sample_count, 4),
        "green_ratio": round(float(np.sum((g >= 150) & (g >= r + 25) & (g >= b + 25))) / sample_count, 4),
        "sample_count": sample_count,
    }


def _relative_roi(
    name: str,
    width: int,
    height: int,
    rel_left: float,
    rel_top: float,
    rel_width: float,
    rel_height: float,
) -> RoiBox:
    left = max(0, min(width - 1, int(width * rel_left)))
    top = max(0, min(height - 1, int(height * rel_top)))
    roi_width = max(1, min(width - left, int(width * rel_width)))
    roi_height = max(1, min(height - top, int(height * rel_height)))
    return RoiBox(name=name, left=left, top=top, width=roi_width, height=roi_height)
