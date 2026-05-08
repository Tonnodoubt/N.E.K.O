"""Detect the Mahjong Soul score panel as a visual anchor.

The score panel is the dark rectangular region at screen center showing
round info, scores, and wind icons. It's theme-invariant because it's
game-state UI rather than cosmetic chrome — every theme keeps the same
shape, position, and dark fill, so the panel makes a stable reference
point for downstream geometry (4-player discard ROIs, dora area, hand
baseline, etc.) without any per-resolution calibration profile.

Detection strategy:
  1. Convert to grayscale, find dark regions (luma < threshold)
  2. Walk connected components in a single pass
  3. Filter by size, aspect ratio, and rectangularity (fill ratio)
  4. Score by area + centrality + rectangularity, return the winner

Public API:
  * :func:`detect_score_panel` — backwards-compatible bbox-only result
  * :func:`detect_score_panel_anchor` — returns a :class:`ScorePanelAnchor`
    dataclass with bbox + center + image_size + dimensions, designed for
    downstream anchor-driven geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


MIN_PANEL_AREA_RATIO = 0.005  # panel area / image area
MAX_PANEL_AREA_RATIO = 0.08
DARK_LUMA_THRESHOLD = 90
MIN_ASPECT_RATIO = 1.5  # panel is wider than tall
MAX_ASPECT_RATIO = 6.0
MIN_FILL_RATIO = 0.4    # connected component must fill its bbox densely


@dataclass(frozen=True)
class ScorePanelAnchor:
    """Stable anchor derived from the score panel.

    Downstream geometry (4-player discard ROIs, hand baseline, dora area)
    is expected to be parameterised relative to ``center`` and ``width`` /
    ``height`` rather than to absolute pixel coordinates, so the same
    code works for any image size and any client zoom.
    """

    bbox: tuple[int, int, int, int]   # (left, top, right, bottom)
    center: tuple[int, int]
    image_size: tuple[int, int]       # (width, height)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox": list(self.bbox),
            "center": list(self.center),
            "image_size": list(self.image_size),
            "width": self.width,
            "height": self.height,
        }


def detect_score_panel(
    image: Image.Image,
    *,
    dark_luma_threshold: int = DARK_LUMA_THRESHOLD,
    min_area_ratio: float = MIN_PANEL_AREA_RATIO,
    max_area_ratio: float = MAX_PANEL_AREA_RATIO,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
    min_fill_ratio: float = MIN_FILL_RATIO,
) -> tuple[int, int, int, int] | None:
    """Return ``(left, top, right, bottom)`` of the score panel, or ``None``.

    All thresholds are exposed as keyword arguments so downstream tests
    and integration code can tighten or loosen the gates without forking
    the algorithm. The defaults are tuned on the multi-theme fixture set
    in ``tests/fixtures/multi_theme``.
    """
    width, height = image.size
    arr = np.array(image.convert("L"), dtype=np.float32)
    mask = arr < dark_luma_threshold

    visited = np.zeros(mask.shape, dtype=bool)
    center_x, center_y = width / 2, height / 2
    img_area = width * height
    min_area = int(img_area * min_area_ratio)
    max_area = int(img_area * max_area_ratio)

    best: tuple[int, int, int, int] | None = None
    best_score = -1.0

    ys_all, xs_all = np.where(mask)
    for i in range(len(ys_all)):
        y, x = int(ys_all[i]), int(xs_all[i])
        if visited[y, x]:
            continue

        cx, cy = _flood_fill_component(mask, visited, x, y, width, height)
        area = len(cx)
        if area < min_area or area > max_area:
            continue

        x1, x2 = min(cx), max(cx)
        y1, y2 = min(cy), max(cy)
        bw, bh = x2 - x1 + 1, y2 - y1 + 1

        aspect = bw / max(1, bh)
        if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
            continue

        rect_area = bw * bh
        fill = area / max(1, rect_area)
        if fill < min_fill_ratio:
            continue

        comp_cx = (x1 + x2) / 2
        comp_cy = (y1 + y2) / 2
        dist = float(np.hypot(comp_cx - center_x, comp_cy - center_y))
        max_dist = float(np.hypot(center_x, center_y))
        centrality = 1.0 - dist / max(1.0, max_dist)
        score = fill * 0.3 + centrality * 0.5 + (area / img_area) * 100 * 0.2

        if score > best_score:
            best_score = score
            best = (int(x1), int(y1), int(x2) + 1, int(y2) + 1)

    return best


def detect_score_panel_anchor(
    image: Image.Image,
    **kwargs: object,
) -> ScorePanelAnchor | None:
    """High-level wrapper returning a :class:`ScorePanelAnchor` for use
    as a geometric reference point downstream."""
    bbox = detect_score_panel(image, **kwargs)  # type: ignore[arg-type]
    if bbox is None:
        return None
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    return ScorePanelAnchor(
        bbox=bbox,
        center=(cx, cy),
        image_size=image.size,
    )


def _flood_fill_component(
    mask: np.ndarray,
    visited: np.ndarray,
    start_x: int,
    start_y: int,
    width: int,
    height: int,
) -> tuple[list[int], list[int]]:
    """Iterative 4-connected flood fill returning the (xs, ys) of the
    component containing ``(start_x, start_y)``."""
    stack = [(start_x, start_y)]
    cx: list[int] = []
    cy: list[int] = []
    visited[start_y, start_x] = True
    while stack:
        px, py = stack.pop()
        cx.append(px)
        cy.append(py)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx] and mask[ny, nx]:
                visited[ny, nx] = True
                stack.append((nx, ny))
    return cx, cy


def _demo_main() -> None:  # pragma: no cover - exploratory script
    """Run detection on the multi-theme fixture set and dump an overlay.

    Kept around because it's a useful debugging tool, but every
    fixture-path lookup is lazy so importing this module never touches
    the filesystem or mutates ``sys.path``.
    """
    from pathlib import Path

    from PIL import ImageDraw

    repo_root = Path(__file__).resolve().parents[4]
    fixtures = repo_root / "plugin/plugins/mahjong_companion/tests/fixtures/multi_theme"
    out = repo_root / "plugin/plugins/mahjong_companion/tests/_artifacts/panel_detection.png"

    cases = []
    for theme_dir in sorted(fixtures.iterdir()):
        if not theme_dir.is_dir() or theme_dir.name.startswith("_"):
            continue
        for png in sorted(theme_dir.glob("*.png")):
            cases.append((theme_dir.name, png))

    print(f"Testing panel detection on {len(cases)} screenshots\n")
    centers: list[tuple[int, int]] = []
    for theme_id, png_path in cases:
        with Image.open(png_path) as opened:
            anchor = detect_score_panel_anchor(opened.convert("RGB"))
        if anchor is None:
            print(f"  {theme_id}/{png_path.name}: NOT DETECTED")
            continue
        print(
            f"  {theme_id}/{png_path.name}: bbox={anchor.bbox} "
            f"center={anchor.center} size={anchor.width}x{anchor.height}"
        )
        centers.append(anchor.center)

    if centers:
        mean_cx = float(np.mean([c[0] for c in centers]))
        mean_cy = float(np.mean([c[1] for c in centers]))
        max_dx = max(abs(c[0] - mean_cx) for c in centers)
        max_dy = max(abs(c[1] - mean_cy) for c in centers)
        print(
            f"\nPanel center stability: mean=({mean_cx:.0f},{mean_cy:.0f}) "
            f"max_deviation=({max_dx:.0f},{max_dy:.0f})"
        )

    if cases:
        with Image.open(cases[0][1]) as opened:
            preview = opened.convert("RGB").copy()
        anchor = detect_score_panel_anchor(preview)
        if anchor is not None:
            draw = ImageDraw.Draw(preview)
            draw.rectangle(anchor.bbox, outline=(0, 255, 0), width=3)
            cx, cy = anchor.center
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 0, 0))
            draw.line((cx - 15, cy, cx + 15, cy), fill=(255, 0, 0), width=1)
            draw.line((cx, cy - 15, cx, cy + 15), fill=(255, 0, 0), width=1)
        out.parent.mkdir(parents=True, exist_ok=True)
        preview.save(out)
        print(f"\nSaved detection overlay to {out}")


if __name__ == "__main__":
    _demo_main()
