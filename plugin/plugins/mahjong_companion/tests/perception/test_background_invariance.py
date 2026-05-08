"""Regression: perception accuracy across different table backgrounds / UI themes.

Walks ``tests/fixtures/multi_theme/<theme_id>/<case>.png`` pairs (with ``.tiles.json``
sidecars), runs the real perception pipeline against each (bypassing the fixture
shortcut), and reports per-theme accuracy on hand recognition, discard
recognition, and scene classification.

Designed to be a **measurement tool**, not a strict gate:
- If no fixtures are present, the test skips cleanly with an informational
  message — safe to merge before materials are populated.
- When fixtures are present, the test always passes; it prints a summary table
  and writes a JSON report to ``tests/_artifacts/background_invariance.json``
  so we can track baseline → improvement over time.
- A hard accuracy gate can be added later (after we have a baseline) by reading
  ``BACKGROUND_INVARIANCE_MIN_HAND_F1`` etc. from env vars.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from plugin.plugins.mahjong_companion.perception.pipeline import analyze_image_path

logger = logging.getLogger(__name__)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "multi_theme"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "background_invariance.json"

DISCARD_PLAYERS = ("self", "left_opponent", "top_opponent", "right_opponent")


# ─── data structures ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThemeCase:
    """One screenshot + its ground truth, tagged with theme metadata."""

    theme_id: str
    case_name: str
    image_path: Path
    ground_truth_path: Path
    theme_metadata: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.theme_id}/{self.case_name}"


@dataclass
class MultisetMetrics:
    predicted: int = 0
    expected: int = 0
    matched: int = 0

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.expected if self.expected else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


@dataclass
class CaseResult:
    case: ThemeCase
    scene_predicted: str = ""
    scene_expected: str = "in_match"
    hand: MultisetMetrics = field(default_factory=MultisetMetrics)
    discard: MultisetMetrics = field(default_factory=MultisetMetrics)
    error: str = ""

    @property
    def scene_correct(self) -> bool:
        return bool(self.scene_predicted) and self.scene_predicted == self.scene_expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.case.theme_id,
            "case_name": self.case.case_name,
            "scene": {
                "predicted": self.scene_predicted,
                "expected": self.scene_expected,
                "correct": self.scene_correct,
            },
            "hand": {
                "predicted": self.hand.predicted,
                "expected": self.hand.expected,
                "matched": self.hand.matched,
                "precision": round(self.hand.precision, 4),
                "recall": round(self.hand.recall, 4),
                "f1": round(self.hand.f1, 4),
            },
            "discard": {
                "predicted": self.discard.predicted,
                "expected": self.discard.expected,
                "matched": self.discard.matched,
                "precision": round(self.discard.precision, 4),
                "recall": round(self.discard.recall, 4),
                "f1": round(self.discard.f1, 4),
            },
            "error": self.error,
        }


# ─── fixture discovery ──────────────────────────────────────────────────────


def _discover_theme_cases(root: Path) -> list[ThemeCase]:
    """Find every PNG/JPG with a matching ``.tiles.json`` sidecar under ``root``.

    Folders whose name starts with ``_`` are skipped (reserved for examples /
    documentation). Cases without a sidecar are skipped with a warning.
    """
    if not root.is_dir():
        return []

    cases: list[ThemeCase] = []
    for theme_dir in sorted(root.iterdir()):
        if not theme_dir.is_dir() or theme_dir.name.startswith("_"):
            continue
        metadata = _load_theme_metadata(theme_dir)
        for image_path in sorted(_iter_images(theme_dir)):
            ground_truth = _resolve_sidecar(image_path)
            if ground_truth is None:
                logger.warning(
                    "skipping %s: no .tiles.json / -tiles.json / .label.json sidecar",
                    image_path,
                )
                continue
            cases.append(
                ThemeCase(
                    theme_id=theme_dir.name,
                    case_name=image_path.stem,
                    image_path=image_path,
                    ground_truth_path=ground_truth,
                    theme_metadata=metadata,
                ),
            )
    return cases


def _iter_images(theme_dir: Path) -> Iterator[Path]:
    for path in theme_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield path


def _resolve_sidecar(image_path: Path) -> Path | None:
    candidates = [
        image_path.with_name(f"{image_path.stem}-tiles.json"),
        image_path.with_suffix(".tiles.json"),
        image_path.with_suffix(".label.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_theme_metadata(theme_dir: Path) -> dict[str, Any]:
    meta_path = theme_dir / "theme.json"
    if not meta_path.exists():
        return {"theme_id": theme_dir.name}
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("invalid theme metadata at %s: %s", meta_path, exc)
        return {"theme_id": theme_dir.name}
    if not isinstance(loaded, dict):
        return {"theme_id": theme_dir.name}
    loaded.setdefault("theme_id", theme_dir.name)
    return loaded


# ─── ground-truth normalization ─────────────────────────────────────────────


def _load_ground_truth(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_hand_tiles(ground_truth: dict[str, Any]) -> list[str]:
    raw = ground_truth.get("hand_tiles")
    return [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else []


def _expected_discards(ground_truth: dict[str, Any]) -> dict[str, list[str]]:
    """Normalize the discard piles into ``{player: [tile, tile, ...]}``."""
    raw = ground_truth.get("discard_piles")
    if not isinstance(raw, dict):
        return {player: [] for player in DISCARD_PLAYERS}
    normalized: dict[str, list[str]] = {player: [] for player in DISCARD_PLAYERS}
    for player, items in raw.items():
        player_key = str(player).strip()
        if player_key not in normalized or not isinstance(items, list):
            continue
        for item in items:
            tile = ""
            if isinstance(item, dict):
                tile = str(item.get("tile", "")).strip()
            elif isinstance(item, str):
                tile = item.strip()
            if tile:
                normalized[player_key].append(tile)
    return normalized


def _expected_scene(ground_truth: dict[str, Any]) -> str:
    raw = ground_truth.get("scene")
    return str(raw).strip() if isinstance(raw, str) and raw.strip() else "in_match"


# ─── prediction normalization ───────────────────────────────────────────────


def _predicted_discards(state_discards: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {player: [] for player in DISCARD_PLAYERS}
    if not isinstance(state_discards, dict):
        return normalized
    for player, items in state_discards.items():
        player_key = str(player).strip()
        if player_key not in normalized or not isinstance(items, list):
            continue
        for item in items:
            tile = ""
            if isinstance(item, dict):
                tile = str(item.get("tile", "")).strip()
            elif isinstance(item, str):
                tile = item.strip()
            if tile:
                normalized[player_key].append(tile)
    return normalized


# ─── scoring ────────────────────────────────────────────────────────────────


def _score_multiset(predicted: list[str], expected: list[str]) -> MultisetMetrics:
    """Multiset overlap: matched = sum over tiles of min(predicted_count, expected_count)."""
    pred_counter = Counter(predicted)
    exp_counter = Counter(expected)
    matched = sum((pred_counter & exp_counter).values())
    return MultisetMetrics(
        predicted=sum(pred_counter.values()),
        expected=sum(exp_counter.values()),
        matched=matched,
    )


def _score_case(case: ThemeCase) -> CaseResult:
    result = CaseResult(case=case)
    try:
        ground_truth = _load_ground_truth(case.ground_truth_path)
    except (json.JSONDecodeError, OSError) as exc:
        result.error = f"sidecar_load_failed: {exc}"
        return result

    result.scene_expected = _expected_scene(ground_truth)
    expected_hand = _expected_hand_tiles(ground_truth)
    expected_discards = _expected_discards(ground_truth)

    try:
        # fixture_mode="disabled" forces the real perception path. Without this,
        # tile_parser._load_fixture would short-circuit to the sidecar and we'd
        # be measuring the ground truth against itself.
        perceived, _ = analyze_image_path(case.image_path, fixture_mode="disabled")
    except Exception as exc:  # noqa: BLE001 - regression must survive any pipeline error
        result.error = f"pipeline_raised: {type(exc).__name__}: {exc}"
        return result

    result.scene_predicted = perceived.scene
    result.hand = _score_multiset(predicted=list(perceived.hand_tiles), expected=expected_hand)

    predicted_discards = _predicted_discards(perceived.discard_piles)
    aggregate = MultisetMetrics()
    for player in DISCARD_PLAYERS:
        per_player = _score_multiset(
            predicted=predicted_discards.get(player, []),
            expected=expected_discards.get(player, []),
        )
        aggregate.predicted += per_player.predicted
        aggregate.expected += per_player.expected
        aggregate.matched += per_player.matched
    result.discard = aggregate
    return result


# ─── reporting ──────────────────────────────────────────────────────────────


def _aggregate_by_theme(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = summary.setdefault(
            result.case.theme_id,
            {
                "display_name": result.case.theme_metadata.get("display_name", result.case.theme_id),
                "is_official": result.case.theme_metadata.get("is_official"),
                "case_count": 0,
                "errors": 0,
                "scene_correct": 0,
                "hand": MultisetMetrics(),
                "discard": MultisetMetrics(),
            },
        )
        bucket["case_count"] += 1
        if result.error:
            bucket["errors"] += 1
            continue
        if result.scene_correct:
            bucket["scene_correct"] += 1
        for axis_name in ("hand", "discard"):
            agg: MultisetMetrics = bucket[axis_name]
            metric: MultisetMetrics = getattr(result, axis_name)
            agg.predicted += metric.predicted
            agg.expected += metric.expected
            agg.matched += metric.matched
    return summary


def _format_summary_table(theme_summary: dict[str, dict[str, Any]]) -> str:
    if not theme_summary:
        return "(no fixtures)"
    rows = ["", "Background invariance summary:", ""]
    header = (
        f"{'theme':<28} {'cases':>6} {'errs':>5} {'scene%':>7} "
        f"{'hand_F1':>9} {'hand_R':>8} {'disc_F1':>9} {'disc_R':>8}"
    )
    rows.append(header)
    rows.append("-" * len(header))
    for theme_id in sorted(theme_summary.keys()):
        bucket = theme_summary[theme_id]
        cases: int = bucket["case_count"]
        scene_pct = (bucket["scene_correct"] / cases * 100.0) if cases else 0.0
        hand: MultisetMetrics = bucket["hand"]
        discard: MultisetMetrics = bucket["discard"]
        rows.append(
            f"{theme_id[:28]:<28} {cases:>6} {bucket['errors']:>5} {scene_pct:>6.1f}% "
            f"{hand.f1:>9.3f} {hand.recall:>8.3f} {discard.f1:>9.3f} {discard.recall:>8.3f}",
        )
    rows.append("")
    return "\n".join(rows)


def _write_artifact(results: list[CaseResult], theme_summary: dict[str, dict[str, Any]]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fixture_root": str(FIXTURE_ROOT),
        "case_count": len(results),
        "themes": {
            theme_id: {
                "display_name": bucket["display_name"],
                "is_official": bucket["is_official"],
                "case_count": bucket["case_count"],
                "errors": bucket["errors"],
                "scene_accuracy": (
                    bucket["scene_correct"] / bucket["case_count"]
                    if bucket["case_count"]
                    else 0.0
                ),
                "hand": {
                    "precision": round(bucket["hand"].precision, 4),
                    "recall": round(bucket["hand"].recall, 4),
                    "f1": round(bucket["hand"].f1, 4),
                },
                "discard": {
                    "precision": round(bucket["discard"].precision, 4),
                    "recall": round(bucket["discard"].recall, 4),
                    "f1": round(bucket["discard"].f1, 4),
                },
            }
            for theme_id, bucket in theme_summary.items()
        },
        "cases": [result.to_dict() for result in results],
    }
    ARTIFACT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── tests ──────────────────────────────────────────────────────────────────


@pytest.mark.plugin_unit
def test_background_invariance_corpus_present() -> None:
    """Sanity check: surfaces a clear skip reason when no fixtures exist."""
    cases = _discover_theme_cases(FIXTURE_ROOT)
    if not cases:
        pytest.skip(
            "No multi-theme fixtures found at "
            f"{FIXTURE_ROOT}. See {FIXTURE_ROOT / 'README.md'} for what to add.",
        )


@pytest.mark.plugin_unit
def test_background_invariance_report() -> None:
    """Run perception against every theme fixture, print summary, write artifact.

    This test is **measurement-only by default** — it always passes when the
    pipeline doesn't crash. To enable hard gates (fail when accuracy regresses),
    set environment variables:

    - ``BACKGROUND_INVARIANCE_MIN_HAND_F1`` (e.g. ``0.85``)
    - ``BACKGROUND_INVARIANCE_MIN_DISCARD_F1`` (e.g. ``0.70``)
    - ``BACKGROUND_INVARIANCE_MIN_SCENE_ACCURACY`` (e.g. ``0.90``)
    """
    cases = _discover_theme_cases(FIXTURE_ROOT)
    if not cases:
        pytest.skip(f"No multi-theme fixtures found at {FIXTURE_ROOT}.")

    results = [_score_case(case) for case in cases]
    theme_summary = _aggregate_by_theme(results)
    _write_artifact(results, theme_summary)

    table = _format_summary_table(theme_summary)
    logger.info(table)
    print(table)  # noqa: T201 - pytest -s shows it; useful for local runs.

    _maybe_enforce_thresholds(theme_summary)


def _maybe_enforce_thresholds(theme_summary: dict[str, dict[str, Any]]) -> None:
    """If env-var thresholds are set, fail the test when any theme falls short."""
    min_hand_f1 = _env_float("BACKGROUND_INVARIANCE_MIN_HAND_F1")
    min_discard_f1 = _env_float("BACKGROUND_INVARIANCE_MIN_DISCARD_F1")
    min_scene_acc = _env_float("BACKGROUND_INVARIANCE_MIN_SCENE_ACCURACY")
    if min_hand_f1 is None and min_discard_f1 is None and min_scene_acc is None:
        return

    failures: list[str] = []
    for theme_id, bucket in sorted(theme_summary.items()):
        if min_hand_f1 is not None and bucket["hand"].f1 < min_hand_f1:
            failures.append(
                f"{theme_id}: hand F1 {bucket['hand'].f1:.3f} < threshold {min_hand_f1:.3f}",
            )
        if min_discard_f1 is not None and bucket["discard"].f1 < min_discard_f1:
            failures.append(
                f"{theme_id}: discard F1 {bucket['discard'].f1:.3f} < threshold {min_discard_f1:.3f}",
            )
        if min_scene_acc is not None and bucket["case_count"]:
            scene_acc = bucket["scene_correct"] / bucket["case_count"]
            if scene_acc < min_scene_acc:
                failures.append(
                    f"{theme_id}: scene accuracy {scene_acc:.3f} < threshold {min_scene_acc:.3f}",
                )
    if failures:
        pytest.fail("Background invariance thresholds violated:\n  " + "\n  ".join(failures))


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", name, raw)
        return None
