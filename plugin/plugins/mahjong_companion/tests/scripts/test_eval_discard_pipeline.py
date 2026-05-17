from __future__ import annotations

from plugin.plugins.mahjong_companion.scripts.eval_discard_pipeline import (
    Score,
    _threshold_failures,
)


def test_score_metrics() -> None:
    score = Score(correct=8, expected=10, predicted=16)

    assert score.precision == 0.5
    assert score.recall == 0.8
    assert round(score.f1, 4) == 0.6154


def test_threshold_failures_reports_all_failed_axes() -> None:
    score = Score(correct=8, expected=10, predicted=16)

    failures = _threshold_failures(
        score,
        min_precision=0.7,
        min_recall=0.9,
        min_f1=0.8,
    )

    assert failures == [
        "precision 0.500 < 0.700",
        "recall 0.800 < 0.900",
        "F1 0.615 < 0.800",
    ]


def test_threshold_failures_empty_when_gate_passes() -> None:
    score = Score(correct=95, expected=100, predicted=100)

    assert _threshold_failures(
        score,
        min_precision=0.7,
        min_recall=0.9,
        min_f1=0.8,
    ) == []
