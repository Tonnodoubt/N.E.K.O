from __future__ import annotations

from collections import Counter

import pytest

from scripts import mahjong_simulator_baseline as module
from scripts import mahjong_simulate_four_player_strategy as simulator
from scripts.mahjong_simulate_four_player_strategy import Aggregate
from plugin.plugins.mahjong_companion.decision.tile_efficiency import (
    _calculate_discard_ukeire_from_counts,
    _estimate_post_discard_shanten_from_counts,
    _estimate_shanten_from_counts,
    _remaining_tile_counts,
)


def test_aggregate_to_rates_includes_risk_adjusted_score() -> None:
    aggregate = Aggregate(games=10, hero_wins=4, opponent_wins=5, draws=1, dealt_in_by_hero=2)

    rates = module._aggregate_to_rates(aggregate)

    assert rates["win_rate"] == 0.4
    assert rates["deal_in_rate"] == 0.2
    assert rates["risk_adjusted_score"] == pytest.approx(0.1)


def test_summarize_reports_sample_stdev_and_ci95() -> None:
    samples = [
        {"win_rate": 0.30, "deal_in_rate": 0.10, "risk_adjusted_score": 0.15},
        {"win_rate": 0.35, "deal_in_rate": 0.12, "risk_adjusted_score": 0.17},
        {"win_rate": 0.40, "deal_in_rate": 0.14, "risk_adjusted_score": 0.19},
        {"win_rate": 0.45, "deal_in_rate": 0.16, "risk_adjusted_score": 0.21},
        {"win_rate": 0.50, "deal_in_rate": 0.18, "risk_adjusted_score": 0.23},
    ]

    summary = module._summarize(samples)

    assert summary["sample_count"] == 5
    assert summary["win_rate_mean"] == pytest.approx(0.40)
    assert summary["win_rate_stdev"] == pytest.approx(0.07905694)
    assert summary["win_rate_ci95_low"] == pytest.approx(0.3307035)
    assert summary["win_rate_ci95_high"] == pytest.approx(0.4692965)
    assert summary["risk_adjusted_score_mean"] == pytest.approx(0.19)
    assert "risk_adjusted_score_ci95_low" in summary


def test_run_baseline_covers_v11_matchup_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_batch(games: int, *, seed: int, hero_policy: object, opponent_policy: object) -> Aggregate:
        del seed, hero_policy, opponent_policy
        return Aggregate(games=games, hero_wins=2, opponent_wins=4, draws=1, dealt_in_by_hero=1)

    monkeypatch.setattr(module, "run_batch", fake_run_batch)

    report = module.run_baseline(games=7, seeds=2)

    assert report["baseline_version"] == "v1.1"
    assert report["config"]["risk_deal_in_weight"] == 1.5
    assert len(report["matchups"]) == 12
    assert {item["hero_policy"] for item in report["matchups"]} == {"random", "shanten", "companion", "oracle"}
    assert {item["opponent_policy"] for item in report["matchups"]} == {"random", "shanten", "oracle"}
    assert all(item["summary"]["sample_count"] == 2 for item in report["matchups"])


def test_run_baseline_matrix_can_filter_matchups(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_batch(games: int, *, seed: int, hero_policy: object, opponent_policy: object) -> Aggregate:
        del seed, hero_policy, opponent_policy
        return Aggregate(games=games, hero_wins=2, opponent_wins=4, draws=1, dealt_in_by_hero=1)

    monkeypatch.setattr(module, "run_batch", fake_run_batch)

    report = module.run_baseline_matrix(
        games=7,
        seeds=2,
        hero_policies=["companion"],
        opponent_policies=["shanten"],
    )

    assert len(report["matchups"]) == 1
    assert report["config"]["hero_policies"] == ["companion"]
    assert report["config"]["opponent_policies"] == ["shanten"]
    assert report["matchups"][0]["hero_policy"] == "companion"
    assert report["matchups"][0]["opponent_policy"] == "shanten"


def test_simulator_fast_tuple_estimators_match_tile_efficiency_helpers() -> None:
    hand = ["1m", "2m", "3m", "4m", "5m", "6m", "2p", "3p", "4p", "6s", "7s", "8s", "5z", "5z"]
    visible_tiles = ["9m", "9m", "1p", "7z"]
    counter = Counter(hand)
    count_tuple = simulator._hand_counts_tuple(hand)
    remaining = _remaining_tile_counts(counter, visible_tiles)
    remaining_tuple = tuple(simulator._remaining_tile_counts_tuple(count_tuple, visible_tiles))

    assert simulator._estimate_shanten_tuple(count_tuple) == _estimate_shanten_from_counts(counter)
    assert remaining_tuple == tuple(remaining)

    for tile in counter:
        tile_index = simulator.TILE_TO_INDEX[tile]
        assert simulator._estimate_post_discard_shanten_tuple(
            count_tuple,
            tile_index,
        ) == _estimate_post_discard_shanten_from_counts(counter, tile)
        assert simulator._calculate_discard_ukeire_tuple(
            count_tuple,
            tile_index,
            remaining_tuple,
        ) == _calculate_discard_ukeire_from_counts(counter, tile, remaining)
