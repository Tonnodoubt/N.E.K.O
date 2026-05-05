"""Generate the simulator-baseline-v1.1 report.

Runs the toy four-player simulator across multiple opponent baselines,
multiple seeds, and a configurable game count. Writes a JSON report to
``plugin/plugins/mahjong_companion/plans/artifacts/simulator-baseline-v1.1.json``
so the companion strategy has a regression baseline beyond the v0.x evaluation
gates.

Usage::

    python scripts/mahjong_simulator_baseline.py --games 500 --seeds 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from plugin.plugins.mahjong_companion.scripts import (  # noqa: F401  ensure namespace
    check_v10_release as _check_v10_release_module,
)
from scripts.mahjong_simulate_four_player_strategy import (
    POLICIES,
    Aggregate,
    run_batch,
)

DEFAULT_REPORT_PATH = Path(
    "plugin/plugins/mahjong_companion/plans/artifacts/simulator-baseline-v1.1.json"
)
DEFAULT_GAMES = 500
DEFAULT_SEEDS = 5
BASE_SEED = 100_001
RISK_DEAL_IN_WEIGHT = 1.5
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def _aggregate_to_rates(aggregate: Aggregate) -> dict[str, float]:
    games = max(1, aggregate.games)
    win_rate = aggregate.hero_wins / games
    deal_in_rate = aggregate.dealt_in_by_hero / games
    return {
        "win_rate": win_rate,
        "opponent_win_rate": aggregate.opponent_wins / games,
        "draw_rate": aggregate.draws / games,
        "deal_in_rate": deal_in_rate,
        "risk_adjusted_score": win_rate - deal_in_rate * RISK_DEAL_IN_WEIGHT,
        "tenpai_or_win_rate": aggregate.hero_tenpai_or_win / games,
        "avg_final_shanten": aggregate.hero_final_shanten_sum / games,
        "avg_turn_draws": aggregate.turn_draw_sum / games,
        "tsumo_count": aggregate.hero_tsumo,
        "ron_count": aggregate.hero_ron,
    }


def _summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    summary = {
        "sample_count": len(samples),
    }
    summary.update(_metric_summary("win_rate", samples, lower=0.0, upper=1.0))
    summary.update(_metric_summary("deal_in_rate", samples, lower=0.0, upper=1.0))
    summary.update(_metric_summary("risk_adjusted_score", samples))
    return summary


def _metric_summary(
    metric: str,
    samples: list[dict[str, float]],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, float]:
    values = [sample[metric] for sample in samples]
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * stdev / (len(values) ** 0.5) if len(values) > 1 else 0.0
    ci95_low = mean - ci95
    ci95_high = mean + ci95
    if lower is not None:
        ci95_low = max(lower, ci95_low)
    if upper is not None:
        ci95_high = min(upper, ci95_high)
    return {
        f"{metric}_mean": mean,
        f"{metric}_min": min(values),
        f"{metric}_max": max(values),
        f"{metric}_stdev": stdev,
        f"{metric}_ci95_low": ci95_low,
        f"{metric}_ci95_high": ci95_high,
    }


def run_baseline(games: int, seeds: int, *, workers: int = 1) -> dict[str, Any]:
    return run_baseline_matrix(
        games=games,
        seeds=seeds,
        workers=workers,
        hero_policies=["random", "shanten", "companion", "oracle"],
        opponent_policies=["random", "shanten", "oracle"],
    )


def run_baseline_matrix(
    games: int,
    seeds: int,
    *,
    workers: int = 1,
    hero_policies: list[str],
    opponent_policies: list[str],
) -> dict[str, Any]:
    started_at = time.time()
    seed_offsets = [BASE_SEED + step * 7919 for step in range(seeds)]
    matchups: list[dict[str, Any]] = []

    samples_by_matchup = _run_samples(
        games=games,
        seed_offsets=seed_offsets,
        hero_policies=hero_policies,
        opponent_policies=opponent_policies,
        workers=workers,
    )

    for hero in hero_policies:
        for opponent in opponent_policies:
            samples = samples_by_matchup[(hero, opponent)]
            matchups.append(
                {
                    "hero_policy": hero,
                    "opponent_policy": opponent,
                    "games_per_seed": games,
                    "seeds": seed_offsets,
                    "samples": samples,
                    "summary": _summarize(samples),
                }
            )

    duration_seconds = time.time() - started_at
    return {
        "schema_version": "mahjong-companion-simulator-baseline-v1",
        "baseline_version": "v1.1",
        "plugin_version": "1.0.1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "games_per_seed": games,
            "seed_count": seeds,
            "max_draws_per_game": 70,
            "risk_deal_in_weight": RISK_DEAL_IN_WEIGHT,
            "worker_count": max(1, workers),
            "hero_policies": hero_policies,
            "opponent_policies": opponent_policies,
        },
        "matchups": matchups,
        "duration_seconds": round(duration_seconds, 3),
    }


def _run_samples(
    *,
    games: int,
    seed_offsets: list[int],
    hero_policies: list[str],
    opponent_policies: list[str],
    workers: int,
) -> dict[tuple[str, str], list[dict[str, float]]]:
    tasks = [
        (games, seed, hero, opponent)
        for hero in hero_policies
        for opponent in opponent_policies
        for seed in seed_offsets
    ]
    worker_count = max(1, workers)
    if worker_count == 1:
        rows = [_run_sample(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(_run_sample, tasks))

    grouped: dict[tuple[str, str], list[dict[str, float]]] = {
        (hero, opponent): [] for hero in hero_policies for opponent in opponent_policies
    }
    for hero, opponent, _seed, sample in rows:
        grouped[(hero, opponent)].append(sample)
    return grouped


def _run_sample(task: tuple[int, int, str, str]) -> tuple[str, str, int, dict[str, float]]:
    games, seed, hero, opponent = task
    aggregate = run_batch(
        games,
        seed=seed,
        hero_policy=POLICIES[hero],
        opponent_policy=POLICIES[opponent],
    )
    return hero, opponent, seed, _aggregate_to_rates(aggregate)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run companion simulator baseline.")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--hero-policy", choices=sorted(POLICIES), action="append")
    parser.add_argument("--opponent-policy", choices=sorted(POLICIES), action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_baseline_matrix(
        games=args.games,
        seeds=args.seeds,
        workers=args.workers,
        hero_policies=args.hero_policy or ["random", "shanten", "companion", "oracle"],
        opponent_policies=args.opponent_policy or ["random", "shanten", "oracle"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2 if args.pretty else None, ensure_ascii=False))
    print(json.dumps(report["matchups"], indent=2 if args.pretty else None, ensure_ascii=False))
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
