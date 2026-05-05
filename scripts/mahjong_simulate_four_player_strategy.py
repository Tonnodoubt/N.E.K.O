from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from plugin.plugins.mahjong_companion.decision.tile_efficiency import (
    _estimate_candidate_discards,
    _estimate_raw_shanten_from_counts,
    _raw_discard_score,
    _raw_ukeire_score,
)

TILES = [f"{number}{suit}" for suit in "mps" for number in range(1, 10)] + [
    f"{number}z" for number in range(1, 8)
]
TILE_TO_INDEX = {tile: index for index, tile in enumerate(TILES)}
BASE_DECK = [tile for tile in TILES for _ in range(4)]
MAX_DRAWS = 70

Policy = Callable[[list[str], list[str], random.Random], str]


@dataclass
class GameResult:
    winner: int | None
    win_type: str = ""
    turn_draws: int = 0
    hero_final_shanten: int = 9
    hero_tenpai: bool = False


@dataclass
class Aggregate:
    games: int = 0
    hero_wins: int = 0
    opponent_wins: int = 0
    draws: int = 0
    hero_tsumo: int = 0
    hero_ron: int = 0
    dealt_in_by_hero: int = 0
    hero_tenpai_or_win: int = 0
    hero_final_shanten_sum: int = 0
    turn_draw_sum: int = 0


def companion_policy(hand: list[str], visible_tiles: list[str], rng: random.Random) -> str:
    candidates = _estimate_candidate_discards(
        hand,
        dora_indicators=[],
        hints={
            "deck_state_complete": True,
            "visible_tiles": list(visible_tiles),
        },
    )
    if candidates:
        return str(candidates[0].get("tile") or "")
    return rng.choice(hand)


def oracle_policy(hand: list[str], visible_tiles: list[str], rng: random.Random) -> str:
    ranked = _oracle_rank(hand, visible_tiles)
    return ranked[0][-1] if ranked else rng.choice(hand)


def shanten_policy(hand: list[str], visible_tiles: list[str], rng: random.Random) -> str:
    del visible_tiles
    counts = Counter(hand)
    count_tuple = _hand_counts_tuple(hand)
    rows: list[tuple[int, int, float, str]] = []
    for tile in counts:
        tile_index = TILE_TO_INDEX.get(tile)
        post = _estimate_post_discard_shanten_tuple(count_tuple, tile_index) if tile_index is not None else None
        raw_ukeire = _raw_ukeire_score(tile, counts)
        raw_score = _raw_discard_score(tile, counts, set())
        rows.append((post if post is not None else 99, -raw_ukeire, -raw_score, tile))
    if not rows:
        return rng.choice(hand)
    rows.sort()
    return rows[0][-1]


def random_policy(hand: list[str], visible_tiles: list[str], rng: random.Random) -> str:
    del visible_tiles
    return rng.choice(hand)


POLICIES: dict[str, Policy] = {
    "companion": companion_policy,
    "oracle": oracle_policy,
    "shanten": shanten_policy,
    "random": random_policy,
}


def simulate_game(
    seed: int,
    *,
    hero_policy: Policy,
    opponent_policy: Policy,
) -> GameResult:
    rng = random.Random(seed)
    deck = list(BASE_DECK)
    rng.shuffle(deck)
    hands = [deck[index * 13 : (index + 1) * 13] for index in range(4)]
    wall = deck[52:]
    visible_tiles: list[str] = []
    draw_count = 0

    while wall and draw_count < MAX_DRAWS:
        for player in range(4):
            if not wall or draw_count >= MAX_DRAWS:
                break
            hands[player].append(wall.pop(0))
            draw_count += 1
            if _is_agari(hands[player]):
                return _result(
                    winner=player,
                    win_type="tsumo",
                    turn_draws=draw_count,
                    hero_hand=hands[0],
                )

            policy = hero_policy if player == 0 else opponent_policy
            discard = policy(hands[player], visible_tiles, rng)
            if discard not in hands[player]:
                discard = hands[player][-1]
            hands[player].remove(discard)
            visible_tiles.append(discard)

            for target in range(4):
                if target == player:
                    continue
                candidate = [*hands[target], discard]
                if _is_agari(candidate):
                    return _result(
                        winner=target,
                        win_type="ron_dealt_by_hero" if player == 0 else "ron",
                        turn_draws=draw_count,
                        hero_hand=hands[0],
                    )

    return _result(winner=None, win_type="draw", turn_draws=draw_count, hero_hand=hands[0])


def run_batch(games: int, *, seed: int, hero_policy: Policy, opponent_policy: Policy) -> Aggregate:
    aggregate = Aggregate(games=games)
    for offset in range(games):
        result = simulate_game(seed + offset, hero_policy=hero_policy, opponent_policy=opponent_policy)
        aggregate.turn_draw_sum += result.turn_draws
        aggregate.hero_final_shanten_sum += result.hero_final_shanten
        if result.winner == 0:
            aggregate.hero_wins += 1
            aggregate.hero_tenpai_or_win += 1
            if result.win_type == "tsumo":
                aggregate.hero_tsumo += 1
            else:
                aggregate.hero_ron += 1
        elif result.winner is None:
            aggregate.draws += 1
            if result.hero_tenpai:
                aggregate.hero_tenpai_or_win += 1
        else:
            aggregate.opponent_wins += 1
            if result.win_type == "ron_dealt_by_hero":
                aggregate.dealt_in_by_hero += 1
    return aggregate


def _oracle_rank(hand: list[str], visible_tiles: list[str]) -> list[tuple[int, int, float, str]]:
    counts = Counter(hand)
    count_tuple = _hand_counts_tuple(hand)
    remaining = _remaining_tile_counts_tuple(count_tuple, visible_tiles)
    rows: list[tuple[int, int, float, str]] = []
    for tile in counts:
        tile_index = TILE_TO_INDEX.get(tile)
        post = _estimate_post_discard_shanten_tuple(count_tuple, tile_index) if tile_index is not None else None
        ukeire = _calculate_discard_ukeire_tuple(count_tuple, tile_index, tuple(remaining))
        if ukeire is None:
            ukeire = _raw_ukeire_score(tile, counts)
        raw_score = _raw_discard_score(tile, counts, set())
        rows.append((post if post is not None else 99, -(ukeire or 0), -raw_score, tile))
    rows.sort()
    return rows


def _is_agari(hand: list[str]) -> bool:
    shanten = _estimate_unclamped_shanten_tuple(_hand_counts_tuple(hand))
    return shanten is not None and shanten < 0


def _result(winner: int | None, win_type: str, turn_draws: int, hero_hand: list[str]) -> GameResult:
    shanten = _estimate_shanten_tuple(_hand_counts_tuple(hero_hand))
    final_shanten = shanten if shanten is not None else 9
    return GameResult(
        winner=winner,
        win_type=win_type,
        turn_draws=turn_draws,
        hero_final_shanten=final_shanten,
        hero_tenpai=final_shanten == 0,
    )


def _hand_counts_tuple(hand: list[str]) -> tuple[int, ...]:
    counts = [0] * len(TILES)
    for tile in hand:
        index = TILE_TO_INDEX.get(tile)
        if index is not None:
            counts[index] += 1
    return tuple(counts)


def _remaining_tile_counts_tuple(hand_counts: tuple[int, ...], visible_tiles: list[str]) -> list[int]:
    remaining = [4 - count for count in hand_counts]
    for tile in visible_tiles:
        index = TILE_TO_INDEX.get(tile)
        if index is not None:
            remaining[index] -= 1
    return [max(0, count) for count in remaining]


def _estimate_post_discard_shanten_tuple(hand_counts: tuple[int, ...], tile_index: int | None) -> int | None:
    if tile_index is None or hand_counts[tile_index] <= 0:
        return None
    after_discard = list(hand_counts)
    after_discard[tile_index] -= 1
    return _estimate_shanten_tuple(tuple(after_discard))


def _calculate_discard_ukeire_tuple(
    hand_counts: tuple[int, ...],
    tile_index: int | None,
    remaining_counts: tuple[int, ...],
) -> int | None:
    if tile_index is None or hand_counts[tile_index] <= 0:
        return None

    after_discard = list(hand_counts)
    after_discard[tile_index] -= 1
    after_tuple = tuple(after_discard)
    baseline_shanten = _estimate_unclamped_shanten_tuple(after_tuple)
    if baseline_shanten is None:
        return None

    ukeire = 0
    for draw_index, remaining in enumerate(remaining_counts):
        if remaining <= 0:
            continue
        drawn_counts = list(after_tuple)
        drawn_counts[draw_index] += 1
        shanten_after_draw = _estimate_unclamped_shanten_tuple(tuple(drawn_counts))
        if shanten_after_draw is not None and shanten_after_draw < baseline_shanten:
            ukeire += remaining
    return ukeire


def _estimate_shanten_tuple(counts: tuple[int, ...]) -> int | None:
    best = _estimate_unclamped_shanten_tuple(counts)
    if best is None:
        return None
    return max(0, min(8, best))


@lru_cache(maxsize=262144)
def _estimate_unclamped_shanten_tuple(counts: tuple[int, ...]) -> int | None:
    if not any(counts):
        return None
    return min(8, _estimate_raw_shanten_from_counts(list(counts)))


def _print_summary(aggregate: Aggregate) -> None:
    games = max(1, aggregate.games)
    print(f"games={aggregate.games}")
    print(f"hero_win_rate={aggregate.hero_wins / games:.2%} ({aggregate.hero_wins}/{aggregate.games})")
    print(f"opponent_win_rate={aggregate.opponent_wins / games:.2%} ({aggregate.opponent_wins}/{aggregate.games})")
    print(f"draw_rate={aggregate.draws / games:.2%} ({aggregate.draws}/{aggregate.games})")
    print(f"hero_tenpai_or_win={aggregate.hero_tenpai_or_win / games:.2%}")
    print(f"hero_deal_in_rate={aggregate.dealt_in_by_hero / games:.2%}")
    print(f"hero_tsumo={aggregate.hero_tsumo} hero_ron={aggregate.hero_ron}")
    print(f"avg_hero_final_shanten={aggregate.hero_final_shanten_sum / games:.2f}")
    print(f"avg_turn_draws={aggregate.turn_draw_sum / games:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy four-player Mahjong discard-policy simulator.")
    parser.add_argument("--games", type=int, default=80)
    parser.add_argument("--seed", type=int, default=92000)
    parser.add_argument("--hero-policy", choices=sorted(POLICIES), default="companion")
    parser.add_argument("--opponent-policy", choices=sorted(POLICIES), default="shanten")
    args = parser.parse_args()

    aggregate = run_batch(
        max(1, args.games),
        seed=args.seed,
        hero_policy=POLICIES[args.hero_policy],
        opponent_policy=POLICIES[args.opponent_policy],
    )
    print(f"hero_policy={args.hero_policy} opponent_policy={args.opponent_policy} seed={args.seed}")
    _print_summary(aggregate)


if __name__ == "__main__":
    main()
