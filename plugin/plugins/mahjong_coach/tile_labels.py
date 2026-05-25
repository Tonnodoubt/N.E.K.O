from __future__ import annotations


RED_FIVE_ALIASES = {
    "0m": "0m",
    "0p": "0p",
    "0s": "0s",
    "R5m": "0m",
    "R5p": "0p",
    "R5s": "0s",
    "r5m": "0m",
    "r5p": "0p",
    "r5s": "0s",
}


def normalize_tile(tile: str) -> str:
    value = str(tile or "").strip()
    if not value:
        return ""
    if value in RED_FIVE_ALIASES:
        return RED_FIVE_ALIASES[value]
    if len(value) == 2 and value[0] in "123456789" and value[1] in "mpsz":
        return value
    return value


def tile_suit(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in {"0m", "0p", "0s"}:
        return normalized[1]
    return normalized[1] if len(normalized) == 2 else ""


def tile_rank(tile: str) -> str:
    normalized = normalize_tile(tile)
    if normalized in {"0m", "0p", "0s"}:
        return "5"
    return normalized[0] if len(normalized) == 2 else ""


def is_honor(tile: str) -> bool:
    return tile_suit(tile) == "z"


def is_terminal(tile: str) -> bool:
    return tile_rank(tile) in {"1", "9"} and tile_suit(tile) in {"m", "p", "s"}


def is_simple(tile: str) -> bool:
    return tile_rank(tile) in {"2", "3", "4", "5", "6", "7", "8"} and tile_suit(tile) in {"m", "p", "s"}


def hand_signature(tiles: list[str]) -> str:
    normalized = sorted(normalize_tile(tile) for tile in tiles if normalize_tile(tile))
    return "|".join(normalized)

