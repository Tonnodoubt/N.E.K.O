from __future__ import annotations

import re
from typing import Any

_NUMERAL_LABELS = {
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}
_SUIT_LABELS = {
    "m": "万",
    "p": "筒",
    "s": "索",
}
_HONOR_LABELS = {
    "1": "东风",
    "2": "南风",
    "3": "西风",
    "4": "北风",
    "5": "白板",
    "6": "发财",
    "7": "红中",
}
_TILE_CODE_RE = re.compile(r"(?<![0-9A-Za-z])(?:R5|r5|0|[1-9])[mpsz](?![0-9A-Za-z])")


def format_tile_label(tile: Any, *, include_code: bool = False) -> str:
    value = str(tile).strip()
    if not value:
        return ""

    normalized, red_five = _normalize_label_tile(value)
    if len(normalized) != 2:
        return value

    number = normalized[0]
    suit = normalized[1]
    if suit in _SUIT_LABELS and number in _NUMERAL_LABELS:
        label = f"{'红五' if red_five else _NUMERAL_LABELS[number]}{_SUIT_LABELS[suit]}"
    elif suit == "z" and number in _HONOR_LABELS:
        label = _HONOR_LABELS[number]
    else:
        return value

    return f"{label} ({normalized})" if include_code else label


def replace_tile_codes_in_text(text: Any) -> str:
    value = str(text).strip()
    if not value:
        return ""
    return _TILE_CODE_RE.sub(lambda match: format_tile_label(match.group(0)), value)


def _normalize_label_tile(tile: str) -> tuple[str, bool]:
    value = tile.strip()
    lower = value.lower()
    if len(lower) == 3 and lower[:2] == "r5" and lower[2] in _SUIT_LABELS:
        return f"5{lower[2]}", True
    if len(lower) == 2 and lower[0] == "0" and lower[1] in _SUIT_LABELS:
        return f"5{lower[1]}", True

    honor_aliases = {
        "E": "1z",
        "S": "2z",
        "W": "3z",
        "N": "4z",
        "P": "5z",
        "F": "6z",
        "C": "7z",
    }
    aliased = honor_aliases.get(value.upper())
    if aliased:
        return aliased, False

    if len(lower) == 2 and lower[0].isdigit() and lower[1] in {"m", "p", "s", "z"}:
        return lower, False
    return value, False


_HONOR_ALIASES = {
    "E": "1z",
    "S": "2z",
    "W": "3z",
    "N": "4z",
    "P": "5z",
    "F": "6z",
    "C": "7z",
}
_TILE_SUITS = {"m", "p", "s"}


def normalize_tile(tile: Any) -> str:
    """Normalize tile code to canonical `Ns` form (e.g. `5m`, `7z`).

    Accepts:
    - canonical codes like `5m`, `9p`, `7z`
    - red-five aliases `r5m` / `R5m` and `0m` / `0p` / `0s` (Mahjong Soul / Tenhou)
    - honor letter aliases E/S/W/N/P/F/C

    Returns empty string for unrecognised inputs.
    """
    value = str(tile).strip()
    lower = value.lower()
    if len(lower) == 3 and lower[0] == "r" and lower[1] == "5" and lower[2] in _TILE_SUITS:
        return f"5{lower[2]}"
    if len(lower) == 2 and lower[0] == "0" and lower[1] in _TILE_SUITS:
        return f"5{lower[1]}"
    aliased = _HONOR_ALIASES.get(value.upper())
    if aliased:
        return aliased
    if len(lower) == 2 and lower[0].isdigit() and lower[1] in {"m", "p", "s", "z"}:
        return lower
    return ""


def normalize_tile_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [tile for tile in (normalize_tile(item) for item in value) if tile]


def normalize_tile_set(value: Any) -> set[str]:
    return set(normalize_tile_list(value))


def dedupe(items: list[str]) -> list[str]:
    """Order-preserving deduplication. Shared single source of truth.

    Used across review / decision / perception modules to keep alert and
    advice strings unique without losing the first-occurrence order.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
