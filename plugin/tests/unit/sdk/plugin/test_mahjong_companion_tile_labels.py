from __future__ import annotations

import pytest

from plugin.plugins.mahjong_companion.tile_labels import (
    dedupe,
    format_tile_label,
    normalize_tile,
    normalize_tile_list,
    normalize_tile_set,
)


@pytest.mark.unit
def test_dedupe_preserves_first_occurrence_order() -> None:
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert dedupe([]) == []
    assert dedupe(["only"]) == ["only"]


@pytest.mark.unit
def test_normalize_tile_canonical_codes() -> None:
    assert normalize_tile("5m") == "5m"
    assert normalize_tile("9p") == "9p"
    assert normalize_tile("1s") == "1s"
    assert normalize_tile("7z") == "7z"


@pytest.mark.unit
def test_normalize_tile_red_five_aliases() -> None:
    # r5* (Tenhou-style) and 0* (Mahjong Soul export) both fold to canonical 5*.
    assert normalize_tile("r5m") == "5m"
    assert normalize_tile("R5p") == "5p"
    assert normalize_tile("R5S") == "5s"
    assert normalize_tile("0m") == "5m"
    assert normalize_tile("0p") == "5p"
    assert normalize_tile("0s") == "5s"


@pytest.mark.unit
def test_normalize_tile_honor_letter_aliases() -> None:
    assert normalize_tile("E") == "1z"
    assert normalize_tile("S") == "2z"
    assert normalize_tile("W") == "3z"
    assert normalize_tile("N") == "4z"
    assert normalize_tile("P") == "5z"
    assert normalize_tile("F") == "6z"
    assert normalize_tile("C") == "7z"


@pytest.mark.unit
def test_normalize_tile_returns_empty_for_garbage() -> None:
    assert normalize_tile("xx") == ""
    assert normalize_tile("") == ""
    assert normalize_tile(None) == ""
    assert normalize_tile("10m") == ""


@pytest.mark.unit
def test_normalize_tile_list_filters_invalid_entries() -> None:
    assert normalize_tile_list(["5m", "0p", "xx", "E"]) == ["5m", "5p", "1z"]
    assert normalize_tile_list("not a list") == []
    assert normalize_tile_list([]) == []


@pytest.mark.unit
def test_normalize_tile_set_dedupes_after_normalisation() -> None:
    assert normalize_tile_set(["5m", "5m", "0m"]) == {"5m"}
    assert normalize_tile_set("not a list") == set()


@pytest.mark.unit
def test_format_tile_label_handles_red_five_from_both_aliases() -> None:
    # Both r5m and 0m should render as 红五万 / 红五.
    assert "红五" in format_tile_label("r5m")
    assert "红五" in format_tile_label("0m")
    assert "五万" == format_tile_label("5m")
