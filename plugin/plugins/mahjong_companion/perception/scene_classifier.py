from __future__ import annotations

from typing import Any


def classify_scene(metrics: dict[str, dict[str, Any]]) -> tuple[str, float, list[str], dict[str, bool]]:
    full = metrics["full_frame"]
    top = metrics["top_banner"]
    center = metrics["center_dialog"]
    bottom = metrics["bottom_action_bar"]
    hand = metrics["bottom_hand_area"]
    right = metrics["right_replay_panel"]
    dark_dialog = _looks_like_dark_dialog(center)
    bright_dialog = center["bright_ratio"] >= 0.24 and center["stddev"] <= 70.0
    bottom_bar_visible = _looks_like_bottom_action_bar(bottom)
    match_table_visible = _looks_like_match_table(full, top, hand)
    player_hand_visible = _looks_like_player_hand_strip(bottom, hand)
    blue_table_visible = _looks_like_blue_table_match(full, top, center)
    live_table_visible = _looks_like_live_table_with_hand(full, center, bottom)
    strong_live_match = player_hand_visible and (blue_table_visible or live_table_visible)

    roi_hits = {
        "top_banner": top["gold_ratio"] >= 0.03 or top["bright_ratio"] >= 0.18,
        "center_dialog": bright_dialog or dark_dialog,
        "bottom_action_bar": bottom_bar_visible,
        "right_replay_panel": right["dark_ratio"] >= 0.24 and right["bright_ratio"] >= 0.08,
    }
    notes: list[str] = []

    if (
        right["dark_ratio"] >= 0.42
        and right["bright_ratio"] >= 0.10
        and right["stddev"] <= 78.0
        and not strong_live_match
    ):
        notes.append("right replay panel signature detected")
        return "replay", 0.76, notes, roi_hits

    if top["gold_ratio"] >= 0.08 and full["dark_ratio"] >= 0.18:
        notes.append("gold-heavy top banner suggests result screen")
        return "result", 0.67, notes, roi_hits

    if dark_dialog and match_table_visible:
        notes.append("dark center dialog over match table detected")
        return "dialog", 0.82, notes, roi_hits

    if bright_dialog and bottom["orange_ratio"] < 0.02:
        notes.append("center dialog panel detected")
        return "dialog", 0.66, notes, roi_hits

    if _looks_like_lobby(full, top):
        if center["bright_ratio"] >= 0.18 or center["mean_luma"] >= 110:
            notes.append("bright lobby-style frame detected")
            return "lobby", 0.7, notes, roi_hits
        notes.append("bright menu-style frame detected")
        return "menu", 0.64, notes, roi_hits

    if _looks_like_room_setup_menu(full, top, center, bottom):
        notes.append("bright room setup menu detected")
        return "menu", 0.68, notes, roi_hits

    if bottom_bar_visible and match_table_visible and player_hand_visible:
        notes.append("bottom action bar over match table detected")
        return "in_match", 0.78, notes, roi_hits

    if match_table_visible and not bottom_bar_visible:
        notes.append("dark match table layout detected")
        return "in_match", 0.61, notes, roi_hits

    if player_hand_visible and blue_table_visible:
        notes.append("player hand strip over blue match table detected")
        return "in_match", 0.72, notes, roi_hits

    if player_hand_visible and live_table_visible:
        notes.append("player hand strip over active table detected")
        return "in_match", 0.72, notes, roi_hits

    notes.append("insufficient scene evidence")
    return "unknown", 0.22, notes, roi_hits


def _looks_like_dark_dialog(center: dict[str, Any]) -> bool:
    return (
        center["dark_ratio"] >= 0.55
        and center["stddev"] <= 48.0
        and center["colorful_ratio"] >= 0.16
        and (center["gold_ratio"] >= 0.012 or center["orange_ratio"] >= 0.012)
    )


def _looks_like_bottom_action_bar(bottom: dict[str, Any]) -> bool:
    return (
        bottom["orange_ratio"] >= 0.03
        or bottom["gold_ratio"] >= 0.04
        or bottom["red_ratio"] >= 0.015
        or (
            bottom["dark_ratio"] <= 0.58
            and bottom["colorful_ratio"] >= 0.14
            and (
                bottom["orange_ratio"] >= 0.015
                or bottom["gold_ratio"] >= 0.02
                or bottom["green_ratio"] >= 0.02
            )
        )
    )


def _looks_like_match_table(
    full: dict[str, Any],
    top: dict[str, Any],
    hand: dict[str, Any],
) -> bool:
    return (
        full["dark_ratio"] >= 0.5
        and full["colorful_ratio"] >= 0.08
        and full["colorful_ratio"] <= 0.28
        and top["dark_ratio"] >= 0.65
        and hand["dark_ratio"] >= 0.45
    )


def _looks_like_blue_table_match(
    full: dict[str, Any],
    top: dict[str, Any],
    center: dict[str, Any],
) -> bool:
    return (
        full["mean_luma"] >= 55
        and full["mean_luma"] <= 95
        and full["dark_ratio"] >= 0.35
        and full["colorful_ratio"] >= 0.55
        and top["dark_ratio"] >= 0.6
        and (top["gold_ratio"] >= 0.04 or top["orange_ratio"] >= 0.04)
        and center["bright_ratio"] <= 0.08
        and center["dark_ratio"] <= 0.2
    )


def _looks_like_live_table_with_hand(
    full: dict[str, Any],
    center: dict[str, Any],
    bottom: dict[str, Any],
) -> bool:
    return (
        full["mean_luma"] >= 55
        and full["mean_luma"] <= 100
        and full["dark_ratio"] >= 0.32
        and full["colorful_ratio"] >= 0.45
        and center["dark_ratio"] <= 0.3
        and center["bright_ratio"] <= 0.22
        and bottom["bright_ratio"] >= 0.12
        and bottom["colorful_ratio"] >= 0.45
    )


def _looks_like_player_hand_strip(
    bottom: dict[str, Any],
    hand: dict[str, Any],
) -> bool:
    return (
        hand["bright_ratio"] >= 0.12
        and hand["colorful_ratio"] >= 0.55
        and abs(bottom["dark_ratio"] - hand["dark_ratio"]) <= 0.08
        and abs(bottom["colorful_ratio"] - hand["colorful_ratio"]) <= 0.12
    )


def _looks_like_lobby(full: dict[str, Any], top: dict[str, Any]) -> bool:
    return (
        full["mean_luma"] >= 95
        and full["colorful_ratio"] >= 0.3
        and full["dark_ratio"] <= 0.35
        and (top["gold_ratio"] >= 0.05 or top["orange_ratio"] >= 0.06 or top["bright_ratio"] >= 0.08)
    )


def _looks_like_room_setup_menu(
    full: dict[str, Any],
    top: dict[str, Any],
    center: dict[str, Any],
    bottom: dict[str, Any],
) -> bool:
    return (
        full["mean_luma"] >= 90
        and full["dark_ratio"] <= 0.35
        and full["colorful_ratio"] >= 0.22
        and top["bright_ratio"] >= 0.12
        and top["dark_ratio"] <= 0.18
        and center["gold_ratio"] >= 0.05
        and center["orange_ratio"] >= 0.04
        and bottom["gold_ratio"] >= 0.06
        and bottom["green_ratio"] >= 0.04
    )
