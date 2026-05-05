from __future__ import annotations

from plugin.plugins.mahjong_companion.action.locator import (
    ButtonCandidateLocator,
    FixedOffsetLocator,
    input_command_from_located_action,
)
from plugin.plugins.mahjong_companion.window_binding import WindowBindingResult


def _binding() -> WindowBindingResult:
    return WindowBindingResult(
        bound=True,
        window_title="Mahjong Soul",
        left=100,
        top=200,
        width=800,
        height=600,
    )


def test_fixed_offset_locator_returns_replay_next_coordinate() -> None:
    located = FixedOffsetLocator().locate("replay_next", binding=_binding())

    assert located is not None
    assert located.target_x == 650
    assert located.target_y == 700
    assert located.source == "fixed_offset"
    assert located.click is True


def test_fixed_offset_locator_returns_none_for_unknown_action() -> None:
    located = FixedOffsetLocator().locate("unknown_action", binding=_binding())

    assert located is None


def test_fixed_offset_locator_returns_none_without_window_bounds() -> None:
    located = FixedOffsetLocator().locate("replay_next", binding=WindowBindingResult(bound=True))

    assert located is None


def test_button_candidate_locator_returns_region_center() -> None:
    located = ButtonCandidateLocator().locate(
        "ui_riichi",
        binding=_binding(),
        perceived={
            "button_regions": [
                {
                    "button_type": "riichi",
                    "bbox": [300, 420, 500, 480],
                    "confidence": 0.93,
                    "template_id": "riichi",
                }
            ],
        },
    )

    assert located is not None
    assert located.target_x == 400
    assert located.target_y == 450
    assert located.source == "button_candidate"
    assert located.button_region is not None
    assert located.button_region["template_id"] == "riichi"


def test_button_candidate_locator_maps_ui_pon_to_pon_region() -> None:
    located = ButtonCandidateLocator().locate(
        "ui_pon",
        binding=_binding(),
        perceived={
            "button_regions": [
                {
                    "button_type": "pon",
                    "bbox": [840, 750, 1120, 890],
                    "confidence": 0.95,
                    "template_id": "pon",
                },
                {
                    "button_type": "skip",
                    "bbox": [1180, 750, 1540, 890],
                    "confidence": 0.94,
                    "template_id": "skip",
                },
            ],
        },
    )

    assert located is not None
    assert located.target_x == 980
    assert located.target_y == 820
    assert located.source == "button_candidate"
    assert located.button_region is not None
    assert located.button_region["button_type"] == "pon"


def test_button_candidate_locator_returns_none_when_unmatched() -> None:
    located = ButtonCandidateLocator().locate(
        "ui_riichi",
        binding=_binding(),
        perceived={"button_regions": []},
    )

    assert located is None


def test_input_command_from_located_action_converts_location() -> None:
    located = FixedOffsetLocator().locate("menu_back", binding=_binding())

    command = input_command_from_located_action(located)

    assert command is not None
    assert command.target_x == 150
    assert command.target_y == 250
    assert command.click is True
