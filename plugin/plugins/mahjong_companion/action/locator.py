from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..window_binding import WindowBindingResult
from .input_adapter import InputCommand


@dataclass(frozen=True)
class LocatedAction:
    target_x: int
    target_y: int
    source: str = "fixed_offset"
    button_region: dict[str, Any] | None = None
    click: bool = True

    def to_input_command(self) -> InputCommand:
        return InputCommand(
            target_x=self.target_x,
            target_y=self.target_y,
            click=self.click,
        )


class ActionLocator(Protocol):
    def locate(
        self,
        action_id: str,
        *,
        binding: WindowBindingResult,
        perceived: dict[str, Any] | None = None,
    ) -> LocatedAction | None:
        ...


class FixedOffsetLocator:
    def locate(
        self,
        action_id: str,
        *,
        binding: WindowBindingResult,
        perceived: dict[str, Any] | None = None,
    ) -> LocatedAction | None:
        if not binding.has_bounds():
            return None

        window_left = binding.left or 0
        window_top = binding.top or 0
        window_width = binding.width or 0
        window_height = binding.height or 0
        center_x = window_left + window_width // 2
        center_y = window_top + window_height // 2

        action_offsets: dict[str, tuple[int, int, bool]] = {
            "replay_next": (window_width // 2 + 150, window_height - 100, True),
            "replay_prev": (window_width // 2 - 150, window_height - 100, True),
            "dialog_confirm": (window_width // 2 + 100, window_height // 2 + 50, True),
            "dialog_cancel": (window_width // 2 - 100, window_height // 2 + 50, True),
            "menu_back": (50, 50, True),
            "menu_start_replay": (center_x - window_left, center_y - window_top + 100, True),
        }

        offset = action_offsets.get(action_id)
        if offset is None:
            return None

        dx, dy, click = offset
        return LocatedAction(
            target_x=window_left + dx,
            target_y=window_top + dy,
            click=click,
        )


class ButtonCandidateLocator:
    _ACTION_TO_BUTTON: dict[str, str] = {
        "ui_chi": "chi",
        "ui_pon": "pon",
        "ui_kan": "kan",
        "ui_riichi": "riichi",
        "ui_ron": "ron",
        "ui_tsumo": "tsumo",
        "ui_skip": "skip",
        "dialog_confirm": "confirm",
        "dialog_cancel": "cancel",
    }

    def locate(
        self,
        action_id: str,
        *,
        binding: WindowBindingResult,
        perceived: dict[str, Any] | None = None,
    ) -> LocatedAction | None:
        button_type = self._ACTION_TO_BUTTON.get(action_id)
        if button_type is None or not isinstance(perceived, dict):
            return None
        regions = perceived.get("button_regions", [])
        if not isinstance(regions, list):
            return None
        match = self._best_match(regions, button_type)
        if match is None:
            return None
        bbox = match.get("bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            return None
        try:
            left, top, right, bottom = [int(value) for value in bbox]
        except (TypeError, ValueError):
            return None
        if right <= left or bottom <= top:
            return None
        return LocatedAction(
            target_x=(left + right) // 2,
            target_y=(top + bottom) // 2,
            source="button_candidate",
            button_region=dict(match),
        )

    def _best_match(self, regions: list[Any], button_type: str) -> dict[str, Any] | None:
        matches: list[dict[str, Any]] = []
        for item in regions:
            if not isinstance(item, dict):
                continue
            if str(item.get("button_type", "")).strip() != button_type:
                continue
            matches.append(item)
        if not matches:
            return None
        return max(matches, key=lambda item: float(item.get("confidence", 0.0) or 0.0))


def input_command_from_located_action(located: LocatedAction | None) -> InputCommand | None:
    if located is None:
        return None
    return located.to_input_command()
