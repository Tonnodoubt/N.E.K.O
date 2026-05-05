from __future__ import annotations

from plugin.plugins.mahjong_companion.contracts import PerceivedGameState
from plugin.plugins.mahjong_companion.decision.generator import build_decision
from plugin.plugins.mahjong_companion.overlay import _advice_view, _reason_line, _render_screen_markers
from plugin.plugins.mahjong_companion.tile_labels import format_tile_label


class _FakeMarker:
    def __init__(self) -> None:
        self.geometry_value = ""
        self.visible = False
        self.lifted = False

    def geometry(self, value: str) -> None:
        self.geometry_value = value

    def deiconify(self) -> None:
        self.visible = True

    def lift(self) -> None:
        self.lifted = True

    def withdraw(self) -> None:
        self.visible = False


class _FakeCanvas:
    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.rectangles: list[tuple[int, int, int, int, str, int]] = []

    def config(self, *, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def delete(self, _target: str) -> None:
        self.rectangles.clear()

    def create_rectangle(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        outline: str,
        width: int,
    ) -> None:
        self.rectangles.append((x1, y1, x2, y2, outline, width))


def test_overlay_advice_prioritizes_call_action_over_tile_candidate() -> None:
    state = PerceivedGameState(
        scene="in_match",
        confidence=0.86,
        buttons=["chi", "skip"],
        hand_tiles=[
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "3s",
            "4s",
            "5s",
            "7s",
            "8s",
            "2z",
            "2z",
        ],
        analysis_hints={
            "tile_level_available": True,
            "recognized_hand_tile_count": 13,
            "shanten_estimate": 2,
        },
    )

    decision = build_decision(state)
    view = _advice_view(
        {
            "last_decision": decision.to_dict(),
            "last_decision_type": decision.decision_type,
            "window_bound": True,
        }
    )
    top_tile = decision.mahjong_analysis["candidate_discards"][0]["tile"]

    assert view["primary"] != format_tile_label(top_tile)
    assert view["reason"] == _reason_line(decision.suggestion)


def test_overlay_advice_surfaces_meld_selection_prompt() -> None:
    view = _advice_view(
        {
            "window_bound": True,
            "last_decision": {
                "decision_type": "meld_selection",
                "recommended_focus": "meld_selection",
                "suggestion": "choose 3m 4m",
                "engine_meta": {
                    "screen_overlays": [
                        {
                            "kind": "meld_selection_recommendation",
                            "box": {"left": 100, "top": 200, "width": 40, "height": 50},
                        }
                    ],
                },
            },
        }
    )

    assert view == {"primary": "选牌", "reason": "choose 三万 四万"}


def test_screen_marker_renders_meld_selection_boxes() -> None:
    marker = _FakeMarker()
    canvas = _FakeCanvas()

    _render_screen_markers(
        marker,
        canvas,
        {
            "screen_overlays": [
                {
                    "kind": "meld_selection_recommendation",
                    "box": {"left": 100, "top": 200, "width": 40, "height": 50},
                }
            ],
        },
    )

    assert marker.visible is True
    assert marker.lifted is True
    assert marker.geometry_value == "56x66+92+192"
    assert canvas.rectangles[0] == (8, 8, 48, 58, "#ff2d2d", 5)
