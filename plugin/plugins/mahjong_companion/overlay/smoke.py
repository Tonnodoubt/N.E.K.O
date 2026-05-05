from __future__ import annotations

import sys
import time

from . import CompanionOverlay


def main() -> int:
    overlay = CompanionOverlay()
    if not overlay.start():
        print("overlay unsupported on this platform/display")
        return 1

    time.sleep(0.25)
    if not overlay.is_running():
        print("overlay process exited before first status update")
        return 1

    overlay.update_status(
        {
            "window_bound": True,
            "last_scene": "in_match",
            "last_is_user_turn": True,
            "runtime_mode": "active",
            "runtime_status": "scanning",
            "window_left": 120,
            "window_top": 120,
            "window_width": 1280,
            "last_decision": {
                "decision_type": "tile_efficiency_hint",
                "mahjong_analysis": {
                    "candidate_discards": [
                        {
                            "tile": "5z",
                            "reason": "overlay smoke",
                            "ukeire_estimate": 8,
                        }
                    ]
                },
            },
            "screen_overlays": [
                {
                    "kind": "discard_recommendation",
                    "box": {"left": 360, "top": 640, "width": 72, "height": 96},
                }
            ],
        }
    )
    time.sleep(3.0)
    overlay.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

