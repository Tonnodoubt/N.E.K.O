from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "mahjong_companion": {
        "default_mode": "teaching",
        "sample_interval_ms": 300,
        "target_window_title_keywords": ["雀魂", "Mahjong Soul"],
        "capture": {
            "prefer_active_window": True,
            "save_format": "png",
        },
        "debug_samples": {
            "auto_prune_enabled": True,
            "max_frames": 180,
            "max_age_sec": 600,
            "prune_interval_sec": 15,
        },
        "fast_poll": {
            "enabled": True,
            "interval_ms": 120,
            "duration_sec": 7,
            "force_process": True,
            "trigger_buttons": ["chi", "pon", "kan", "skip"],
            "trigger_focuses": ["call_decision", "kan_decision", "confirm_or_skip"],
        },
        "frame_change_gate": {
            "enabled": True,
            "min_change_distance": 3,
            "stable_skip_limit": 300,
        },
        "perception": {
            "enabled": True,
            "debug_dump": True,
            "tile_classifier": {
                "backend": "vit",
                "enabled": True,
                "live_enabled": False,
                "require_accelerator": True,
                "hand_enabled": False,
                "discard_enabled": True,
                "model": "krmin/mahjong_soul_vision",
                "device": "auto",
                "top_k": 3,
                "min_confidence": 0.65,
                "min_mean_confidence": 0.70,
            },
        },
        "decision": {
            "enabled": True,
            "debug_dump": True,
            "preturn_planning_enabled": True,
        },
        "narration": {
            "enabled": True,
            "debug_dump": True,
        },
        "speech_policy": {
            "voice_enabled": True,
            "voice_mode": "key_events_only",
            "normal_channel": "silent_ui",
            "normal_voice_cooldown_sec": 18,
            "danger_voice_cooldown_sec": 5,
            "normal_notification_cooldown_sec": 18,
            "danger_notification_cooldown_sec": 5,
            "dedupe_window_sec": 8,
            "auto_dispatch_enabled": True,
            "target_lanlan": "",
        },
        "overlay": {
            "enabled": True,
            "auto_show_on_bind": True,
            "discard_marker_max_age_ms": 1500,
            "discard_marker_poll_interval_ms": 100,
            "discard_marker_region_change_threshold": 10,
            "fast_button_scan_enabled": True,
            "fast_button_scan_min_interval_ms": 60,
        },
        "game_agent_runtime": {
            "mode": "active",
        },
    }
}


def merge_runtime_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_runtime_config(merged[key], value)
        else:
            merged[key] = value
    return merged
