from __future__ import annotations

from plugin.plugins.mahjong_companion.perception.action_detector import detect_actions
from plugin.plugins.mahjong_companion.perception.scene_classifier import classify_scene


def test_classify_scene_detects_dark_match_dialog() -> None:
    metrics = _build_metrics(
        full_frame={
            "mean_luma": 54.43,
            "stddev": 42.47,
            "bright_ratio": 0.0045,
            "dark_ratio": 0.8178,
            "white_ratio": 0.0,
            "colorful_ratio": 0.1646,
            "gold_ratio": 0.0144,
            "orange_ratio": 0.016,
            "red_ratio": 0.0136,
            "green_ratio": 0.0005,
        },
        top_banner={
            "mean_luma": 44.08,
            "stddev": 28.42,
            "bright_ratio": 0.0,
            "dark_ratio": 0.9251,
            "white_ratio": 0.0,
            "colorful_ratio": 0.229,
            "gold_ratio": 0.0,
            "orange_ratio": 0.0015,
            "red_ratio": 0.0492,
            "green_ratio": 0.0,
        },
        center_dialog={
            "mean_luma": 50.22,
            "stddev": 33.23,
            "bright_ratio": 0.0183,
            "dark_ratio": 0.8636,
            "white_ratio": 0.0,
            "colorful_ratio": 0.283,
            "gold_ratio": 0.0297,
            "orange_ratio": 0.0297,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
        bottom_action_bar={
            "mean_luma": 74.16,
            "stddev": 46.98,
            "bright_ratio": 0.0,
            "dark_ratio": 0.6777,
            "white_ratio": 0.0,
            "colorful_ratio": 0.199,
            "gold_ratio": 0.0003,
            "orange_ratio": 0.0,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
        bottom_hand_area={
            "mean_luma": 69.04,
            "stddev": 47.12,
            "bright_ratio": 0.0,
            "dark_ratio": 0.7129,
            "white_ratio": 0.0,
            "colorful_ratio": 0.1397,
            "gold_ratio": 0.0171,
            "orange_ratio": 0.0092,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
        right_replay_panel={
            "mean_luma": 48.51,
            "stddev": 37.42,
            "bright_ratio": 0.0,
            "dark_ratio": 0.8429,
            "white_ratio": 0.0,
            "colorful_ratio": 0.1002,
            "gold_ratio": 0.0003,
            "orange_ratio": 0.027,
            "red_ratio": 0.0273,
            "green_ratio": 0.0,
        },
    )

    scene, confidence, notes, roi_hits = classify_scene(metrics)

    assert scene == "dialog"
    assert confidence >= 0.8
    assert roi_hits["center_dialog"] is True
    assert any("dialog" in note for note in notes)


def test_classify_scene_keeps_lobby_out_of_match() -> None:
    metrics = _build_metrics(
        full_frame={
            "mean_luma": 127.03,
            "stddev": 64.27,
            "bright_ratio": 0.1504,
            "dark_ratio": 0.175,
            "white_ratio": 0.0392,
            "colorful_ratio": 0.5011,
            "gold_ratio": 0.0358,
            "orange_ratio": 0.0594,
            "red_ratio": 0.0448,
            "green_ratio": 0.0133,
        },
        top_banner={
            "mean_luma": 129.33,
            "stddev": 67.68,
            "bright_ratio": 0.0888,
            "dark_ratio": 0.1903,
            "white_ratio": 0.0248,
            "colorful_ratio": 0.6067,
            "gold_ratio": 0.1105,
            "orange_ratio": 0.1157,
            "red_ratio": 0.013,
            "green_ratio": 0.0,
        },
        center_dialog={
            "mean_luma": 147.41,
            "stddev": 62.89,
            "bright_ratio": 0.2962,
            "dark_ratio": 0.1042,
            "white_ratio": 0.0725,
            "colorful_ratio": 0.6074,
            "gold_ratio": 0.0181,
            "orange_ratio": 0.0286,
            "red_ratio": 0.0121,
            "green_ratio": 0.0,
        },
        bottom_action_bar={
            "mean_luma": 104.39,
            "stddev": 58.23,
            "bright_ratio": 0.0846,
            "dark_ratio": 0.3195,
            "white_ratio": 0.0178,
            "colorful_ratio": 0.2527,
            "gold_ratio": 0.0257,
            "orange_ratio": 0.0401,
            "red_ratio": 0.02,
            "green_ratio": 0.0229,
        },
        bottom_hand_area={
            "mean_luma": 111.1,
            "stddev": 59.96,
            "bright_ratio": 0.0889,
            "dark_ratio": 0.2852,
            "white_ratio": 0.0222,
            "colorful_ratio": 0.3575,
            "gold_ratio": 0.0323,
            "orange_ratio": 0.0659,
            "red_ratio": 0.0531,
            "green_ratio": 0.0247,
        },
        right_replay_panel={
            "mean_luma": 139.74,
            "stddev": 59.48,
            "bright_ratio": 0.2035,
            "dark_ratio": 0.0831,
            "white_ratio": 0.0446,
            "colorful_ratio": 0.6562,
            "gold_ratio": 0.0782,
            "orange_ratio": 0.0981,
            "red_ratio": 0.0407,
            "green_ratio": 0.0113,
        },
    )

    scene, confidence, notes, _ = classify_scene(metrics)

    assert scene == "lobby"
    assert confidence >= 0.7
    assert all("match" not in note for note in notes)


def test_classify_scene_keeps_dark_non_game_window_unknown() -> None:
    metrics = _build_metrics(
        full_frame={
            "mean_luma": 38.95,
            "stddev": 40.24,
            "bright_ratio": 0.0372,
            "dark_ratio": 0.914,
            "white_ratio": 0.0002,
            "colorful_ratio": 0.0151,
            "gold_ratio": 0.0032,
            "orange_ratio": 0.0033,
            "red_ratio": 0.0,
            "green_ratio": 0.0046,
        },
        top_banner={
            "mean_luma": 38.46,
            "stddev": 36.41,
            "bright_ratio": 0.0118,
            "dark_ratio": 0.906,
            "white_ratio": 0.001,
            "colorful_ratio": 0.0333,
            "gold_ratio": 0.0,
            "orange_ratio": 0.0007,
            "red_ratio": 0.0002,
            "green_ratio": 0.0075,
        },
        center_dialog={
            "mean_luma": 36.32,
            "stddev": 29.79,
            "bright_ratio": 0.0175,
            "dark_ratio": 0.9461,
            "white_ratio": 0.0,
            "colorful_ratio": 0.0123,
            "gold_ratio": 0.0052,
            "orange_ratio": 0.0054,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
        bottom_action_bar={
            "mean_luma": 51.53,
            "stddev": 58.52,
            "bright_ratio": 0.0904,
            "dark_ratio": 0.8603,
            "white_ratio": 0.0,
            "colorful_ratio": 0.0094,
            "gold_ratio": 0.0,
            "orange_ratio": 0.0,
            "red_ratio": 0.0,
            "green_ratio": 0.0039,
        },
        bottom_hand_area={
            "mean_luma": 47.93,
            "stddev": 53.93,
            "bright_ratio": 0.0721,
            "dark_ratio": 0.8642,
            "white_ratio": 0.0,
            "colorful_ratio": 0.0131,
            "gold_ratio": 0.0,
            "orange_ratio": 0.0,
            "red_ratio": 0.0,
            "green_ratio": 0.0092,
        },
        right_replay_panel={
            "mean_luma": 43.99,
            "stddev": 50.66,
            "bright_ratio": 0.0571,
            "dark_ratio": 0.8848,
            "white_ratio": 0.0,
            "colorful_ratio": 0.0,
            "gold_ratio": 0.0,
            "orange_ratio": 0.0,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
    )

    scene, confidence, notes, _ = classify_scene(metrics)

    assert scene == "unknown"
    assert confidence == 0.22
    assert notes == ["insufficient scene evidence"]


def test_classify_scene_detects_live_blue_table_match() -> None:
    metrics = _build_metrics(
        full_frame={
            "mean_luma": 77.47,
            "stddev": 54.26,
            "bright_ratio": 0.07,
            "dark_ratio": 0.4909,
            "white_ratio": 0.0033,
            "colorful_ratio": 0.7312,
            "gold_ratio": 0.0197,
            "orange_ratio": 0.034,
            "red_ratio": 0.0012,
            "green_ratio": 0.0007,
        },
        top_banner={
            "mean_luma": 61.64,
            "stddev": 40.6,
            "bright_ratio": 0.0019,
            "dark_ratio": 0.7395,
            "white_ratio": 0.0019,
            "colorful_ratio": 0.823,
            "gold_ratio": 0.0662,
            "orange_ratio": 0.0665,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
        center_dialog={
            "mean_luma": 82.53,
            "stddev": 32.02,
            "bright_ratio": 0.0199,
            "dark_ratio": 0.1089,
            "white_ratio": 0.0033,
            "colorful_ratio": 0.7917,
            "gold_ratio": 0.0083,
            "orange_ratio": 0.0047,
            "red_ratio": 0.0022,
            "green_ratio": 0.0,
        },
        bottom_action_bar={
            "mean_luma": 107.53,
            "stddev": 67.92,
            "bright_ratio": 0.2146,
            "dark_ratio": 0.3719,
            "white_ratio": 0.0022,
            "colorful_ratio": 0.6271,
            "gold_ratio": 0.001,
            "orange_ratio": 0.0372,
            "red_ratio": 0.0,
            "green_ratio": 0.0003,
        },
        bottom_hand_area={
            "mean_luma": 97.74,
            "stddev": 63.02,
            "bright_ratio": 0.1755,
            "dark_ratio": 0.3815,
            "white_ratio": 0.0018,
            "colorful_ratio": 0.7024,
            "gold_ratio": 0.0035,
            "orange_ratio": 0.0044,
            "red_ratio": 0.002,
            "green_ratio": 0.0,
        },
        right_replay_panel={
            "mean_luma": 68.69,
            "stddev": 52.95,
            "bright_ratio": 0.0396,
            "dark_ratio": 0.7988,
            "white_ratio": 0.0034,
            "colorful_ratio": 0.7847,
            "gold_ratio": 0.0312,
            "orange_ratio": 0.091,
            "red_ratio": 0.0013,
            "green_ratio": 0.0,
        },
    )

    scene, confidence, notes, _ = classify_scene(metrics)

    assert scene == "in_match"
    assert confidence >= 0.7
    assert any("match" in note for note in notes)


def test_classify_scene_keeps_midgame_live_table_out_of_unknown() -> None:
    metrics = _build_metrics(
        full_frame={
            "mean_luma": 84.67,
            "stddev": 59.43,
            "bright_ratio": 0.0935,
            "dark_ratio": 0.4362,
            "white_ratio": 0.0037,
            "colorful_ratio": 0.7018,
            "gold_ratio": 0.0229,
            "orange_ratio": 0.0542,
            "red_ratio": 0.0013,
            "green_ratio": 0.001,
        },
        top_banner={
            "mean_luma": 77.94,
            "stddev": 52.72,
            "bright_ratio": 0.0174,
            "dark_ratio": 0.431,
            "white_ratio": 0.0081,
            "colorful_ratio": 0.8249,
            "gold_ratio": 0.0742,
            "orange_ratio": 0.1813,
            "red_ratio": 0.0012,
            "green_ratio": 0.0,
        },
        center_dialog={
            "mean_luma": 98.69,
            "stddev": 53.19,
            "bright_ratio": 0.1098,
            "dark_ratio": 0.1388,
            "white_ratio": 0.0074,
            "colorful_ratio": 0.6011,
            "gold_ratio": 0.0093,
            "orange_ratio": 0.012,
            "red_ratio": 0.0021,
            "green_ratio": 0.0,
        },
        bottom_action_bar={
            "mean_luma": 108.91,
            "stddev": 66.41,
            "bright_ratio": 0.2215,
            "dark_ratio": 0.3045,
            "white_ratio": 0.0011,
            "colorful_ratio": 0.6213,
            "gold_ratio": 0.0016,
            "orange_ratio": 0.0239,
            "red_ratio": 0.0003,
            "green_ratio": 0.0016,
        },
        bottom_hand_area={
            "mean_luma": 98.67,
            "stddev": 61.53,
            "bright_ratio": 0.1742,
            "dark_ratio": 0.3489,
            "white_ratio": 0.0015,
            "colorful_ratio": 0.7115,
            "gold_ratio": 0.0041,
            "orange_ratio": 0.0166,
            "red_ratio": 0.0014,
            "green_ratio": 0.0007,
        },
        right_replay_panel={
            "mean_luma": 70.53,
            "stddev": 52.71,
            "bright_ratio": 0.0414,
            "dark_ratio": 0.7582,
            "white_ratio": 0.0024,
            "colorful_ratio": 0.7894,
            "gold_ratio": 0.0363,
            "orange_ratio": 0.095,
            "red_ratio": 0.0011,
            "green_ratio": 0.0,
        },
    )

    scene, confidence, notes, _ = classify_scene(metrics)

    assert scene == "in_match"
    assert confidence >= 0.7
    assert any("player hand strip" in note for note in notes)


def test_classify_scene_prefers_live_table_over_false_replay_signature() -> None:
    metrics = _build_metrics(
        full_frame={
            "mean_luma": 91.11,
            "stddev": 64.47,
            "bright_ratio": 0.1363,
            "dark_ratio": 0.4479,
            "white_ratio": 0.0063,
            "colorful_ratio": 0.6286,
            "gold_ratio": 0.0225,
            "orange_ratio": 0.0504,
            "red_ratio": 0.0011,
            "green_ratio": 0.001,
        },
        top_banner={
            "mean_luma": 112.78,
            "stddev": 72.03,
            "bright_ratio": 0.2099,
            "dark_ratio": 0.2974,
            "white_ratio": 0.009,
            "colorful_ratio": 0.6308,
            "gold_ratio": 0.0744,
            "orange_ratio": 0.1595,
            "red_ratio": 0.0013,
            "green_ratio": 0.0,
        },
        center_dialog={
            "mean_luma": 107.75,
            "stddev": 64.39,
            "bright_ratio": 0.19,
            "dark_ratio": 0.2547,
            "white_ratio": 0.0208,
            "colorful_ratio": 0.4165,
            "gold_ratio": 0.0087,
            "orange_ratio": 0.0141,
            "red_ratio": 0.0025,
            "green_ratio": 0.0,
        },
        bottom_action_bar={
            "mean_luma": 96.34,
            "stddev": 60.81,
            "bright_ratio": 0.1631,
            "dark_ratio": 0.3936,
            "white_ratio": 0.001,
            "colorful_ratio": 0.702,
            "gold_ratio": 0.0015,
            "orange_ratio": 0.0179,
            "red_ratio": 0.0003,
            "green_ratio": 0.0007,
        },
        bottom_hand_area={
            "mean_luma": 92.09,
            "stddev": 57.58,
            "bright_ratio": 0.1424,
            "dark_ratio": 0.3957,
            "white_ratio": 0.002,
            "colorful_ratio": 0.7524,
            "gold_ratio": 0.0038,
            "orange_ratio": 0.0138,
            "red_ratio": 0.0012,
            "green_ratio": 0.0004,
        },
        right_replay_panel={
            "mean_luma": 85.37,
            "stddev": 66.95,
            "bright_ratio": 0.1391,
            "dark_ratio": 0.6837,
            "white_ratio": 0.0025,
            "colorful_ratio": 0.6061,
            "gold_ratio": 0.0363,
            "orange_ratio": 0.095,
            "red_ratio": 0.0011,
            "green_ratio": 0.0,
        },
    )

    scene, confidence, notes, _ = classify_scene(metrics)

    assert scene == "in_match"
    assert confidence >= 0.7
    assert any("player hand strip" in note for note in notes)


def test_classify_scene_detects_room_setup_menu() -> None:
    metrics = _build_metrics(
        full_frame={
            "mean_luma": 100.92,
            "stddev": 57.78,
            "bright_ratio": 0.0632,
            "dark_ratio": 0.3091,
            "white_ratio": 0.0172,
            "colorful_ratio": 0.2804,
            "gold_ratio": 0.046,
            "orange_ratio": 0.037,
            "red_ratio": 0.0007,
            "green_ratio": 0.0108,
        },
        top_banner={
            "mean_luma": 143.69,
            "stddev": 61.52,
            "bright_ratio": 0.1538,
            "dark_ratio": 0.0919,
            "white_ratio": 0.0297,
            "colorful_ratio": 0.5984,
            "gold_ratio": 0.0077,
            "orange_ratio": 0.0037,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
        center_dialog={
            "mean_luma": 92.14,
            "stddev": 50.79,
            "bright_ratio": 0.0442,
            "dark_ratio": 0.2156,
            "white_ratio": 0.0326,
            "colorful_ratio": 0.3308,
            "gold_ratio": 0.0728,
            "orange_ratio": 0.0594,
            "red_ratio": 0.0009,
            "green_ratio": 0.0,
        },
        bottom_action_bar={
            "mean_luma": 116.63,
            "stddev": 53.39,
            "bright_ratio": 0.0401,
            "dark_ratio": 0.2212,
            "white_ratio": 0.0079,
            "colorful_ratio": 0.3586,
            "gold_ratio": 0.0935,
            "orange_ratio": 0.0817,
            "red_ratio": 0.0111,
            "green_ratio": 0.0693,
        },
        bottom_hand_area={
            "mean_luma": 104.0,
            "stddev": 51.74,
            "bright_ratio": 0.0348,
            "dark_ratio": 0.3215,
            "white_ratio": 0.0066,
            "colorful_ratio": 0.2958,
            "gold_ratio": 0.0708,
            "orange_ratio": 0.0608,
            "red_ratio": 0.001,
            "green_ratio": 0.0482,
        },
        right_replay_panel={
            "mean_luma": 90.62,
            "stddev": 51.85,
            "bright_ratio": 0.0905,
            "dark_ratio": 0.3546,
            "white_ratio": 0.0223,
            "colorful_ratio": 0.107,
            "gold_ratio": 0.0024,
            "orange_ratio": 0.0018,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
    )

    scene, confidence, notes, _ = classify_scene(metrics)

    assert scene == "menu"
    assert confidence >= 0.65
    assert any("menu" in note for note in notes)


def test_detect_actions_uses_dialog_signal_instead_of_false_bottom_confirm() -> None:
    metrics = _build_metrics(
        bottom_action_bar={
            "dark_ratio": 0.6777,
            "colorful_ratio": 0.199,
            "gold_ratio": 0.0003,
            "orange_ratio": 0.0,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
        },
        center_dialog={
            "dark_ratio": 0.8636,
            "stddev": 33.23,
            "colorful_ratio": 0.283,
            "gold_ratio": 0.0297,
            "orange_ratio": 0.0297,
            "bright_ratio": 0.0183,
        },
        right_replay_panel={
            "dark_ratio": 0.8429,
        },
    )

    buttons, is_user_turn, notes = detect_actions("dialog", metrics)

    assert buttons == ["confirm"]
    assert is_user_turn is True
    assert any("dialog" in note for note in notes)


def test_detect_actions_suppresses_match_buttons_in_lobby() -> None:
    metrics = _build_metrics(
        bottom_action_bar={
            "dark_ratio": 0.3195,
            "colorful_ratio": 0.2527,
            "gold_ratio": 0.0257,
            "orange_ratio": 0.0401,
            "red_ratio": 0.02,
            "green_ratio": 0.0229,
        },
        center_dialog={
            "dark_ratio": 0.1042,
            "stddev": 62.89,
            "colorful_ratio": 0.6074,
            "gold_ratio": 0.0181,
            "orange_ratio": 0.0286,
            "bright_ratio": 0.2962,
        },
        right_replay_panel={
            "dark_ratio": 0.0831,
        },
    )

    buttons, is_user_turn, notes = detect_actions("lobby", metrics)

    assert buttons == []
    assert is_user_turn is False
    assert any("suppresses" in note for note in notes)


def test_detect_actions_ignores_false_skip_from_tile_strip() -> None:
    metrics = _build_metrics(
        bottom_action_bar={
            "mean_luma": 107.53,
            "stddev": 67.92,
            "bright_ratio": 0.2146,
            "dark_ratio": 0.3719,
            "white_ratio": 0.0022,
            "colorful_ratio": 0.6271,
            "gold_ratio": 0.001,
            "orange_ratio": 0.0372,
            "red_ratio": 0.0,
            "green_ratio": 0.0003,
        },
        bottom_hand_area={
            "mean_luma": 97.74,
            "stddev": 63.02,
            "bright_ratio": 0.1755,
            "dark_ratio": 0.3815,
            "white_ratio": 0.0018,
            "colorful_ratio": 0.7024,
            "gold_ratio": 0.0035,
            "orange_ratio": 0.0044,
            "red_ratio": 0.002,
            "green_ratio": 0.0,
        },
        center_dialog={
            "dark_ratio": 0.1089,
            "stddev": 32.02,
            "colorful_ratio": 0.7917,
            "gold_ratio": 0.0083,
            "orange_ratio": 0.0047,
            "bright_ratio": 0.0199,
        },
        right_replay_panel={
            "dark_ratio": 0.7988,
        },
    )

    buttons, is_user_turn, notes = detect_actions("in_match", metrics)

    assert buttons == []
    assert is_user_turn is False
    assert any("hand tiles" in note for note in notes)


def _build_metrics(**overrides: dict[str, float]) -> dict[str, dict[str, float]]:
    defaults = {
        "mean_luma": 0.0,
        "stddev": 0.0,
        "bright_ratio": 0.0,
        "dark_ratio": 0.0,
        "white_ratio": 0.0,
        "colorful_ratio": 0.0,
        "gold_ratio": 0.0,
        "orange_ratio": 0.0,
        "red_ratio": 0.0,
        "green_ratio": 0.0,
    }
    keys = [
        "full_frame",
        "top_banner",
        "center_dialog",
        "bottom_action_bar",
        "bottom_hand_area",
        "right_replay_panel",
    ]
    metrics: dict[str, dict[str, float]] = {key: dict(defaults) for key in keys}
    for key, values in overrides.items():
        metrics[key].update(values)
    return metrics
