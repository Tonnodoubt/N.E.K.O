from __future__ import annotations

from PIL import Image

from plugin.plugins.mahjong_companion.scripts.run_runtime_strategy_smoke import (
    _audit_markdown,
    _build_auto_audit,
    _report_markdown,
    _format_river_by_player,
    _save_contact_sheet,
    _save_frame_review_images,
)


def test_runtime_strategy_report_embeds_contact_sheet():
    markdown = _report_markdown(
        {
            "frame_count": 1,
            "contact_sheet": "runtime_strategy_smoke.png",
            "frame_review_dir": "frame_reviews",
            "frame_review_images": ["01_frame.png"],
            "rows": [],
        }
    )

    assert "![Runtime strategy contact sheet](runtime_strategy_smoke.png)" in markdown
    assert "[01_frame.png](frame_reviews/01_frame.png)" in markdown


def test_runtime_strategy_contact_sheet_writes_image(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (320, 180), (20, 40, 60)).save(image_path)
    out_path = tmp_path / "runtime_strategy_smoke.png"

    _save_contact_sheet(
        [
            {
                "image": str(image_path),
                "effective_scene": "in_match",
                "is_user_turn": True,
                "hand_count": 13,
                "visible_tile_count": 32,
                "candidate_discards": [{"tile": "7p"}],
                "candidate_strength": "strong",
                "overlay_primary": "七筒",
                "overlay_reason": "优先考虑：测试文案",
            }
        ],
        out_path,
    )

    assert out_path.exists()
    with Image.open(out_path) as image:
        assert image.size == (1040, 430)


def test_runtime_strategy_frame_reviews_write_detailed_images(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (320, 180), (20, 40, 60)).save(image_path)

    written = _save_frame_review_images(
        [
            {
                "image": str(image_path),
                "effective_scene": "in_match",
                "is_user_turn": True,
                "confidence": 0.72,
                "decision_type": "scene_update",
                "recommended_focus": "observe",
                "hand_source": "bottom_hand_detector",
                "discard_parser_source": "model_river_adapter",
                "hand_count": 3,
                "hand_tiles": ["1m", "2m", "3m"],
                "visible_tile_count": 4,
                "visible_tiles": ["7p", "7p", "1z", "8s"],
                "discard_piles": {
                    "self": [{"tile": "7p", "turn_index": 1, "bbox": [20, 20, 40, 40]}],
                    "right_opponent": [{"tile": "1z", "turn_index": 1, "bbox": [50, 20, 70, 40]}],
                },
                "candidate_discards": [
                    {
                        "tile": "7p",
                        "recommendation_strength": "strong",
                        "score": 1.0,
                        "ukeire_estimate": 12,
                        "current_shanten": 1,
                        "post_discard_shanten": 1,
                        "safety_hint": "medium",
                        "reason": "测试原因",
                    }
                ],
                "candidate_strength": "strong",
                "overlay_primary": "七筒",
                "overlay_reason": "优先考虑：测试文案",
            }
        ],
        tmp_path / "frame_reviews",
    )

    assert written == ["01_frame.png"]
    with Image.open(tmp_path / "frame_reviews" / "01_frame.png") as image:
        assert image.size == (1500, 980)


def test_runtime_strategy_formats_river_by_player():
    text = _format_river_by_player(
        {
            "self": [{"tile": "7p", "turn_index": 2}, {"tile": "1m", "turn_index": 1}],
            "right_opponent": [{"tile": "1z", "turn_index": 1}],
        }
    )

    assert "self: 一万 七筒" in text
    assert "right: 东风" in text
    assert "top: none" in text


def test_runtime_strategy_auto_audit_flags_candidate_not_in_hand():
    audit = _build_auto_audit(
        {
            "frame_review_images": ["01_frame.png"],
            "rows": [
                {
                    "image": "/tmp/frame.png",
                    "effective_scene": "in_match",
                    "confidence": 0.72,
                    "hand_tiles": ["1m", "2m", "3m"],
                    "visible_tiles": [],
                    "discard_piles": {},
                    "candidate_discards": [{"tile": "7p", "recommendation_strength": "strong"}],
                    "candidate_strength": "strong",
                    "decision_type": "scene_update",
                }
            ],
        }
    )

    assert audit["suspicious_frame_count"] == 1
    codes = {item["code"] for item in audit["findings"][0]["findings"]}
    assert "candidate_not_in_hand" in codes


def test_runtime_strategy_auto_audit_markdown_links_review_image():
    markdown = _audit_markdown(
        {
            "frame_count": 1,
            "suspicious_frame_count": 1,
            "severity_counts": {"high": 1},
            "findings": [
                {
                    "frame": "frame.png",
                    "review_image": "01_frame.png",
                    "severity": "high",
                    "findings": [{"code": "candidate_not_in_hand", "detail": "bad"}],
                }
            ],
        }
    )

    assert "[01_frame.png](frame_reviews/01_frame.png)" in markdown
    assert "candidate_not_in_hand" in markdown
