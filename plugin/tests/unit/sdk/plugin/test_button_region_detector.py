from __future__ import annotations

from pathlib import Path

from PIL import Image

from plugin.plugins.mahjong_companion.perception.action_detector import detect_button_regions
from plugin.plugins.mahjong_companion.perception.calibration import CalibrationProfile
from plugin.plugins.mahjong_companion.perception.pipeline import analyze_image_path
from plugin.plugins.mahjong_companion.perception.roi import build_default_rois, collect_region_metrics


def test_detect_button_regions_finds_pasted_templates() -> None:
    image = Image.new("RGB", (1920, 1080), (32, 69, 118))
    template_dir = Path("plugin/plugins/mahjong_companion/perception/templates")
    kan = Image.open(template_dir / "1920x1080" / "kan.png").convert("RGB")
    skip = Image.open(template_dir / "1920x1080" / "skip.png").convert("RGB")
    image.paste(kan, (860, 760))
    image.paste(skip, (1130, 760))

    rois = build_default_rois(*image.size)
    metrics = {name: collect_region_metrics(image, roi) for name, roi in rois.items()}
    metrics["full_frame"] = collect_region_metrics(image, None)
    metrics["bottom_action_bar"]["orange_ratio"] = 0.04

    regions = detect_button_regions(
        image,
        metrics,
        profile=CalibrationProfile(
            profile_id="test-1920x1080",
            enabled=True,
            screen_width=1920,
            screen_height=1080,
        ),
        template_dir=template_dir,
    )

    by_type = {region.button_type: region for region in regions}
    assert by_type["kan"].bbox == (860, 760, 1155, 890)
    assert by_type["skip"].bbox == (1130, 760, 1505, 890)
    assert by_type["kan"].confidence >= 0.99
    assert by_type["skip"].confidence >= 0.99


def test_analyze_image_path_promotes_action_button_frame_to_in_match(tmp_path: Path) -> None:
    # Synthesize the test frame instead of depending on an un-tracked debug
    # screenshot (CODE_REVIEW_v1.2 M3). Paste the chi button template onto a
    # mahjong-table coloured backdrop so the action-button promotion path runs
    # against a deterministic, reproducible input.
    template_dir = Path("plugin/plugins/mahjong_companion/perception/templates")
    calibration_dir = Path("plugin/plugins/mahjong_companion/data/calibration")
    chi = Image.open(template_dir / "1920x1080" / "chi.png").convert("RGB")
    image = Image.new("RGB", (1920, 1080), (32, 69, 118))
    image.paste(chi, (860, 760))

    image_path = tmp_path / "synthetic-chi-frame.png"
    image.save(image_path)

    state, _ = analyze_image_path(
        image_path,
        calibration_dir=calibration_dir,
        template_dir=template_dir,
    )

    assert state.scene == "in_match"
    assert state.is_user_turn is True
    assert "chi" in state.buttons
    assert any(region["button_type"] == "chi" for region in state.button_regions)
