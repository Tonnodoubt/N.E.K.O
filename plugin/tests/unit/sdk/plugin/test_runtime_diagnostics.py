from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from plugin.plugins.mahjong_companion import diagnostics
from plugin.plugins.mahjong_companion import MahjongCompanionPlugin
from plugin.plugins.mahjong_companion.config_defaults import DEFAULT_CONFIG, merge_runtime_config
from plugin.plugins.mahjong_companion.diagnostics import _check_recent_status, build_runtime_diagnostics
from plugin.plugins.mahjong_companion.orchestrator import SessionOrchestrator


PLUGIN_DIR = Path("plugin/plugins/mahjong_companion")


class _FakePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logger = logging.getLogger("mahjong-companion-runtime-diagnostics-test")
        self.statuses: list[dict[str, object]] = []

    def data_path(self, *parts: str) -> Path:
        path = self.root / "data"
        if parts:
            path = path.joinpath(*parts)
        return path

    def report_status(self, payload: dict[str, object]) -> None:
        self.statuses.append(dict(payload))

    def push_message(self, **kwargs: object) -> dict[str, object]:
        return {"ok": True, **kwargs}


def test_runtime_diagnostics_reports_ready_local_assets(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ready_data_root(data_root)

    result = build_runtime_diagnostics(
        plugin_dir=PLUGIN_DIR,
        data_root=data_root,
        config=merge_runtime_config(DEFAULT_CONFIG, {}),
        status={"window_bound": True, "last_capture_ok": True, "last_perception_ok": True},
    )

    checks = {check["check_id"]: check for check in result["checks"]}
    assert result["ok"] is True
    assert result["health"] == "ok"
    assert checks["calibration_profiles"]["enabled_profile_count"] == 1
    assert checks["button_templates"]["missing_buttons"] == []
    assert checks["onnx_tile_classifier"]["available"] is True
    assert checks["onnx_tile_classifier"]["labels_count"] == 2
    assert checks["advice_only_registry"]["registered_ui_ids"] == []


def test_runtime_diagnostics_warns_when_calibration_profile_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "session_cache").mkdir(parents=True)
    (data_root / "debug_samples").mkdir()
    (data_root / "calibration" / "profiles").mkdir(parents=True)

    result = build_runtime_diagnostics(
        plugin_dir=PLUGIN_DIR,
        data_root=data_root,
        config=merge_runtime_config(DEFAULT_CONFIG, {}),
        status={},
    )

    issue_codes = {issue["code"] for issue in result["issues"]}
    assert result["ok"] is True
    assert result["health"] == "warning"
    assert "calibration_profile_missing" in issue_codes
    assert "window_not_bound" in issue_codes


def test_onnx_tile_classifier_diagnostics_reports_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    model_dir = _write_fake_onnx_model(data_root)
    monkeypatch.setenv("MAHJONG_COMPANION_ONNX_HAND_ENABLED", "yes")

    result = diagnostics._check_onnx_tile_classifier(data_root)

    assert result["ok"] is True
    assert result["available"] is True
    assert result["model_dir"] == str(model_dir)
    assert result["model_size_mb"] == 0.0
    assert result["labels_count"] == 2
    assert result["hand_onnx_enabled"] is True
    assert result["discard_occupancy_confidence"] == 0.90
    assert result["metadata"]["purpose"] == "discard_onnx_default_hand_opt_in"
    assert result["issues"] == []


def test_onnx_tile_classifier_diagnostics_missing_dir_is_info(tmp_path: Path) -> None:
    result = diagnostics._check_onnx_tile_classifier(tmp_path / "data")

    assert result["ok"] is False
    assert result["available"] is False
    assert result["issues"][0]["severity"] == "info"
    assert result["issues"][0]["code"] == "onnx_tile_model_missing"


def test_startup_seed_runtime_data_assets(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    plugin._copy_bundled_asset_tree = (  # type: ignore[attr-defined]
        lambda source_dir, target_dir: MahjongCompanionPlugin._copy_bundled_asset_tree(
            plugin,
            source_dir,
            target_dir,
        )
    )
    copied = MahjongCompanionPlugin._ensure_runtime_data_assets(plugin)

    assert copied["calibration_profiles"] >= 1
    assert copied["onnx_tile_model"] >= 1
    assert (plugin.data_path("calibration", "profiles") / "majsoul-pc-manual-2026.05-vit-discard-2560x1440.json").is_file()
    assert (plugin.data_path("models", "vit_tile_classifier") / "model.onnx").is_file()

    copied_again = MahjongCompanionPlugin._ensure_runtime_data_assets(plugin)

    assert copied_again == {}


def test_recent_status_ok_follows_issue_health_for_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    original_issue = diagnostics._issue

    def _issue_with_recent_error_as_error(
        severity: str,
        code: str,
        message: str,
        details: dict[str, object],
    ) -> dict[str, object]:
        if code == "recent_error":
            severity = "error"
        return original_issue(severity, code, message, details)

    monkeypatch.setattr(diagnostics, "_issue", _issue_with_recent_error_as_error)

    status = diagnostics._check_recent_status({
        "window_bound": True,
        "last_error": "capture backend crashed",
    })

    assert status["ok"] is False


@pytest.mark.asyncio
async def test_orchestrator_get_runtime_diagnostics(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    _write_ready_data_root(plugin.data_path())
    orchestrator = SessionOrchestrator(plugin)
    orchestrator.apply_config(merge_runtime_config(DEFAULT_CONFIG, {}))
    orchestrator.state.window_bound = True

    result = await orchestrator.get_runtime_diagnostics()

    assert result.value["ok"] is True
    assert result.value["schema_version"] == "mahjong-companion-diagnostics-v1"
    assert result.value["summary"]["error_count"] == 0


def _write_ready_data_root(data_root: Path) -> None:
    (data_root / "session_cache").mkdir(parents=True)
    (data_root / "debug_samples").mkdir()
    profiles_dir = data_root / "calibration" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "test-1920x1080.json").write_text(
        json.dumps(
            {
                "profile_id": "test-1920x1080",
                "version": "v0.3-calibration",
                "enabled": True,
                "screen_width": 1920,
                "screen_height": 1080,
                "confidence": 0.91,
                "hand_tile_templates": {"templates": {"1m": {"sample_count": 2}}},
                "discard_tile_templates": {"templates": {"1m": {"source_sample_count": 2}}},
            }
        ),
        encoding="utf-8",
    )
    _write_fake_onnx_model(data_root)


def _write_fake_onnx_model(data_root: Path) -> Path:
    model_dir = data_root / "models" / "vit_tile_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"fake")
    (model_dir / "preprocessor.json").write_text("{}", encoding="utf-8")
    (model_dir / "labels.json").write_text(
        json.dumps({"0": "1m", "1": "empty"}),
        encoding="utf-8",
    )
    (model_dir / "metadata.json").write_text(
        json.dumps({"purpose": "discard_onnx_default_hand_opt_in"}),
        encoding="utf-8",
    )
    return model_dir
