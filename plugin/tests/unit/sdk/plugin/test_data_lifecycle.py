from __future__ import annotations

import json
import logging
from pathlib import Path
import zipfile

import pytest

from plugin.plugins.mahjong_companion.data_lifecycle import (
    _safe_package_name,
    clear_calibration_raw_data,
    clear_local_runtime_data,
    describe_local_data,
    export_local_data,
)
from plugin.plugins.mahjong_companion.orchestrator import SessionOrchestrator


class _FakePlugin:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.logger = logging.getLogger("mahjong-companion-data-lifecycle-test")
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


def test_describe_local_data_reports_export_and_protected_groups(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_fixture_data(data_root)
    plugin_dir = tmp_path / "plugin"
    (plugin_dir / "DATA_LIFECYCLE.md").parent.mkdir(parents=True)
    (plugin_dir / "DATA_LIFECYCLE.md").write_text("本地 导出 删除 不会自动上传\n", encoding="utf-8")

    report = describe_local_data(data_root, plugin_dir=plugin_dir)

    groups = {group["group_id"]: group for group in report["groups"]}
    assert report["ok"] is True
    assert report["default_export_groups"] == ["session_cache", "debug_samples", "calibration_profiles"]
    assert report["runtime_clearable_groups"] == ["session_cache", "debug_samples", "exports"]
    assert report["lifecycle_doc_exists"] is True
    assert groups["session_cache"]["file_count"] == 1
    assert groups["calibration_profiles"]["default_export"] is True
    assert groups["calibration_raw"]["default_export"] is False
    assert groups["calibration_raw"]["clearable_runtime"] is False


def test_export_local_data_default_package_excludes_raw_calibration(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_fixture_data(data_root)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "DATA_LIFECYCLE.md").write_text("本地 导出 删除 不会自动上传\n", encoding="utf-8")

    result = export_local_data(data_root, plugin_dir=plugin_dir, package_name="fixture.zip")

    assert result["ok"] is True
    assert result["selected_groups"] == ["session_cache", "debug_samples", "calibration_profiles"]
    assert result["raw_calibration_included"] is False
    package_path = Path(result["package_path"])
    assert package_path == data_root / "exports" / "fixture.zip"
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("mahjong_companion_data/manifest.json").decode("utf-8"))

    assert "mahjong_companion_data/session_cache/review_candidates.json" in names
    assert "mahjong_companion_data/debug_samples/frame.png" in names
    assert "mahjong_companion_data/calibration/profiles/profile.json" in names
    assert "mahjong_companion_data/docs/DATA_LIFECYCLE.md" in names
    assert not any("/calibration/raw/" in name for name in names)
    assert "calibration_raw" in manifest["excluded_groups"]


def test_safe_package_name_rejects_dot_only_zip_names() -> None:
    timestamp = "20260503-000000"

    assert _safe_package_name("..zip", timestamp) == f"mahjong-companion-data-{timestamp}.zip"
    assert _safe_package_name(".zip", timestamp) == f"mahjong-companion-data-{timestamp}.zip"
    assert _safe_package_name("../fixture", timestamp) == "fixture.zip"
    assert _safe_package_name(".hidden.zip", timestamp) == "hidden.zip"


def test_clear_local_runtime_data_preserves_calibration_and_gitkeep(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_fixture_data(data_root)
    (data_root / "session_cache" / ".gitkeep").write_text("", encoding="utf-8")
    (data_root / "debug_samples" / ".gitkeep").write_text("", encoding="utf-8")

    dry_run = clear_local_runtime_data(data_root)

    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert (data_root / "session_cache" / "review_candidates.json").exists()
    assert (data_root / "debug_samples" / "frame.png").exists()

    result = clear_local_runtime_data(data_root, dry_run=False)

    assert result["ok"] is True
    assert result["protected_groups"] == ["calibration_profiles", "calibration_raw"]
    assert not (data_root / "session_cache" / "review_candidates.json").exists()
    assert not (data_root / "debug_samples" / "frame.png").exists()
    assert (data_root / "session_cache" / ".gitkeep").exists()
    assert (data_root / "debug_samples" / ".gitkeep").exists()
    assert (data_root / "calibration" / "profiles" / "profile.json").exists()
    assert (data_root / "calibration" / "raw" / "secret-frame.png").exists()


def test_clear_calibration_raw_data_requires_explicit_token_and_preserves_profiles(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_fixture_data(data_root)

    dry_run = clear_calibration_raw_data(data_root)
    blocked = clear_calibration_raw_data(data_root, dry_run=False)
    result = clear_calibration_raw_data(
        data_root,
        dry_run=False,
        confirm_token="DELETE_CALIBRATION_RAW",
    )

    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert blocked["ok"] is False
    assert blocked["confirm_token_required"] == "DELETE_CALIBRATION_RAW"
    assert result["ok"] is True
    assert result["selected_groups"] == ["calibration_raw"]
    assert result["protected_groups"] == ["calibration_profiles"]
    assert (data_root / "calibration" / "profiles" / "profile.json").exists()
    assert not (data_root / "calibration" / "raw" / "secret-frame.png").exists()


@pytest.mark.asyncio
async def test_orchestrator_data_lifecycle_entries_export_and_block_running_clear(tmp_path: Path) -> None:
    plugin = _FakePlugin(tmp_path)
    _write_fixture_data(plugin.data_path())
    orchestrator = SessionOrchestrator(plugin)

    lifecycle = await orchestrator.get_data_lifecycle()
    exported = await orchestrator.export_local_data(package_name="orchestrator-export.zip")
    orchestrator.state.running = True
    blocked = await orchestrator.clear_local_runtime_data(dry_run=False)
    dry_run = await orchestrator.clear_local_runtime_data(dry_run=True)
    raw_dry_run = await orchestrator.clear_calibration_raw_data(dry_run=True)
    raw_blocked = await orchestrator.clear_calibration_raw_data(
        dry_run=False,
        confirm_token="DELETE_CALIBRATION_RAW",
    )

    assert lifecycle.value["ok"] is True
    assert exported.value["ok"] is True
    assert Path(exported.value["package_path"]).exists()
    assert blocked.value["ok"] is False
    assert blocked.value["error"] == "stop session before clearing runtime data"
    assert dry_run.value["ok"] is True
    assert dry_run.value["dry_run"] is True
    assert raw_dry_run.value["ok"] is True
    assert raw_dry_run.value["dry_run"] is True
    assert raw_blocked.value["ok"] is False
    assert raw_blocked.value["error"] == "stop session before clearing calibration raw data"


def _write_fixture_data(data_root: Path) -> None:
    _write_text(data_root / "session_cache" / "review_candidates.json", '{"items": []}\n')
    _write_bytes(data_root / "debug_samples" / "frame.png", b"fake image bytes")
    _write_text(data_root / "calibration" / "profiles" / "profile.json", '{"profile": true}\n')
    _write_bytes(data_root / "calibration" / "raw" / "secret-frame.png", b"private frame bytes")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
