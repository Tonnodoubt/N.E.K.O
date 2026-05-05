from __future__ import annotations

import json
from pathlib import Path

from plugin.plugins.mahjong_companion.scripts.check_v10_release import (
    check_v10_release,
    main as check_v10_release_main,
)


def test_check_v10_release_passes_complete_fixture(tmp_path: Path) -> None:
    plugin_dir = _write_release_fixture(tmp_path)

    result = check_v10_release(
        repo_root=tmp_path,
        plugin_dir=plugin_dir,
        release_gate_report=plugin_dir / "plans" / "artifacts" / "eval-report-v1.0-release-gate.json",
    )

    assert result["ok"] is True
    assert result["plugin_version"] == "0.5.0"
    assert result["failures"] == []


def test_check_v10_release_flags_missing_data_lifecycle_doc(tmp_path: Path) -> None:
    plugin_dir = _write_release_fixture(tmp_path)
    (plugin_dir / "DATA_LIFECYCLE.md").unlink()

    result = check_v10_release(
        repo_root=tmp_path,
        plugin_dir=plugin_dir,
        release_gate_report=plugin_dir / "plans" / "artifacts" / "eval-report-v1.0-release-gate.json",
    )

    assert result["ok"] is False
    assert "data_lifecycle_doc" in result["failures"]


def test_check_v10_release_cli_writes_report(tmp_path: Path, capsys) -> None:
    plugin_dir = _write_release_fixture(tmp_path)
    report_path = tmp_path / "release-checklist.json"

    exit_code = check_v10_release_main(
        [
            "--repo-root",
            str(tmp_path),
            "--plugin-dir",
            str(plugin_dir),
            "--release-gate-report",
            str(plugin_dir / "plans" / "artifacts" / "eval-report-v1.0-release-gate.json"),
            "--report",
            str(report_path),
            "--pretty",
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout["ok"] is True
    assert written["schema_version"] == "mahjong-companion-release-checklist-v1"


def _write_release_fixture(root: Path) -> Path:
    plugin_dir = root / "plugin" / "plugins" / "mahjong_companion"
    plans_dir = plugin_dir / "plans"
    artifacts_dir = plans_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (root / ".gitignore").write_text(
        "plugin/plugins/mahjong_companion/data/calibration/raw/\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.toml").write_text('version = "0.5.0"\n', encoding="utf-8")
    (plugin_dir / "README.md").write_text("当前版本：`v0.5.0`\n", encoding="utf-8")
    (plugin_dir / "CHANGELOG.md").write_text("## v0.5.0 - 2026-05-02\n", encoding="utf-8")
    (plans_dir / "blueprint-v0.5-closeout.md").write_text("closeout\n", encoding="utf-8")
    (plans_dir / "blueprint-v1.1-execution-plan.md").write_text("plan\n", encoding="utf-8")
    (plugin_dir / "DATA_LIFECYCLE.md").write_text(
        "本地数据只保存在本地。用户可以导出，也可以删除。不会自动上传。\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        '\n'.join([
            '@plugin_entry(id="get_data_lifecycle", name="查看本地数据", kind="action")',
            '@plugin_entry(id="export_local_data", name="导出本地数据包", kind="action")',
            '@plugin_entry(id="clear_local_runtime_data", name="清理本地运行数据", kind="action")',
            '@plugin_entry(id="clear_calibration_raw_data", name="清理原始校准素材", kind="action")',
            '@plugin_entry(id="get_runtime_diagnostics", name="查看运行诊断", kind="action")',
        ]),
        encoding="utf-8",
    )
    (plugin_dir / "orchestrator.py").write_text(
        "\n".join([
            "async def get_data_lifecycle(self): pass",
            "async def export_local_data(self): pass",
            "async def clear_local_runtime_data(self): pass",
            "async def clear_calibration_raw_data(self): pass",
            "async def get_runtime_diagnostics(self): pass",
        ]),
        encoding="utf-8",
    )
    (plugin_dir / "data_lifecycle.py").write_text(
        "\n".join([
            "def describe_local_data(): pass",
            "def export_local_data(): pass",
            "def clear_local_runtime_data(): pass",
            "def clear_calibration_raw_data(): pass",
        ]),
        encoding="utf-8",
    )
    (plugin_dir / "diagnostics.py").write_text(
        "def build_runtime_diagnostics(): pass\n",
        encoding="utf-8",
    )
    (artifacts_dir / "eval-report-v1.0-release-gate.json").write_text(
        json.dumps({"ok": True, "schema_version": "mahjong-companion-release-gate-v1"}),
        encoding="utf-8",
    )
    return plugin_dir
