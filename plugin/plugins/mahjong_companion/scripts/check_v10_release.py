from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

from ..action.action_registry import ActionRegistry
from ..storage import load_json_payload, write_json_atomic


SCHEMA_VERSION = "mahjong-companion-release-checklist-v1"
DEFAULT_PLUGIN_DIR = Path("plugin/plugins/mahjong_companion")
DEFAULT_RELEASE_GATE_REPORT = DEFAULT_PLUGIN_DIR / "plans" / "artifacts" / "eval-report-v1.0-release-gate.json"
RAW_CALIBRATION_IGNORE = "plugin/plugins/mahjong_companion/data/calibration/raw/"
UI_GAME_ACTION_IDS = (
    "ui_chi",
    "ui_pon",
    "ui_kan",
    "ui_riichi",
    "ui_ron",
    "ui_tsumo",
    "ui_skip",
)


@dataclass(frozen=True)
class ReleaseCheck:
    check_id: str
    ok: bool
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "check_id": self.check_id,
            "ok": self.ok,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = check_v10_release(
        repo_root=Path(args.repo_root),
        plugin_dir=Path(args.plugin_dir),
        release_gate_report=Path(args.release_gate_report),
    )
    if args.report:
        write_json_atomic(Path(args.report), result)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result["ok"] else 1


def check_v10_release(
    *,
    repo_root: Path = Path("."),
    plugin_dir: Path = DEFAULT_PLUGIN_DIR,
    release_gate_report: Path = DEFAULT_RELEASE_GATE_REPORT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plugin_dir = _resolve_under(repo_root, plugin_dir)
    release_gate_report = _resolve_under(repo_root, release_gate_report)
    version = _plugin_version(plugin_dir / "plugin.toml")
    checks = [
        _check_plugin_version(version),
        _check_readme_version(plugin_dir / "README.md", version),
        _check_changelog(plugin_dir / "CHANGELOG.md", version),
        _check_closeout_docs(plugin_dir / "plans", version),
        _check_data_lifecycle_doc(plugin_dir / "DATA_LIFECYCLE.md"),
        _check_data_lifecycle_entries(plugin_dir),
        _check_runtime_diagnostics_entry(plugin_dir),
        _check_raw_calibration_ignored(repo_root / ".gitignore"),
        _check_release_gate_report(release_gate_report),
        _check_advice_only_action_registry(),
    ]
    failures = [check.check_id for check in checks if not check.ok]
    return {
        "ok": not failures,
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(repo_root),
        "plugin_dir": str(plugin_dir),
        "plugin_version": version,
        "release_gate_report": str(release_gate_report),
        "checks": [check.to_dict() for check in checks],
        "failures": failures,
    }


def _check_plugin_version(version: str) -> ReleaseCheck:
    ok = bool(re.fullmatch(r"\d+\.\d+\.\d+", version))
    return ReleaseCheck(
        check_id="plugin_version",
        ok=ok,
        message=f"plugin.toml version is {version!r}" if ok else "plugin.toml version is missing or invalid",
    )


def _check_readme_version(path: Path, version: str) -> ReleaseCheck:
    text = _read_text(path)
    expected = f"当前版本：`v{version}`"
    ok = bool(version) and expected in text
    return ReleaseCheck(
        check_id="readme_version",
        ok=ok,
        message=f"README current version matches v{version}" if ok else f"README must include {expected}",
    )


def _check_changelog(path: Path, version: str) -> ReleaseCheck:
    text = _read_text(path)
    expected = f"## v{version}"
    ok = bool(version) and expected in text
    return ReleaseCheck(
        check_id="changelog_version",
        ok=ok,
        message=f"CHANGELOG has v{version} section" if ok else f"CHANGELOG must include {expected}",
    )


def _check_closeout_docs(plans_dir: Path, version: str) -> ReleaseCheck:
    minor = ".".join(version.split(".")[:2]) if version else ""
    closeout_path = plans_dir / f"blueprint-v{minor}-closeout.md"
    next_plan_path = plans_dir / "blueprint-v1.1-execution-plan.md"
    ok = closeout_path.exists() and next_plan_path.exists()
    return ReleaseCheck(
        check_id="closeout_docs",
        ok=ok,
        message="closeout and next execution plan docs exist"
        if ok
        else "closeout doc or next execution plan doc is missing",
        details={
            "closeout_path": str(closeout_path),
            "next_plan_path": str(next_plan_path),
        },
    )


def _check_data_lifecycle_doc(path: Path) -> ReleaseCheck:
    text = _read_text(path)
    required_terms = ["本地", "导出", "删除", "不会自动上传"]
    missing = [term for term in required_terms if term not in text]
    ok = path.exists() and not missing
    return ReleaseCheck(
        check_id="data_lifecycle_doc",
        ok=ok,
        message="data lifecycle doc covers local storage, export, deletion, and upload boundary"
        if ok
        else "data lifecycle doc is missing required user-data terms",
        details={"path": str(path), "missing_terms": missing},
    )


def _check_data_lifecycle_entries(plugin_dir: Path) -> ReleaseCheck:
    entry_text = _read_text(plugin_dir / "__init__.py")
    controller_text = "\n".join([
        _read_text(plugin_dir / "orchestrator.py"),
        _read_text(plugin_dir / "lifecycle_controller.py"),
    ])
    module_text = _read_text(plugin_dir / "data_lifecycle.py")
    required_entry_ids = [
        "get_data_lifecycle",
        "export_local_data",
        "clear_local_runtime_data",
        "clear_calibration_raw_data",
    ]
    missing_entries = [entry_id for entry_id in required_entry_ids if f'id="{entry_id}"' not in entry_text]
    required_functions = [
        "describe_local_data",
        "export_local_data",
        "clear_local_runtime_data",
        "clear_calibration_raw_data",
    ]
    missing_functions = [name for name in required_functions if f"def {name}" not in module_text]
    missing_orchestrator_methods = [
        name for name in required_entry_ids if f"async def {name}" not in controller_text
    ]
    ok = not missing_entries and not missing_functions and not missing_orchestrator_methods
    return ReleaseCheck(
        check_id="data_lifecycle_entries",
        ok=ok,
        message="data lifecycle plugin entries and implementation exist"
        if ok
        else "data lifecycle plugin entries or implementation are missing",
        details={
            "missing_entries": missing_entries,
            "missing_functions": missing_functions,
            "missing_orchestrator_methods": missing_orchestrator_methods,
        },
    )


def _check_runtime_diagnostics_entry(plugin_dir: Path) -> ReleaseCheck:
    entry_text = _read_text(plugin_dir / "__init__.py")
    controller_text = "\n".join([
        _read_text(plugin_dir / "orchestrator.py"),
        _read_text(plugin_dir / "lifecycle_controller.py"),
    ])
    module_text = _read_text(plugin_dir / "diagnostics.py")
    missing = []
    if 'id="get_runtime_diagnostics"' not in entry_text:
        missing.append("plugin_entry:get_runtime_diagnostics")
    if "async def get_runtime_diagnostics" not in controller_text:
        missing.append("orchestrator:get_runtime_diagnostics")
    if "def build_runtime_diagnostics" not in module_text:
        missing.append("diagnostics:build_runtime_diagnostics")
    ok = not missing
    return ReleaseCheck(
        check_id="runtime_diagnostics_entry",
        ok=ok,
        message="runtime diagnostics entry exists" if ok else "runtime diagnostics entry is missing",
        details={"missing": missing},
    )


def _check_raw_calibration_ignored(path: Path) -> ReleaseCheck:
    text = _read_text(path)
    ok = RAW_CALIBRATION_IGNORE in text
    return ReleaseCheck(
        check_id="raw_calibration_ignored",
        ok=ok,
        message="raw calibration screenshots are ignored by git"
        if ok
        else f".gitignore must contain {RAW_CALIBRATION_IGNORE}",
    )


def _check_release_gate_report(path: Path) -> ReleaseCheck:
    payload = load_json_payload(path, default={}, expected_type=dict)
    ok = bool(payload.get("ok") is True)
    return ReleaseCheck(
        check_id="release_gate_report",
        ok=ok,
        message="release gate report exists and is ok=true" if ok else "release gate report is missing or not ok=true",
        details={"path": str(path), "schema_version": payload.get("schema_version", "")},
    )


def _check_advice_only_action_registry() -> ReleaseCheck:
    registry = ActionRegistry()
    registered_ids = {action.action_id for action in registry.list_actions()}
    registered_ui_ids = sorted(set(UI_GAME_ACTION_IDS) & registered_ids)
    validation_results = {
        action_id: registry.validate(
            action_id,
            current_scene="in_match",
            action_mode="assist",
            session_running=True,
            user_confirmed=True,
        )
        for action_id in UI_GAME_ACTION_IDS
    }
    accepted_ui_ids = sorted(action_id for action_id, (ok, _reason) in validation_results.items() if ok)
    ok = not registered_ui_ids and not accepted_ui_ids
    return ReleaseCheck(
        check_id="advice_only_action_registry",
        ok=ok,
        message="in-match UI button actions remain unregistered"
        if ok
        else "one or more in-match UI button actions are executable",
        details={
            "registered_ui_ids": registered_ui_ids,
            "accepted_ui_ids": accepted_ui_ids,
        },
    )


def _plugin_version(path: Path) -> str:
    text = _read_text(path)
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    return match.group(1).strip() if match else ""


def _resolve_under(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mahjong Companion v1.0 release checklist checks.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--plugin-dir", default=str(DEFAULT_PLUGIN_DIR))
    parser.add_argument("--release-gate-report", default=str(DEFAULT_RELEASE_GATE_REPORT))
    parser.add_argument("--report", default="")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
