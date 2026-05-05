from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from utils.logger_config import get_module_logger

from .action.action_registry import ActionRegistry
from .perception.calibration import load_calibration_profile
from .perception.external_discard_recognizer import ENV_COMMAND, ENV_ENDPOINT, ENV_TIMEOUT

logger = get_module_logger(__name__)

SCHEMA_VERSION = "mahjong-companion-diagnostics-v1"
REQUIRED_ADVICE_BUTTONS = ("chi", "pon", "kan", "riichi", "ron", "tsumo", "skip")
IN_MATCH_UI_ACTION_IDS = (
    "ui_chi",
    "ui_pon",
    "ui_kan",
    "ui_riichi",
    "ui_ron",
    "ui_tsumo",
    "ui_skip",
)


def build_runtime_diagnostics(
    *,
    plugin_dir: Path,
    data_root: Path,
    config: dict[str, Any],
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plugin_dir = Path(plugin_dir).resolve()
    data_root = Path(data_root).resolve()
    status = dict(status or {})
    checks = [
        _check_data_directories(data_root),
        _check_calibration_profiles(data_root / "calibration" / "profiles"),
        _check_button_templates(plugin_dir / "perception" / "templates"),
        _check_external_discard_recognizer(),
        _check_runtime_config(config),
        _check_recent_status(status),
        _check_advice_only_registry(),
    ]
    issues = [issue for check in checks for issue in check.get("issues", []) if isinstance(issue, dict)]
    return {
        "ok": not any(issue.get("severity") == "error" for issue in issues),
        "schema_version": SCHEMA_VERSION,
        "health": _health_from_issues(issues),
        "plugin_dir": str(plugin_dir),
        "data_root": str(data_root),
        "checks": checks,
        "issues": issues,
        "issue_count": len(issues),
        "summary": _build_summary(checks, issues),
    }


def _check_data_directories(data_root: Path) -> dict[str, Any]:
    required = {
        "session_cache": data_root / "session_cache",
        "debug_samples": data_root / "debug_samples",
        "calibration": data_root / "calibration",
        "calibration_profiles": data_root / "calibration" / "profiles",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    issues = [
        _issue(
            "warning",
            "missing_data_directory",
            f"本地数据目录缺失：{', '.join(missing)}",
            {"missing": missing},
        )
    ] if missing else []
    return {
        "check_id": "data_directories",
        "ok": not missing,
        "paths": {name: str(path) for name, path in required.items()},
        "missing": missing,
        "issues": issues,
    }


def _check_calibration_profiles(profiles_dir: Path) -> dict[str, Any]:
    profile_paths = sorted(
        path for path in profiles_dir.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    ) if profiles_dir.exists() else []
    loaded = []
    errors = []
    for path in profile_paths:
        try:
            profile = load_calibration_profile(path)
        except Exception as exc:
            logger.warning(
                "calibration profile load failed at %s: %s", path, exc
            )
            errors.append({"path": str(path), "error": str(exc)})
            continue
        loaded.append({
            "path": str(path),
            "profile_id": profile.profile_id,
            "enabled": profile.enabled,
            "screen_width": profile.screen_width,
            "screen_height": profile.screen_height,
            "confidence": profile.confidence,
            "hand_tile_kind_count": _template_kind_count(profile.hand_tile_templates),
            "discard_tile_kind_count": _template_kind_count(profile.discard_tile_templates),
            "discard_source_sample_count": _source_sample_count(profile.discard_tile_templates),
        })
    enabled = [profile for profile in loaded if profile["enabled"]]
    issues = []
    if not loaded:
        issues.append(_issue(
            "warning",
            "calibration_profile_missing",
            "没有找到本地校准 profile，牌级识别会退回低置信默认布局。",
            {"profiles_dir": str(profiles_dir)},
        ))
    if errors:
        issues.append(_issue(
            "warning",
            "calibration_profile_load_error",
            "有校准 profile 无法读取。",
            {"errors": errors},
        ))
    if loaded and not enabled:
        issues.append(_issue(
            "warning",
            "calibration_profile_disabled",
            "校准 profile 存在，但都未启用。",
            {"profiles": loaded},
        ))
    return {
        "check_id": "calibration_profiles",
        "ok": bool(loaded) and bool(enabled) and not errors,
        "profiles_dir": str(profiles_dir),
        "profile_count": len(loaded),
        "enabled_profile_count": len(enabled),
        "profiles": loaded,
        "load_errors": errors,
        "issues": issues,
    }


def _check_button_templates(template_dir: Path) -> dict[str, Any]:
    meta_path = template_dir / "meta.json"
    meta = _read_json_dict(meta_path)
    templates = meta.get("templates") if isinstance(meta.get("templates"), dict) else {}
    present_buttons: set[str] = set()
    missing_files = []
    malformed = []
    entries = []
    if isinstance(templates, dict):
        for template_id, payload in templates.items():
            if not isinstance(payload, dict):
                malformed.append(str(template_id))
                continue
            button_type = str(payload.get("button_type") or template_id).strip()
            file_value = str(payload.get("file", "")).strip()
            path = template_dir / file_value if file_value else template_dir / "__missing__"
            exists = bool(file_value and path.is_file())
            if button_type:
                present_buttons.add(button_type)
            if not exists:
                missing_files.append({"template_id": str(template_id), "file": file_value})
            entries.append({
                "template_id": str(template_id),
                "button_type": button_type,
                "file": file_value,
                "exists": exists,
                "resolution": payload.get("resolution", []),
                "match_threshold": payload.get("match_threshold", meta.get("default_match_threshold", "")),
            })
    missing_buttons = [button for button in REQUIRED_ADVICE_BUTTONS if button not in present_buttons]
    issues = []
    if not meta_path.exists():
        issues.append(_issue(
            "warning",
            "button_template_meta_missing",
            "按钮模板 meta.json 缺失，按钮候选定位会降级。",
            {"meta_path": str(meta_path)},
        ))
    if missing_buttons:
        issues.append(_issue(
            "warning",
            "button_template_missing",
            f"建议按钮模板缺失：{', '.join(missing_buttons)}。",
            {"missing_buttons": missing_buttons},
        ))
    if missing_files:
        issues.append(_issue(
            "warning",
            "button_template_file_missing",
            "有按钮模板登记了文件，但图片不存在。",
            {"missing_files": missing_files},
        ))
    if malformed:
        issues.append(_issue(
            "warning",
            "button_template_malformed",
            "有按钮模板 meta 条目格式不正确。",
            {"malformed": malformed},
        ))
    return {
        "check_id": "button_templates",
        "ok": meta_path.exists() and not missing_buttons and not missing_files and not malformed,
        "template_dir": str(template_dir),
        "meta_path": str(meta_path),
        "required_buttons": list(REQUIRED_ADVICE_BUTTONS),
        "present_buttons": sorted(present_buttons),
        "missing_buttons": missing_buttons,
        "templates": entries,
        "issues": issues,
    }


def _check_external_discard_recognizer() -> dict[str, Any]:
    command = os.environ.get(ENV_COMMAND, "").strip()
    endpoint = os.environ.get(ENV_ENDPOINT, "").strip()
    timeout = os.environ.get(ENV_TIMEOUT, "").strip()
    mode = "off"
    if command:
        mode = "command"
    elif endpoint:
        mode = "http"
    return {
        "check_id": "external_discard_recognizer",
        "ok": True,
        "enabled": bool(command or endpoint),
        "mode": mode,
        "command_configured": bool(command),
        "endpoint_configured": bool(endpoint),
        "timeout_sec": timeout or "2.5",
        "issues": [],
    }


def _check_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    companion_cfg = config.get("mahjong_companion") if isinstance(config.get("mahjong_companion"), dict) else {}
    speech_cfg = companion_cfg.get("speech_policy") if isinstance(companion_cfg.get("speech_policy"), dict) else {}
    runtime_cfg = (
        companion_cfg.get("game_agent_runtime")
        if isinstance(companion_cfg.get("game_agent_runtime"), dict)
        else {}
    )
    action_cfg = companion_cfg.get("action_policy") if isinstance(companion_cfg.get("action_policy"), dict) else {}
    perception_cfg = companion_cfg.get("perception") if isinstance(companion_cfg.get("perception"), dict) else {}
    issues = []
    sample_interval_ms = int(companion_cfg.get("sample_interval_ms", 1200) or 1200)
    if sample_interval_ms < 300:
        issues.append(_issue(
            "warning",
            "sample_interval_too_low",
            "采样间隔很低，可能造成截图和识别负载过高。",
            {"sample_interval_ms": sample_interval_ms},
        ))
    return {
        "check_id": "runtime_config",
        "ok": not issues,
        "default_mode": companion_cfg.get("default_mode", "teaching"),
        "sample_interval_ms": sample_interval_ms,
        "target_window_title_keywords": companion_cfg.get("target_window_title_keywords", []),
        "perception_enabled": bool(perception_cfg.get("enabled", True)),
        "perception_debug_dump": bool(perception_cfg.get("debug_dump", True)),
        "voice_enabled": bool(speech_cfg.get("voice_enabled", True)),
        "voice_mode": str(speech_cfg.get("voice_mode", "key_events_only")),
        "runtime_mode": str(runtime_cfg.get("mode", "active")),
        "action_mode": str(action_cfg.get("mode", "off")),
        "issues": issues,
    }


def _check_recent_status(status: dict[str, Any]) -> dict[str, Any]:
    issues = []
    if not status.get("window_bound"):
        issues.append(_issue(
            "info",
            "window_not_bound",
            "还没有绑定雀魂窗口；真实抓帧前需要先绑定或允许全屏回退截图。",
            {},
        ))
    last_error = str(status.get("last_error", "")).strip()
    if last_error:
        issues.append(_issue(
            "warning",
            "recent_error",
            f"最近错误：{last_error}",
            {"last_error": last_error},
        ))
    if status.get("last_capture_ok") is False and status.get("last_frame_at"):
        issues.append(_issue(
            "warning",
            "recent_capture_failed",
            "最近一次截图失败，可以先检查窗口绑定和系统截图权限。",
            {"last_frame_at": status.get("last_frame_at", "")},
        ))
    if status.get("last_perception_ok") is False and status.get("last_perception_at"):
        issues.append(_issue(
            "warning",
            "recent_perception_failed",
            "最近一次感知失败，优先看截图路径和校准 profile。",
            {"last_perception_at": status.get("last_perception_at", "")},
        ))
    return {
        "check_id": "recent_status",
        "ok": _health_from_issues(issues) == "ok",
        "window_bound": bool(status.get("window_bound", False)),
        "last_capture_ok": status.get("last_capture_ok", False),
        "last_perception_ok": status.get("last_perception_ok", False),
        "last_error": last_error,
        "issues": issues,
    }


def _check_advice_only_registry() -> dict[str, Any]:
    registry = ActionRegistry()
    registered_ids = {action.action_id for action in registry.list_actions()}
    registered_ui_ids = sorted(set(IN_MATCH_UI_ACTION_IDS) & registered_ids)
    accepted_ui_ids = sorted(
        action_id
        for action_id in IN_MATCH_UI_ACTION_IDS
        if registry.validate(
            action_id,
            current_scene="in_match",
            action_mode="assist",
            session_running=True,
            user_confirmed=True,
        )[0]
    )
    issues = [
        _issue(
            "error",
            "in_match_ui_action_registered",
            "对局按钮被注册成可执行动作，这会破坏 advice-only 边界。",
            {"registered_ui_ids": registered_ui_ids, "accepted_ui_ids": accepted_ui_ids},
        )
    ] if registered_ui_ids or accepted_ui_ids else []
    return {
        "check_id": "advice_only_registry",
        "ok": not issues,
        "registered_ui_ids": registered_ui_ids,
        "accepted_ui_ids": accepted_ui_ids,
        "issues": issues,
    }


def _health_from_issues(issues: list[dict[str, Any]]) -> str:
    severities = {str(issue.get("severity", "")) for issue in issues}
    if "error" in severities:
        return "error"
    if "warning" in severities:
        return "warning"
    return "ok"


def _build_summary(checks: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "health": _health_from_issues(issues),
        "check_count": len(checks),
        "ok_checks": sum(1 for check in checks if check.get("ok")),
        "warning_count": sum(1 for issue in issues if issue.get("severity") == "warning"),
        "error_count": sum(1 for issue in issues if issue.get("severity") == "error"),
        "info_count": sum(1 for issue in issues if issue.get("severity") == "info"),
    }


def _template_kind_count(payload: dict[str, Any]) -> int:
    templates = payload.get("templates") if isinstance(payload, dict) else None
    if isinstance(templates, dict):
        return len(templates)
    return len(payload) if isinstance(payload, dict) else 0


def _source_sample_count(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        return 0
    templates = payload.get("templates")
    if not isinstance(templates, dict):
        templates = payload
    total = 0
    for item in templates.values():
        if isinstance(item, dict):
            total += int(item.get("source_sample_count", item.get("sample_count", 0)) or 0)
    return total


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _issue(severity: str, code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "details": details,
    }
