from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
import zipfile


SCHEMA_VERSION = "mahjong-companion-data-lifecycle-v1"
EXPORT_ROOT_NAME = "exports"
PACKAGE_ROOT = "mahjong_companion_data"


@dataclass(frozen=True)
class DataGroupSpec:
    group_id: str
    relative_path: tuple[str, ...]
    label: str
    description: str
    default_export: bool = False
    clearable_runtime: bool = False
    contains_private_frames: bool = False

    @property
    def archive_path(self) -> str:
        return "/".join(self.relative_path)


DATA_GROUPS: tuple[DataGroupSpec, ...] = (
    DataGroupSpec(
        group_id="session_cache",
        relative_path=("session_cache",),
        label="运行缓存",
        description="会话状态、动作日志、复盘候选、复盘摘要、训练趋势和记忆桥队列。",
        default_export=True,
        clearable_runtime=True,
    ),
    DataGroupSpec(
        group_id="debug_samples",
        relative_path=("debug_samples",),
        label="调试截图",
        description="用户主动抓取或调试时保存的截图、感知 overlay 和调试 JSON。",
        default_export=True,
        clearable_runtime=True,
        contains_private_frames=True,
    ),
    DataGroupSpec(
        group_id="calibration_profiles",
        relative_path=("calibration", "profiles"),
        label="校准 profile",
        description="从本地标注样本训练出的模板签名和 profile。",
        default_export=True,
    ),
    DataGroupSpec(
        group_id="calibration_raw",
        relative_path=("calibration", "raw"),
        label="原始校准素材",
        description="本地原始截图和 sidecar 标注，可能包含完整牌局画面；默认不导出、不清理。",
        contains_private_frames=True,
    ),
    DataGroupSpec(
        group_id="exports",
        relative_path=(EXPORT_ROOT_NAME,),
        label="导出包",
        description="由数据生命周期入口生成的本地 zip 导出包。",
        clearable_runtime=True,
    ),
)

DATA_GROUP_BY_ID = {group.group_id: group for group in DATA_GROUPS}
DEFAULT_EXPORT_GROUPS = tuple(group.group_id for group in DATA_GROUPS if group.default_export)
RUNTIME_CLEARABLE_GROUPS = tuple(group.group_id for group in DATA_GROUPS if group.clearable_runtime)
PROTECTED_GROUPS = tuple(
    group.group_id for group in DATA_GROUPS if not group.clearable_runtime and not group.group_id == "exports"
)


def describe_local_data(data_root: Path, *, plugin_dir: Path | None = None) -> dict[str, Any]:
    data_root = _resolve_data_root(data_root)
    groups = []
    for spec in DATA_GROUPS:
        group_path = _resolve_group_path(data_root, spec)
        stats = _summarize_path(group_path)
        groups.append({
            "group_id": spec.group_id,
            "label": spec.label,
            "description": spec.description,
            "path": str(group_path),
            "relative_path": spec.archive_path,
            "exists": group_path.exists(),
            "default_export": spec.default_export,
            "clearable_runtime": spec.clearable_runtime,
            "contains_private_frames": spec.contains_private_frames,
            **stats,
        })

    lifecycle_doc_path = Path(plugin_dir).resolve() / "DATA_LIFECYCLE.md" if plugin_dir is not None else None
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "data_root": str(data_root),
        "default_export_groups": list(DEFAULT_EXPORT_GROUPS),
        "runtime_clearable_groups": list(RUNTIME_CLEARABLE_GROUPS),
        "protected_groups": list(PROTECTED_GROUPS),
        "groups": groups,
        "lifecycle_doc_path": str(lifecycle_doc_path) if lifecycle_doc_path is not None else "",
        "lifecycle_doc_exists": bool(lifecycle_doc_path and lifecycle_doc_path.exists()),
        "upload_boundary": {
            "raw_screenshots_auto_uploaded": False,
            "full_game_log_auto_uploaded": False,
            "calibration_raw_default_export": False,
        },
    }


def export_local_data(
    data_root: Path,
    *,
    plugin_dir: Path | None = None,
    package_name: str = "",
    include_session_cache: bool = True,
    include_debug_samples: bool = True,
    include_calibration_profiles: bool = True,
    include_raw_calibration: bool = False,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    data_root = _resolve_data_root(data_root)
    selected_groups = _selected_export_groups(
        include_session_cache=include_session_cache,
        include_debug_samples=include_debug_samples,
        include_calibration_profiles=include_calibration_profiles,
        include_raw_calibration=include_raw_calibration,
    )
    if not selected_groups:
        return {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "error": "no data groups selected",
            "selected_groups": [],
        }

    output_dir = _resolve_exports_dir(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _format_package_timestamp(created_at or datetime.now(timezone.utc))
    package_path = output_dir / _safe_package_name(package_name, timestamp)
    if not _is_under(package_path, output_dir):
        raise ValueError(f"export package path escapes exports dir: {package_path}")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "data_root": str(data_root),
        "selected_groups": list(selected_groups),
        "excluded_groups": [group.group_id for group in DATA_GROUPS if group.group_id not in selected_groups],
        "raw_calibration_included": bool(include_raw_calibration),
        "files": [],
        "missing_groups": [],
        "skipped_files": [],
    }
    if include_raw_calibration:
        manifest["privacy_warning"] = "calibration_raw may contain full game screenshots and is never exported by default"

    total_files = 0
    total_bytes = 0
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for group_id in selected_groups:
            spec = DATA_GROUP_BY_ID[group_id]
            group_root = _resolve_group_path(data_root, spec)
            if not group_root.exists():
                manifest["missing_groups"].append(group_id)
                continue
            files, skipped = _collect_export_files(group_root)
            manifest["skipped_files"].extend({
                "group_id": group_id,
                "path": str(path),
                "reason": reason,
            } for path, reason in skipped)
            for file_path in files:
                rel_path = file_path.relative_to(group_root)
                archive_name = f"{PACKAGE_ROOT}/{spec.archive_path}/{rel_path.as_posix()}"
                archive.write(file_path, archive_name)
                file_size = _file_size(file_path)
                total_files += 1
                total_bytes += file_size
                manifest["files"].append({
                    "group_id": group_id,
                    "archive_path": archive_name,
                    "size_bytes": file_size,
                })

        lifecycle_doc = Path(plugin_dir).resolve() / "DATA_LIFECYCLE.md" if plugin_dir is not None else None
        if lifecycle_doc is not None and lifecycle_doc.is_file():
            archive.write(lifecycle_doc, f"{PACKAGE_ROOT}/docs/DATA_LIFECYCLE.md")
            manifest["lifecycle_doc_included"] = True
        else:
            manifest["lifecycle_doc_included"] = False

        archive.writestr(
            f"{PACKAGE_ROOT}/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    package_size = _file_size(package_path)
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "package_path": str(package_path),
        "package_size_bytes": package_size,
        "file_count": total_files,
        "total_bytes": total_bytes,
        "selected_groups": list(selected_groups),
        "excluded_groups": manifest["excluded_groups"],
        "raw_calibration_included": bool(include_raw_calibration),
        "manifest": manifest,
    }


def clear_local_runtime_data(
    data_root: Path,
    *,
    include_session_cache: bool = True,
    include_debug_samples: bool = True,
    include_exports: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    data_root = _resolve_data_root(data_root)
    selected_groups = _selected_clear_groups(
        include_session_cache=include_session_cache,
        include_debug_samples=include_debug_samples,
        include_exports=include_exports,
    )
    if not selected_groups:
        return {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "error": "no runtime data groups selected",
            "dry_run": bool(dry_run),
            "selected_groups": [],
        }

    group_results = []
    for group_id in selected_groups:
        spec = DATA_GROUP_BY_ID[group_id]
        if not spec.clearable_runtime:
            raise ValueError(f"{group_id} is not a clearable runtime data group")
        group_path = _resolve_group_path(data_root, spec)
        before = _summarize_path(group_path)
        planned_entries = _clearable_entries(group_path)
        removed_entries: list[str] = []
        if not dry_run:
            removed_entries = _remove_entries(planned_entries, group_path)
        after = _summarize_path(group_path)
        group_results.append({
            "group_id": group_id,
            "path": str(group_path),
            "exists": group_path.exists(),
            "before": before,
            "after": after,
            "planned_entries": [str(path) for path in planned_entries],
            "removed_entries": removed_entries,
        })

    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "dry_run": bool(dry_run),
        "data_root": str(data_root),
        "selected_groups": list(selected_groups),
        "protected_groups": ["calibration_profiles", "calibration_raw"],
        "groups": group_results,
    }


def clear_calibration_raw_data(
    data_root: Path,
    *,
    dry_run: bool = True,
    confirm_token: str = "",
) -> dict[str, Any]:
    data_root = _resolve_data_root(data_root)
    spec = DATA_GROUP_BY_ID["calibration_raw"]
    group_path = _resolve_group_path(data_root, spec)
    before = _summarize_path(group_path)
    planned_entries = _clearable_entries(group_path)
    confirmation_required = "DELETE_CALIBRATION_RAW"
    if not dry_run and confirm_token != confirmation_required:
        return {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "dry_run": bool(dry_run),
            "data_root": str(data_root),
            "selected_groups": ["calibration_raw"],
            "error": "confirm_token must be DELETE_CALIBRATION_RAW to remove raw calibration screenshots",
            "confirm_token_required": confirmation_required,
            "groups": [
                {
                    "group_id": "calibration_raw",
                    "path": str(group_path),
                    "exists": group_path.exists(),
                    "before": before,
                    "after": before,
                    "planned_entries": [str(path) for path in planned_entries],
                    "removed_entries": [],
                }
            ],
            "protected_groups": ["calibration_profiles"],
            "privacy_note": "calibration raw screenshots may contain full game frames and are not needed at runtime once profiles and eval fixtures are saved",
        }

    removed_entries: list[str] = []
    if not dry_run:
        removed_entries = _remove_entries(planned_entries, group_path)
    after = _summarize_path(group_path)
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "dry_run": bool(dry_run),
        "data_root": str(data_root),
        "selected_groups": ["calibration_raw"],
        "protected_groups": ["calibration_profiles"],
        "privacy_note": "calibration raw screenshots may contain full game frames and are not needed at runtime once profiles and eval fixtures are saved",
        "groups": [
            {
                "group_id": "calibration_raw",
                "path": str(group_path),
                "exists": group_path.exists(),
                "before": before,
                "after": after,
                "planned_entries": [str(path) for path in planned_entries],
                "removed_entries": removed_entries,
            }
        ],
    }


def _selected_export_groups(
    *,
    include_session_cache: bool,
    include_debug_samples: bool,
    include_calibration_profiles: bool,
    include_raw_calibration: bool,
) -> tuple[str, ...]:
    selected: list[str] = []
    if include_session_cache:
        selected.append("session_cache")
    if include_debug_samples:
        selected.append("debug_samples")
    if include_calibration_profiles:
        selected.append("calibration_profiles")
    if include_raw_calibration:
        selected.append("calibration_raw")
    return tuple(selected)


def _selected_clear_groups(
    *,
    include_session_cache: bool,
    include_debug_samples: bool,
    include_exports: bool,
) -> tuple[str, ...]:
    selected: list[str] = []
    if include_session_cache:
        selected.append("session_cache")
    if include_debug_samples:
        selected.append("debug_samples")
    if include_exports:
        selected.append("exports")
    return tuple(selected)


def _resolve_data_root(data_root: Path) -> Path:
    return Path(data_root).expanduser().resolve()


def _resolve_exports_dir(data_root: Path) -> Path:
    exports_dir = (data_root / EXPORT_ROOT_NAME).resolve()
    if not _is_under(exports_dir, data_root):
        raise ValueError(f"exports dir escapes data root: {exports_dir}")
    return exports_dir


def _resolve_group_path(data_root: Path, spec: DataGroupSpec) -> Path:
    group_path = data_root.joinpath(*spec.relative_path).resolve()
    if not _is_under(group_path, data_root):
        raise ValueError(f"data group {spec.group_id} escapes data root: {group_path}")
    return group_path


def _summarize_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "file_count": 0,
            "dir_count": 0,
            "size_bytes": 0,
            "skipped_symlink_count": 0,
            "last_modified_at": "",
        }

    file_count = 0
    dir_count = 0
    size_bytes = 0
    skipped_symlink_count = 0
    last_mtime = 0.0
    for item in _iter_path_members(path):
        try:
            stat = item.lstat()
        except OSError:
            continue
        last_mtime = max(last_mtime, stat.st_mtime)
        if item.is_symlink():
            skipped_symlink_count += 1
            file_count += 1
            size_bytes += stat.st_size
        elif item.is_file():
            file_count += 1
            size_bytes += stat.st_size
        elif item.is_dir():
            dir_count += 1

    try:
        last_mtime = max(last_mtime, path.stat().st_mtime)
    except OSError:
        pass
    return {
        "file_count": file_count,
        "dir_count": dir_count,
        "size_bytes": size_bytes,
        "skipped_symlink_count": skipped_symlink_count,
        "last_modified_at": _iso_from_timestamp(last_mtime) if last_mtime else "",
    }


def _iter_path_members(path: Path) -> Iterator[Path]:
    """Lazily yield every path member under `path`.

    Previously this returned `sorted(path.rglob("*"))`, which materialised the
    entire tree (and a sorted copy of it) before any consumer ran. With a
    long-running `debug_samples` directory of tens of thousands of PNGs, that
    blocked the entire status / export pipeline. Now consumers see a streaming
    iterator and can sort only the post-filter slice they actually need
    (CODE_REVIEW_v1.2 N-M1).
    """
    if path.is_file() or path.is_symlink():
        yield path
        return
    if not path.is_dir():
        return
    try:
        yield from path.rglob("*")
    except OSError:
        return


def _collect_export_files(root: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    files: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for path in _iter_path_members(root):
        try:
            if path.is_symlink():
                skipped.append((path, "symlink"))
                continue
            if not path.is_file():
                continue
            if path.name.endswith(".lock") or path.name.endswith(".tmp"):
                skipped.append((path, "runtime_lock_or_temp"))
                continue
            files.append(path)
        except OSError:
            skipped.append((path, "os_error"))
    # Keep archive layout deterministic; sort the post-filter slice instead of
    # the entire tree.
    files.sort(key=lambda item: item.as_posix())
    skipped.sort(key=lambda pair: pair[0].as_posix())
    return files, skipped


def _clearable_entries(group_path: Path) -> list[Path]:
    if not group_path.exists() or not group_path.is_dir():
        return []
    entries: list[Path] = []
    for child in sorted(group_path.iterdir(), key=lambda item: item.as_posix()):
        if child.name == ".gitkeep":
            continue
        entries.append(child)
    return entries


def _remove_entries(entries: list[Path], group_path: Path) -> list[str]:
    removed: list[str] = []
    group_root = group_path.resolve()
    for entry in entries:
        if entry.name == ".gitkeep":
            continue
        if entry.is_symlink():
            entry.unlink(missing_ok=True)
            removed.append(str(entry))
            continue
        if not _is_under(entry.resolve(), group_root):
            raise ValueError(f"refusing to remove path outside runtime group: {entry}")
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)
        removed.append(str(entry))
    return removed


def _safe_package_name(raw_name: str, timestamp: str) -> str:
    fallback = f"mahjong-companion-data-{timestamp}.zip"
    value = str(raw_name).strip()
    if not value:
        return fallback
    name = Path(value).name
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in name)
    safe = safe.lstrip(".")
    if not safe or safe.lower() == "zip":
        return fallback
    return safe


def _format_package_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _is_under(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
