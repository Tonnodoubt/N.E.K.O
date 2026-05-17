from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from plugin.sdk.plugin import (
    NekoPluginBase,
    neko_plugin,
    plugin_entry,
    lifecycle,
    Ok,
    Err,
    SdkError,
    get_plugin_logger,
)

from .config_defaults import DEFAULT_CONFIG, merge_runtime_config
from .orchestrator import SessionOrchestrator


@neko_plugin
class MahjongCompanionPlugin(NekoPluginBase):
    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = get_plugin_logger(__name__)
        self.orchestrator = SessionOrchestrator(self)

    @lifecycle(id="startup")
    async def startup(self, **_):
        cfg = await self.config.dump(timeout=5.0)
        merged = merge_runtime_config(DEFAULT_CONFIG, cfg if isinstance(cfg, dict) else {})
        self.orchestrator.apply_config(merged)
        self.orchestrator.load_cached_outputs()
        seeded_assets = self._ensure_runtime_data_assets()
        if seeded_assets:
            self.logger.info("mahjong companion runtime data assets seeded: {}", seeded_assets)

        if self._ensure_static_ui_assets():
            ok = self.register_static_ui(
                "static",
                index_file="index.html",
                cache_control="no-cache, no-store, must-revalidate",
            )
            if ok:
                self.logger.info("mahjong companion static ui registered at /plugin/{}/ui/", self.plugin_id)
            else:
                self.logger.warning("mahjong companion static ui registration failed")
        else:
            self.logger.warning("mahjong companion bundled static ui not found")

        self.report_status(self.orchestrator.get_status())
        return Ok({"status": "ready"})

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        await self.orchestrator.shutdown()
        self.report_status(self.orchestrator.get_status())
        return Ok({"status": "stopped"})

    @lifecycle(id="config_change")
    async def on_config_change(self, **_):
        cfg = await self.config.dump(timeout=5.0)
        merged = merge_runtime_config(DEFAULT_CONFIG, cfg if isinstance(cfg, dict) else {})
        self.orchestrator.apply_config(merged)
        return Ok({"reloaded": True, "mode": self.orchestrator.state.mode})

    @plugin_entry(id="start_session", name="启动会话", kind="action")
    async def start_session(self, **_):
        return await self.orchestrator.start()

    @plugin_entry(id="stop_session", name="停止会话", kind="action")
    async def stop_session(self, **_):
        return await self.orchestrator.stop()

    @plugin_entry(id="get_session_status", name="获取会话状态", kind="action")
    async def get_session_status(self, **_):
        return Ok(self.orchestrator.get_status())

    @plugin_entry(id="get_data_lifecycle", name="查看本地数据", kind="action")
    async def get_data_lifecycle(self, **_):
        return await self.orchestrator.get_data_lifecycle()

    @plugin_entry(id="export_local_data", name="导出本地数据包", kind="action")
    async def export_local_data(
        self,
        package_name: str = "",
        include_session_cache: bool = True,
        include_debug_samples: bool = True,
        include_calibration_profiles: bool = True,
        include_raw_calibration: bool = False,
        **_,
    ):
        return await self.orchestrator.export_local_data(
            package_name=package_name,
            include_session_cache=include_session_cache,
            include_debug_samples=include_debug_samples,
            include_calibration_profiles=include_calibration_profiles,
            include_raw_calibration=include_raw_calibration,
        )

    @plugin_entry(id="clear_local_runtime_data", name="清理本地运行数据", kind="action")
    async def clear_local_runtime_data(
        self,
        include_session_cache: bool = True,
        include_debug_samples: bool = True,
        include_exports: bool = False,
        dry_run: bool = True,
        **_,
    ):
        return await self.orchestrator.clear_local_runtime_data(
            include_session_cache=include_session_cache,
            include_debug_samples=include_debug_samples,
            include_exports=include_exports,
            dry_run=dry_run,
        )

    @plugin_entry(id="clear_calibration_raw_data", name="清理原始校准素材", kind="action")
    async def clear_calibration_raw_data(self, dry_run: bool = True, confirm_token: str = "", **_):
        return await self.orchestrator.clear_calibration_raw_data(
            dry_run=dry_run,
            confirm_token=confirm_token,
        )

    @plugin_entry(id="get_runtime_diagnostics", name="查看运行诊断", kind="action")
    async def get_runtime_diagnostics(self, **_):
        return await self.orchestrator.get_runtime_diagnostics()

    @plugin_entry(id="set_mode", name="设置模式", kind="action")
    async def set_mode(self, mode: str, **_):
        if mode not in {"spectate", "replay", "teaching", "silent"}:
            return Err(SdkError(f"invalid mode: {mode}"))
        await self.orchestrator.set_mode(mode)
        return Ok(self.orchestrator.get_status())

    @plugin_entry(id="set_runtime_mode", name="设置运行时模式", kind="action")
    async def set_runtime_mode(self, mode: str, **_):
        return await self.orchestrator.set_runtime_mode(mode)

    @plugin_entry(id="capture_debug_frame", name="抓取调试帧", kind="action")
    async def capture_debug_frame(self, **_):
        return await self.orchestrator.capture_debug_frame()

    @plugin_entry(id="analyze_debug_frame", name="分析最近截图", kind="action")
    async def analyze_debug_frame(self, **_):
        return await self.orchestrator.analyze_debug_frame()

    @plugin_entry(id="analyze_frame_path", name="分析指定截图", kind="action")
    async def analyze_frame_path(self, frame_path: str, **_):
        return await self.orchestrator.analyze_frame_path(frame_path)

    @plugin_entry(id="get_last_perception", name="获取最近感知结果", kind="action")
    async def get_last_perception(self, **_):
        return await self.orchestrator.get_last_perception()

    @plugin_entry(id="generate_decision", name="生成决策结果", kind="action")
    async def generate_decision(self, **_):
        return await self.orchestrator.generate_decision()

    @plugin_entry(id="get_last_decision", name="获取最近决策结果", kind="action")
    async def get_last_decision(self, **_):
        return await self.orchestrator.get_last_decision()

    @plugin_entry(id="generate_narration", name="生成讲解结果", kind="action")
    async def generate_narration(self, **_):
        return await self.orchestrator.generate_narration()

    @plugin_entry(id="get_last_narration", name="获取最近讲解结果", kind="action")
    async def get_last_narration(self, **_):
        return await self.orchestrator.get_last_narration()

    @plugin_entry(id="preview_companion_view", name="预览陪伴视图", kind="action")
    async def preview_companion_view(self, **_):
        return await self.orchestrator.preview_companion_view()

    @plugin_entry(id="run_companion_pipeline", name="跑完整陪伴链路", kind="action")
    async def run_companion_pipeline(
        self,
        frame_path: str = "",
        capture: bool = False,
        dispatch: bool = True,
        force_reply: bool = True,
        **_,
    ):
        return await self.orchestrator.run_companion_pipeline(
            frame_path=frame_path,
            capture=bool(capture),
            dispatch=bool(dispatch),
            force_reply=bool(force_reply),
        )

    @plugin_entry(id="speak_last_narration", name="播报最近讲解", kind="action")
    async def speak_last_narration(self, **_):
        return await self.orchestrator.speak_last_narration()

    @plugin_entry(id="cycle_voice_mode", name="切换语音模式", kind="action")
    async def cycle_voice_mode(self, **_):
        return await self.orchestrator.cycle_voice_mode()

    @plugin_entry(id="set_voice_mode", name="设置语音模式", kind="action")
    async def set_voice_mode(self, mode: str, **_):
        return await self.orchestrator.set_voice_mode(mode)

    @plugin_entry(id="set_unified_mode", name="设置统一工作模式", kind="action")
    async def set_unified_mode(self, mode: str, **_):
        return await self.orchestrator.set_unified_mode(mode)

    @plugin_entry(id="show_overlay", name="显示悬浮建议", kind="action")
    async def show_overlay(self, **_):
        return await self.orchestrator.show_overlay()

    @plugin_entry(id="hide_overlay", name="隐藏悬浮建议", kind="action")
    async def hide_overlay(self, **_):
        return await self.orchestrator.hide_overlay()

    @plugin_entry(id="list_window_candidates", name="列出可绑定窗口", kind="action")
    async def list_window_candidates(self, **_):
        return await self.orchestrator.list_window_candidates()

    @plugin_entry(id="bind_window", name="尝试绑定窗口", kind="action")
    async def bind_window(self, window_title: str = "", **_):
        return await self.orchestrator.bind_window(window_title=window_title)

    @plugin_entry(id="unbind_window", name="解除窗口绑定", kind="action")
    async def unbind_window(self, **_):
        return await self.orchestrator.unbind_window()

    def _ensure_runtime_data_assets(self) -> dict[str, int]:
        """Seed bundled runtime assets into the per-user data directory.

        The SDK stores plugin runtime data under AppData/Local on Windows, while
        bundled calibration profiles and compact ONNX models live next to the
        plugin source.  Missing seeds leave the live pipeline on heuristic
        layouts, which is too weak for Mahjong Soul screenshots.
        """
        copied: dict[str, int] = {}
        asset_groups = (
            (
                Path("data") / "calibration" / "profiles",
                self.data_path("calibration", "profiles"),
                "calibration_profiles",
            ),
            (
                Path("data") / "models" / "vit_tile_classifier",
                self.data_path("models", "vit_tile_classifier"),
                "onnx_tile_model",
            ),
        )
        plugin_root = Path(__file__).resolve().parent
        for relative_source, target_dir, group_id in asset_groups:
            source_dir = plugin_root / relative_source
            copied[group_id] = self._copy_bundled_asset_tree(source_dir, Path(target_dir))
        return {key: value for key, value in copied.items() if value}

    def _copy_bundled_asset_tree(self, source_dir: Path, target_dir: Path) -> int:
        if not source_dir.is_dir():
            return 0
        copied = 0
        for source_path in source_dir.rglob("*"):
            if source_path.is_dir():
                continue
            relative = source_path.relative_to(source_dir)
            target_path = target_dir / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.is_file() and target_path.stat().st_size == source_path.stat().st_size:
                continue
            shutil.copy2(source_path, target_path)
            copied += 1
        return copied

    def _ensure_static_ui_assets(self) -> bool:
        source_dir = Path(__file__).resolve().parent / "static"
        index_path = source_dir / "index.html"
        if not source_dir.is_dir() or not index_path.is_file():
            return False

        target_dir = self.config_dir / "static"
        if source_dir.resolve() == target_dir.resolve():
            return True

        for source_path in source_dir.rglob("*"):
            relative = source_path.relative_to(source_dir)
            target_path = target_dir / relative
            if source_path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        return True
