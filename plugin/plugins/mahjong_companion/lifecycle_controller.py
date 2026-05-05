from __future__ import annotations

import asyncio
from pathlib import Path

from plugin.sdk.plugin import Ok

from .data_lifecycle import (
    clear_calibration_raw_data,
    clear_local_runtime_data,
    describe_local_data,
    export_local_data,
)
from .diagnostics import build_runtime_diagnostics


class LifecycleControllerMixin:
    async def get_data_lifecycle(self):
        payload = await asyncio.to_thread(
            describe_local_data,
            self.plugin.data_path(),
            plugin_dir=Path(__file__).resolve().parent,
        )
        return Ok(payload)

    async def export_local_data(
        self,
        *,
        package_name: str = "",
        include_session_cache: bool = True,
        include_debug_samples: bool = True,
        include_calibration_profiles: bool = True,
        include_raw_calibration: bool = False,
    ):
        payload = await asyncio.to_thread(
            export_local_data,
            self.plugin.data_path(),
            plugin_dir=Path(__file__).resolve().parent,
            package_name=package_name,
            include_session_cache=bool(include_session_cache),
            include_debug_samples=bool(include_debug_samples),
            include_calibration_profiles=bool(include_calibration_profiles),
            include_raw_calibration=bool(include_raw_calibration),
        )
        return Ok(payload)

    async def clear_local_runtime_data(
        self,
        *,
        include_session_cache: bool = True,
        include_debug_samples: bool = True,
        include_exports: bool = False,
        dry_run: bool = True,
    ):
        if self.state.running and not dry_run:
            return Ok({
                "ok": False,
                "error": "stop session before clearing runtime data",
                "dry_run": False,
            })
        payload = await asyncio.to_thread(
            clear_local_runtime_data,
            self.plugin.data_path(),
            include_session_cache=bool(include_session_cache),
            include_debug_samples=bool(include_debug_samples),
            include_exports=bool(include_exports),
            dry_run=bool(dry_run),
        )
        return Ok(payload)

    async def clear_calibration_raw_data(self, *, dry_run: bool = True, confirm_token: str = ""):
        if self.state.running and not dry_run:
            return Ok({
                "ok": False,
                "error": "stop session before clearing calibration raw data",
                "dry_run": False,
                "confirm_token_required": "DELETE_CALIBRATION_RAW",
            })
        payload = await asyncio.to_thread(
            clear_calibration_raw_data,
            self.plugin.data_path(),
            dry_run=bool(dry_run),
            confirm_token=str(confirm_token or ""),
        )
        return Ok(payload)

    async def get_runtime_diagnostics(self):
        payload = await asyncio.to_thread(
            build_runtime_diagnostics,
            plugin_dir=Path(__file__).resolve().parent,
            data_root=self.plugin.data_path(),
            config=self._config,
            status=self.get_status(),
        )
        return Ok(payload)

