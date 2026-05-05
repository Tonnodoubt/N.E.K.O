from __future__ import annotations

import pytest

from plugin.plugins.mahjong_companion.smoke_test import run_v1_to_v9_smoke


@pytest.mark.asyncio
async def test_v1_to_v9_smoke_runner_reports_success() -> None:
    payload = await run_v1_to_v9_smoke()

    assert payload["ok"] is True
    assert len(payload["results"]) == 3
    assert all(item["ok"] for item in payload["results"])
    assert "latest_session.json" in payload["session_cache_files"]
