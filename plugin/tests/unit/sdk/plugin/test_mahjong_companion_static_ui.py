from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from plugin.plugins.mahjong_companion import MahjongCompanionPlugin


def test_static_ui_asset_sync_skips_in_place_static_directory() -> None:
    plugin_dir = Path(__file__).resolve().parents[4] / "plugins" / "mahjong_companion"
    plugin = MahjongCompanionPlugin.__new__(MahjongCompanionPlugin)
    plugin.ctx = SimpleNamespace(config_path=plugin_dir / "plugin.toml")

    with patch("plugin.plugins.mahjong_companion.shutil.copy2") as copy2:
        assert plugin._ensure_static_ui_assets() is True

    copy2.assert_not_called()


def test_static_ui_uses_current_run_protocol() -> None:
    plugin_dir = Path(__file__).resolve().parents[4] / "plugins" / "mahjong_companion"
    main_js = (plugin_dir / "static" / "main.js").read_text(encoding="utf-8")

    assert 'fetchJson("/runs"' in main_js
    assert '"/plugin/trigger"' not in main_js


def test_static_ui_keeps_controls_ahead_of_debug_json() -> None:
    plugin_dir = Path(__file__).resolve().parents[4] / "plugins" / "mahjong_companion"
    index_html = (plugin_dir / "static" / "index.html").read_text(encoding="utf-8")
    style_css = (plugin_dir / "static" / "style.css").read_text(encoding="utf-8")

    assert 'class="app-header"' in index_html
    assert 'class="panel operations-panel"' in index_html
    assert 'class="panel status-panel"' in index_html
    assert '<details class="debug-json">' in index_html
    assert '<summary>完整状态 JSON</summary>' in index_html
    assert ".operations-panel" in style_css
    assert "order: 1;" in style_css
    assert "max-height: 260px;" in style_css


def test_static_ui_defaults_to_compact_player_view() -> None:
    plugin_dir = Path(__file__).resolve().parents[4] / "plugins" / "mahjong_companion"
    index_html = (plugin_dir / "static" / "index.html").read_text(encoding="utf-8")
    main_js = (plugin_dir / "static" / "main.js").read_text(encoding="utf-8")

    assert 'class="quick-status-grid"' in index_html
    assert 'id="quick-suggestion"' in index_html
    assert 'id="quick-scene"' in index_html
    assert 'id="quick-hand"' in index_html
    assert 'id="quick-window"' in index_html
    assert 'id="quick-session"' in index_html
    assert 'id="quick-capture"' in index_html
    assert 'id="quick-voice"' in index_html
    assert 'id="quick-error"' in index_html
    assert 'id="quick-marker"' not in index_html
    assert 'id="quick-overlay"' not in index_html
    assert 'id="window-select"' in index_html
    assert 'id="refresh-windows-btn"' in index_html
    assert 'id="show-overlay-btn"' not in index_html
    assert 'id="hide-overlay-btn"' not in index_html
    assert 'class="main-actions primary-actions"' in index_html
    assert "刷新屏幕并给建议" in index_html
    assert "<summary>详细状态</summary>" in index_html
    assert "<summary>模式与调试</summary>" in index_html
    assert "<summary>辅助操作</summary>" not in index_html
    assert "复盘与训练" not in index_html
    assert "生成复盘摘要" not in index_html
    assert "查看训练话题" not in index_html
    assert 'id="review-summary-btn"' not in index_html
    assert 'id="coaching-topics-btn"' not in index_html
    assert 'value="summarize_review"' not in index_html
    assert 'value="sync_memory"' not in index_html
    assert 'setText("quick-suggestion"' in main_js
    assert 'setText("quick-scene"' in main_js
    assert 'setText("quick-hand"' in main_js
    assert 'setText("quick-window"' in main_js
    assert 'setText("quick-session"' in main_js
    assert 'setText("quick-capture"' in main_js
    assert 'setText("quick-voice"' in main_js
    assert 'setText("quick-error"' in main_js
    assert 'setText("quick-marker"' not in main_js
    assert 'setText("quick-overlay"' not in main_js
    assert "topDiscardCandidate" in main_js
    assert 'callEntry("list_window_candidates")' in main_js
    assert 'callEntry("bind_window", { window_title: windowTitle })' in main_js
    assert 'runAction("show_overlay")' not in main_js
    assert 'runAction("hide_overlay")' not in main_js
