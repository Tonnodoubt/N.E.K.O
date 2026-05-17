from pathlib import Path


APP_WEBSOCKET_PATH = Path(__file__).resolve().parents[2] / "static" / "app-websocket.js"


def test_response_discarded_visible_in_react_chat():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "function appendAssistantStatusMessage(text)" in source
    assert "window.reactChatWindowHost.appendMessage({" in source
    assert "appendAssistantStatusMessage(translatedDiscardMsg);" in source

    helper_block = source.split("function appendAssistantStatusMessage(text)", 1)[1].split(
        "function websocketTraceEnabled()",
        1,
    )[0]
    assert helper_block.index("window.reactChatWindowHost.appendMessage({") < helper_block.index(
        "document.createElement('div')"
    )
    assert "status: 'failed'" in helper_block
    assert "window.currentGeminiMessage" not in helper_block

    response_discarded_block = source.split("// -------- response_discarded --------", 1)[1].split(
        "// -------- user_transcript --------",
        1,
    )[0]
    assert "document.createElement('div')" not in response_discarded_block
    assert "appendChild(messageDiv)" not in response_discarded_block


def test_mobile_takeover_guard_exists_for_single_websocket_desktop_yield():
    source = APP_WEBSOCKET_PATH.read_text(encoding="utf-8")

    assert "mod.setDesktopMobileTakeover = async function (enabled)" in source
    assert "if (S.mobileTakeoverActive) {" in source
    assert "Desktop websocket yielded to mobile" in source
    assert "connect skipped: desktop yielded to mobile" in source
    assert "window.__nekoSetDesktopMobileTakeover = mod.setDesktopMobileTakeover;" in source
