import asyncio
import builtins
from queue import Queue
from types import SimpleNamespace

import pytest

import main_logic.core as core_module
from main_logic.core import LLMSessionManager


def _make_inactive_manager(*, starting_count=1):
    mgr = LLMSessionManager.__new__(LLMSessionManager)
    mgr.lock = asyncio.Lock()
    mgr.input_cache_lock = asyncio.Lock()
    mgr.is_active = False
    mgr.session = None
    mgr._starting_session_count = starting_count
    mgr.session_ready = True
    mgr.pending_input_data = [{"input_type": "text", "data": "stale"}]
    mgr.message_handler_task = None
    mgr.tts_handler_task = None
    mgr.tts_thread = None
    mgr.tts_request_queue = Queue()
    mgr.tts_response_queue = Queue()
    mgr._audio_stream_epoch = 0
    mgr._reset_tts_retry_state = lambda: None
    mgr._clear_audio_stream_queue = lambda reason: None
    mgr._cancel_audio_stream_worker = lambda reason: None

    async def _teardown_tts_runtime(*args, **kwargs):
        return None

    mgr._teardown_tts_runtime = _teardown_tts_runtime
    return mgr


class _FakeStartConfigManager:
    def get_model_api_config(self, kind):
        return {
            "api_type": "local",
            "base_url": "http://example.test",
            "api_key": "test-key",
            "model": f"test-{kind}",
        }

    async def aget_core_config(self):
        return {
            "AUDIO_API_KEY": "audio-key",
            "CORE_API_TYPE": "local",
            "ENABLE_CUSTOM_API": False,
            "TTS_VOICE_ID": "",
        }

    def cleanup_invalid_voice_ids(self):
        return 0, []

    async def aget_character_data(self):
        return None, None, None, {}, None, None, None, None, None


class _FakeActivityTracker:
    def __init__(self):
        self.voice_modes = []

    def on_voice_mode(self, enabled):
        self.voice_modes.append(enabled)


class _EmptyToolRegistry:
    def __init__(self):
        self._tools = []

    def register(self, tool, *, replace=False):
        if replace:
            self._tools = [existing for existing in self._tools if existing.name != tool.name]
        self._tools.append(tool)

    def all(self):
        return list(self._tools)


def _make_start_session_manager(monkeypatch):
    mgr = LLMSessionManager.__new__(LLMSessionManager)
    mgr.lock = asyncio.Lock()
    mgr.input_cache_lock = asyncio.Lock()
    mgr.tts_cache_lock = asyncio.Lock()
    mgr._config_manager = _FakeStartConfigManager()
    mgr.user_language = "zh-CN"
    mgr.lanlan_name = "dadi"
    mgr.master_name = "master"
    mgr.lanlan_basic_config = {}
    mgr.voice_id = "voice"
    mgr._is_free_preset_voice = False
    mgr.memory_server_port = 48912
    mgr.tool_registry = _EmptyToolRegistry()
    mgr.session = None
    mgr.is_active = False
    mgr._starting_session_count = 0
    mgr._starting_input_mode = None
    mgr._session_start_circuit_open = False
    mgr._idle_session_reset_task = None
    mgr._ensure_idle_session_reset_loop = lambda: None
    mgr.session_start_failure_count = 0
    mgr.session_start_last_failure_time = None
    mgr.session_start_max_failures = 3
    mgr._memory_error_retry_after = 0
    mgr._memory_error_cooldown_seconds = 10
    mgr.tts_thread = None
    mgr.tts_handler_task = None
    mgr._tts_respawn_task = None
    mgr.tts_request_queue = Queue()
    mgr.tts_response_queue = Queue()
    mgr.tts_pending_chunks = []
    mgr.tts_ready = False
    mgr.pending_input_data = []
    mgr.session_ready = True
    mgr.use_tts = False
    mgr.input_mode = "text"
    mgr.current_speech_id = None
    mgr.message_handler_task = None
    mgr.pending_agent_callbacks = []
    mgr._activity_tracker = _FakeActivityTracker()

    events = []

    async def _noop_async(*args, **kwargs):
        return None

    async def _send_session_preparing(input_mode):
        events.append(("preparing", input_mode))

    async def _send_session_started(input_mode):
        events.append(("started", input_mode))

    async def _send_status(message):
        events.append(("status", message))

    async def _build_initial_prompt():
        return "initial prompt\n"

    mgr._cleanup_pending_session_resources = _noop_async
    mgr._reset_preparation_state = _noop_async
    mgr._reset_voice_echo_suppression_cache = lambda: None
    mgr.send_session_preparing = _send_session_preparing
    mgr.send_session_started = _send_session_started
    mgr.send_status = _send_status
    mgr._enqueue_voice_migration_notice = lambda legacy_names: None
    mgr._apply_voice_id_for_route = lambda base_url: None
    mgr._can_preserve_tts_ready_for_session_start = lambda: False
    mgr._resolve_session_use_tts = lambda *args, **kwargs: False
    mgr._get_text_guard_max_length = lambda: 0
    mgr._build_initial_prompt = _build_initial_prompt
    mgr._bind_session_lifecycle_callbacks = lambda session: None
    mgr._sync_tools_to_active_session = _noop_async
    mgr._flush_pending_input_data = _noop_async
    mgr._fire_task = lambda task: None
    mgr.trigger_agent_callbacks = _noop_async
    mgr.handle_text_data = _noop_async
    mgr.handle_input_transcript = _noop_async
    mgr.handle_output_transcript = _noop_async
    mgr.handle_connection_error = _noop_async
    mgr.handle_response_complete = _noop_async
    mgr.handle_repetition_detected = _noop_async
    mgr.handle_response_discarded = _noop_async
    mgr.handle_proactive_complete = _noop_async
    mgr._on_tool_call = _noop_async

    class _FakeMemoryClient:
        async def get(self, *args, **kwargs):
            return SimpleNamespace(is_success=True, status_code=200, text="memory\n")

    import utils.internal_http_client as internal_http_client

    monkeypatch.setattr(
        internal_http_client,
        "get_internal_http_client",
        lambda: _FakeMemoryClient(),
    )

    return mgr, events


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inactive_end_session_clears_starting_guard_for_frontend_timeout():
    mgr = _make_inactive_manager(starting_count=1)

    await LLMSessionManager.end_session(mgr)

    assert mgr._starting_session_count == 0
    assert mgr.session_ready is False
    assert mgr.pending_input_data == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inactive_end_session_preserves_starting_guard_for_internal_cleanup():
    mgr = _make_inactive_manager(starting_count=1)

    await LLMSessionManager.end_session(mgr, reset_starting_count=False)

    assert mgr._starting_session_count == 1
    assert mgr.session_ready is True
    assert mgr.pending_input_data == [{"input_type": "text", "data": "stale"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inactive_end_session_closes_promoted_session_before_active():
    mgr = _make_inactive_manager(starting_count=1)

    class _PromotedSession:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    promoted_session = _PromotedSession()
    mgr.session = promoted_session

    await LLMSessionManager.end_session(mgr)

    assert promoted_session.closed is True
    assert mgr.session is None
    assert mgr._starting_session_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inactive_end_session_does_not_clear_next_start_pending_input():
    mgr = _make_inactive_manager(starting_count=1)
    teardown_started = asyncio.Event()
    finish_teardown = asyncio.Event()

    async def _teardown_tts_runtime(*args, **kwargs):
        teardown_started.set()
        await finish_teardown.wait()

    mgr._teardown_tts_runtime = _teardown_tts_runtime

    end_task = asyncio.create_task(LLMSessionManager.end_session(mgr))
    await teardown_started.wait()

    assert mgr._starting_session_count == 0
    assert mgr.pending_input_data == []

    async with mgr.input_cache_lock:
        mgr._starting_session_count = 1
        mgr.session_ready = False
        mgr.pending_input_data.append({"input_type": "text", "data": "new"})

    finish_teardown.set()
    await end_task

    assert mgr._starting_session_count == 1
    assert mgr.session_ready is False
    assert mgr.pending_input_data == [{"input_type": "text", "data": "new"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_session_ignores_dead_stdout_for_initial_prompt(monkeypatch):
    mgr, events = _make_start_session_manager(monkeypatch)
    constructed = []

    class _FakeOfflineSession:
        def __init__(self, **kwargs):
            constructed.append(self)
            self.closed = False

        async def connect(self, initial_prompt, *, native_audio):
            self.initial_prompt = initial_prompt
            self.native_audio = native_audio

        async def close(self):
            self.closed = True

        async def handle_messages(self):
            return None

    def _raise_dead_stdout(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(core_module, "OmniOfflineClient", _FakeOfflineSession)
    monkeypatch.setattr(builtins, "print", _raise_dead_stdout)

    await LLMSessionManager.start_session(mgr, object(), input_mode="text")
    await asyncio.sleep(0)

    assert events == [("preparing", "text"), ("started", "text")]
    assert mgr._starting_session_count == 0
    assert mgr._starting_input_mode is None
    assert mgr.is_active is True
    assert mgr.session is constructed[0]
    assert constructed[0].initial_prompt.startswith("initial prompt\nmemory\n")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_start_session_cas_abort_releases_starting_guard(monkeypatch):
    mgr, events = _make_start_session_manager(monkeypatch)
    winner_session = object()
    constructed = []

    class _FakeOfflineSession:
        def __init__(self, **kwargs):
            constructed.append(self)
            self.closed = False

        async def connect(self, initial_prompt, *, native_audio):
            mgr.session = winner_session

        async def close(self):
            self.closed = True

        async def handle_messages(self):
            return None

    monkeypatch.setattr(core_module, "OmniOfflineClient", _FakeOfflineSession)

    await LLMSessionManager.start_session(mgr, object(), input_mode="text")

    assert events == [("preparing", "text")]
    assert constructed[0].closed is True
    assert mgr.session is winner_session
    assert mgr.is_active is False
    assert mgr._starting_session_count == 0
    assert mgr._starting_input_mode is None
