import asyncio
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.websockets import WebSocketDisconnect
from httpx import ASGITransport, AsyncClient

import lan_proxy
from lan_proxy import LanProxy

pages_router = importlib.import_module("main_routers.pages_router")
websocket_router = importlib.import_module("main_routers.websocket_router")


@pytest.fixture(scope="session", autouse=True)
def mock_memory_server():
    """Override the global test memory server fixture; these contract tests do not need it."""
    yield None


class _FakeQrImage:
    def __init__(self, payload: str | None = None):
        self.payload = payload

    def save(self, buf, format: str) -> None:
        buf.write(b"fake-png")


def _build_pages_app() -> FastAPI:
    app = FastAPI()
    app.include_router(pages_router.router)
    return app


@pytest.mark.asyncio
async def test_getipqrcode_returns_raw_access_url_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_make(payload: str) -> _FakeQrImage:
        captured["payload"] = payload
        return _FakeQrImage(payload)

    monkeypatch.setitem(sys.modules, "qrcode", SimpleNamespace(make=fake_make))
    monkeypatch.setattr(pages_router, "_get_lan_ip", lambda: "192.168.31.9")
    monkeypatch.setattr(pages_router, "_get_default_character_name", lambda: "Lan Lan")

    app = _build_pages_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/getipqrcode")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-neko-access-url"] == "192.168.31.9:48911?name=Lan%20Lan"
    assert captured["payload"] == "192.168.31.9:48911?name=Lan%20Lan"
    assert response.content == b"fake-png"


@pytest.mark.asyncio
async def test_qr_page_defaults_to_deeplink_qr_endpoint() -> None:
    app = _build_pages_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/qr")

    assert response.status_code == 200
    assert "N.E.K.O 连接二维码" in response.text
    assert '/getipqrcode?format=deeplink&scheme=nekorn&path=main' in response.text


@pytest.mark.asyncio
async def test_lan_proxy_info_falls_back_to_default_character(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_lan_proxy = SimpleNamespace(
        get_proxy_info_from_file=lambda: {
            "lan_ip": "192.168.0.7",
            "port": 48920,
            "token": "secret-token",
        }
    )
    monkeypatch.setitem(sys.modules, "lan_proxy", fake_lan_proxy)
    monkeypatch.setattr(pages_router, "_get_default_character_name", lambda: "Momo")

    app = _build_pages_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/lan-proxy/info")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": True,
        "lan_ip": "192.168.0.7",
        "port": 48920,
        "token": "secret-token",
        "character": "Momo",
    }


@pytest.mark.asyncio
async def test_lan_proxy_qrcode_payload_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_make(payload: str) -> _FakeQrImage:
        captured["payload"] = payload
        return _FakeQrImage(payload)

    fake_lan_proxy = SimpleNamespace(
        get_proxy_info_from_file=lambda: {
            "lan_ip": "10.0.0.5",
            "port": 48920,
            "token": "abc123",
        }
    )
    monkeypatch.setitem(sys.modules, "qrcode", SimpleNamespace(make=fake_make))
    monkeypatch.setitem(sys.modules, "lan_proxy", fake_lan_proxy)
    monkeypatch.setattr(pages_router, "_get_default_character_name", lambda: "Neko")

    app = _build_pages_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/lanproxyqrcode")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-lan-ip"] == "10.0.0.5"
    assert response.headers["x-port"] == "48920"
    assert response.headers["x-token"] == "abc123"
    assert json.loads(captured["payload"]) == {
        "lan_ip": "10.0.0.5",
        "port": 48920,
        "token": "abc123",
        "character": "Neko",
    }


@pytest.mark.asyncio
async def test_lan_proxy_token_middleware_accepts_query_header_and_public_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(LanProxy, "_get_lan_ip", lambda self: "192.168.0.10")
    proxy = LanProxy(enable_cloud=False)
    proxy.token = "super-secret"
    handler = AsyncMock(return_value="ok")

    query_request = SimpleNamespace(
        method="GET",
        path="/api/config/preferences",
        query={"token": "super-secret"},
        headers={},
    )
    assert await proxy.token_middleware(query_request, handler) == "ok"

    header_request = SimpleNamespace(
        method="GET",
        path="/api/config/preferences",
        query={},
        headers={"X-Proxy-Token": "super-secret"},
    )
    assert await proxy.token_middleware(header_request, handler) == "ok"

    public_request = SimpleNamespace(
        method="GET",
        path="/p2p-info",
        query={},
        headers={},
    )
    assert await proxy.token_middleware(public_request, handler) == "ok"

    bad_request = SimpleNamespace(
        method="GET",
        path="/api/config/preferences",
        query={},
        headers={"X-Proxy-Token": "wrong"},
    )
    with pytest.raises(lan_proxy.web.HTTPForbidden):
        await proxy.token_middleware(bad_request, handler)


class _FakeManager:
    def __init__(self) -> None:
        self.active_session_is_idle = True
        self.pending_agent_callbacks: list[object] = []
        self.websocket = None
        self.start_calls: list[tuple[object, bool, str, str | None]] = []

    def set_user_language(self, _language: str) -> None:
        return None

    async def start_session(
        self,
        websocket,
        new: bool = False,
        input_mode: str = "audio",
        audio_format: str | None = None,
    ) -> None:
        self.start_calls.append((websocket, new, input_mode, audio_format))

    async def send_status(self, _message: str) -> None:
        return None

    async def cleanup(self, expected_websocket=None) -> None:
        return None


class _FakeWebSocket:
    def __init__(self, messages: list[str]) -> None:
        self._messages = list(messages)
        self.client = ("127.0.0.1", 54321)
        self.accepted = False
        self.closed = False
        self.sent_texts: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        raise WebSocketDisconnect()

    async def send_text(self, payload: str) -> None:
        self.sent_texts.append(payload)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_websocket_start_session_forwards_audio_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _FakeManager()
    session_manager = {"lanlan": manager}
    session_ids: dict[str, object] = {}
    created_tasks: list[asyncio.Task] = []

    monkeypatch.setattr(websocket_router, "get_config_manager", lambda: object())
    monkeypatch.setattr(websocket_router, "get_session_manager", lambda: session_manager)
    monkeypatch.setattr(websocket_router, "get_session_id", lambda: session_ids)

    real_create_task = asyncio.create_task

    def tracking_create_task(coro):
        task = real_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(websocket_router.asyncio, "create_task", tracking_create_task)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "action": "start_session",
                    "input_type": "audio",
                    "audio_format": "PCM_24000HZ_MONO_16BIT",
                    "new_session": True,
                }
            )
        ]
    )

    await websocket_router.websocket_endpoint(websocket, "lanlan")
    if created_tasks:
        await asyncio.gather(*created_tasks)

    assert websocket.accepted is True
    assert manager.start_calls == [
        (websocket, True, "audio", "PCM_24000HZ_MONO_16BIT")
    ]
