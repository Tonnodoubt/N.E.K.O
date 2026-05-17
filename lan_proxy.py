# -*- coding: utf-8 -*-
"""
LAN Proxy - v2 架构 (HTTP/WebSocket 反向代理)
同WiFi直连P2P连接支持，使用 aiohttp 实现
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import socket
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

# 添加项目根目录到路径
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LAN_PROXY_PORT, MAIN_SERVER_PORT

# 尝试导入 aiohttp
try:
    from aiohttp import web, ClientSession, WSMsgType
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("[LAN Proxy] Error: aiohttp not installed. Run `uv add aiohttp` or `pip install aiohttp`")
    # 创建一个虚拟的 web 模块，避免 NameError
    class DummyWeb:
        def middleware(self, func):
            return func
    web = DummyWeb()

# 尝试导入二维码库
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# 尝试导入云端注册客户端
try:
    from cloud_registry_client import CloudRegistryClient
    CLOUD_REGISTRY_AVAILABLE = True
except ImportError:
    CLOUD_REGISTRY_AVAILABLE = False

# 尝试导入 STUN 客户端
try:
    from stun_client import STUNClient
    STUN_AVAILABLE = True
except ImportError:
    STUN_AVAILABLE = False

# 尝试导入 UDP P2P 服务器
try:
    from udp_server import UDPP2PServer
    UDP_SERVER_AVAILABLE = True
except ImportError:
    UDP_SERVER_AVAILABLE = False

# 尝试加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 配置
PROXY_PORT = LAN_PROXY_PORT
TARGET_BASE = f"http://127.0.0.1:{MAIN_SERVER_PORT}"
TARGET_WS_BASE = f"ws://127.0.0.1:{MAIN_SERVER_PORT}"
MOBILE_API_VERSION = 1

# 状态文件路径
STATUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lan_proxy_status.json")
PAIRING_STORE_VERSION = 1
MAX_PERSISTED_PAIRINGS = 16


def _neko_state_dir() -> Path:
    return Path.home() / ".neko"


def _device_id_file_path() -> Path:
    return _neko_state_dir() / "device_id"


def _pairings_file_path() -> Path:
    return _neko_state_dir() / "mobile_pairings.json"


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _read_env(var_name: str) -> Optional[str]:
    """读取字符串环境变量，优先兼容 `NEKO_*` 前缀。"""
    for key in (f"NEKO_{var_name}", var_name):
        raw = os.getenv(key)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return value
    return None


def _read_bool_env(var_name: str, default: bool = False) -> bool:
    """读取布尔环境变量，默认关闭可选公网能力。"""
    raw = _read_env(var_name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "disable", "disabled"}:
        return False

    print(f"[LAN Proxy] Warning: Invalid {var_name}={raw!r}, using {default}")
    return default


def _read_optional_port(var_name: str) -> Optional[int]:
    """读取可选端口环境变量，非法值时仅打印警告。"""
    raw = _read_env(var_name)
    if raw is None:
        return None

    try:
        value = int(raw)
    except ValueError:
        print(f"[LAN Proxy] Warning: Invalid {var_name}={raw!r}, ignoring")
        return None

    if 1 <= value <= 65535:
        return value

    print(f"[LAN Proxy] Warning: Invalid {var_name}={raw!r}, ignoring")
    return None


def _get_cloud_registry_url() -> Optional[str]:
    value = _read_env("CLOUD_REGISTRY_URL")
    return value.rstrip("/") if value else None


def _extract_hostname(url: str) -> Optional[str]:
    parsed = urlparse(url if "://" in url else f"//{url}", scheme="http")
    return parsed.hostname


def _resolve_stun_endpoint() -> tuple[str, int, str]:
    """
    解析 STUN 服务地址。

    优先级：
    1. `NEKO_STUN_SERVER` / `STUN_SERVER`
    2. `NEKO_CLOUD_REGISTRY_URL` / `CLOUD_REGISTRY_URL` 的 hostname
    3. 公共 STUN 回退（仅便于本地调试）
    """
    explicit_server = _read_env("STUN_SERVER")
    explicit_port = _read_optional_port("STUN_PORT")
    if explicit_server:
        return explicit_server, explicit_port or 3478, "env"

    cloud_registry_url = _get_cloud_registry_url()
    if cloud_registry_url:
        hostname = _extract_hostname(cloud_registry_url)
        if hostname:
            return hostname, explicit_port or 3478, "cloud-registry"

    return "stun.l.google.com", explicit_port or 19302, "public-default"


def _save_status(info: dict):
    """保存状态到文件（供主进程读取）"""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(info, f)
    except Exception as e:
        print(f"[LAN Proxy] Warning: Failed to save status: {e}")


def _clear_status():
    """清除状态文件"""
    try:
        if os.path.exists(STATUS_FILE):
            os.remove(STATUS_FILE)
    except Exception:
        pass


def get_proxy_info_from_file() -> Optional[dict]:
    """从文件读取代理信息（供主进程调用）"""
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


class LanProxy:
    """LAN 代理服务器 - HTTP/WebSocket 反向代理，可选 STUN / 云端注册"""

    def __init__(
        self,
        bind_host: Optional[str] = None,
        enable_cloud: Optional[bool] = None,
        enable_stun: Optional[bool] = None,
        character: str = "test"
    ):
        """
        初始化 LAN 代理

        Args:
            bind_host: 绑定地址（默认自动检测）
            enable_cloud: 是否启用云端注册（None 时读取环境变量，默认关闭）
            enable_stun: 是否启用 STUN（None 时读取环境变量，默认关闭）
            character: 角色名称
        """
        self.token: str = secrets.token_urlsafe(32)
        self.bind_host: Optional[str] = bind_host
        self.lan_ip: str = bind_host or self._get_lan_ip()
        self.character: str = character
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.site_loopback: Optional[web.TCPSite] = None

        # 云端注册相关
        cloud_enabled = (
            _read_bool_env("ENABLE_CLOUD_REGISTRY", False)
            if enable_cloud is None
            else enable_cloud
        )
        self.enable_cloud = bool(cloud_enabled) and CLOUD_REGISTRY_AVAILABLE
        self.cloud_client: Optional[CloudRegistryClient] = None
        self._cloud_refresh_task: Optional[asyncio.Task] = None

        # STUN 相关
        stun_enabled = (
            _read_bool_env("ENABLE_STUN", False)
            if enable_stun is None
            else enable_stun
        )
        self.enable_stun = bool(stun_enabled) and STUN_AVAILABLE
        self.stun_client: Optional[STUNClient] = None
        self.stun_ip: Optional[str] = None
        self.stun_port: Optional[int] = None

        # UDP P2P 服务器相关
        self.udp_server: Optional[UDPP2PServer] = None

    def _get_lan_ip(self) -> str:
        """获取当前WiFi网卡的局域网IP"""
        # 方法1：通过UDP连接外网获取本机出口IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(2)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if self._is_private_ip(ip):
                    return ip
        except Exception:
            pass

        # 方法2：通过 hostname 解析获取私有IP（macOS 上 getaddrinfo 可能失败）
        try:
            hostname = socket.gethostname()
            addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
            for addr in addrs:
                ip = addr[4][0]
                if self._is_private_ip(ip) and ip != "127.0.0.1":
                    return ip
        except Exception:
            pass

        # 方法3：使用系统命令获取 IP（macOS/Linux/Windows 兼容）
        try:
            import subprocess
            import platform
            import re

            system = platform.system()
            if system == "Darwin":
                result = subprocess.run(
                    ["route", "-n", "get", "default"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    interface = None
                    for line in result.stdout.splitlines():
                        if "interface:" in line:
                            interface = line.split(":")[1].strip()
                            break
                    if interface:
                        result = subprocess.run(
                            ["ifconfig", interface],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
                            if match:
                                ip = match.group(1)
                                if self._is_private_ip(ip) and ip != "127.0.0.1":
                                    return ip
            elif system == "Linux":
                result = subprocess.run(
                    ["hostname", "-I"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for ip in result.stdout.strip().split():
                        if self._is_private_ip(ip) and ip != "127.0.0.1":
                            return ip
            elif system == "Windows":
                result = subprocess.run(
                    ["ipconfig"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for ip in re.findall(r"IPv4.*?:\s*(\d+\.\d+\.\d+\.\d+)", result.stdout):
                        if self._is_private_ip(ip) and ip != "127.0.0.1":
                            return ip
        except Exception:
            pass

        # 方法4：使用 netifaces 库（如果已安装）
        try:
            import netifaces
            for interface in netifaces.interfaces():
                if interface.startswith(("lo", "docker", "br-", "veth")):
                    continue
                addrs = netifaces.ifaddresses(interface).get(netifaces.AF_INET, [])
                for addr in addrs:
                    ip = addr.get("addr")
                    if ip and self._is_private_ip(ip) and ip != "127.0.0.1":
                        return ip
        except Exception:
            pass

        # 兜底：本地测试用
        print("[LAN Proxy] Warning: Using 127.0.0.1 (local testing only)")
        return "127.0.0.1"

    def _refresh_lan_ip(self) -> None:
        """Refresh advertised LAN IP when using automatic interface detection."""
        if self.bind_host:
            return
        current_ip = self._get_lan_ip()
        if current_ip and current_ip != self.lan_ip:
            print(f"[LAN Proxy] LAN IP changed: {self.lan_ip} -> {current_ip}")
            self.lan_ip = current_ip

    @staticmethod
    def _is_private_ip(ip: str) -> bool:
        """检查是否为私有IP地址"""
        try:
            parts = ip.split(".")
            if len(parts) != 4:
                return False
            first, second = int(parts[0]), int(parts[1])
            if first == 10:  # 10.0.0.0/8
                return True
            if first == 172 and 16 <= second <= 31:  # 172.16.0.0/12
                return True
            if first == 192 and second == 168:  # 192.168.0.0/16
                return True
            return False
        except (ValueError, IndexError):
            return False

    async def _get_current_character(self) -> str:
        """从主服务获取当前角色名"""
        try:
            async with ClientSession() as session:
                async with session.get(
                    f"{TARGET_BASE}/api/characters/current",
                    timeout=2
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        char = data.get('name') or data.get('character')
                        if char:
                            return char
        except Exception:
            pass

        # 兜底：尝试从配置文件读取
        try:
            from config import get_config_manager
            config_manager = get_config_manager()
            _, her_name, _, _, _, _, _, _, _, _ = config_manager.get_character_data()
            if isinstance(her_name, str) and her_name.strip():
                return her_name.strip()
        except Exception:
            pass

        return "test"

    def _is_loopback_request(self, request: web.Request) -> bool:
        """Return whether the TCP peer is local to this machine."""
        import ipaddress

        peer = request.transport.get_extra_info("peername") if request.transport else None
        host = peer[0] if isinstance(peer, tuple) and peer else request.remote
        if not host:
            return False

        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host == "localhost"

    def _load_mobile_pairings(self) -> dict:
        device_id = self._get_device_id()
        store_path = _pairings_file_path()

        if not store_path.exists():
            return {
                "version": PAIRING_STORE_VERSION,
                "device_id": device_id,
                "pairings": {},
            }

        try:
            payload = json.loads(store_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[LAN Proxy] Warning: failed to read mobile pairings: {e}")
            return {
                "version": PAIRING_STORE_VERSION,
                "device_id": device_id,
                "pairings": {},
            }

        pairings = payload.get("pairings")
        if (
            payload.get("version") != PAIRING_STORE_VERSION
            or payload.get("device_id") != device_id
            or not isinstance(pairings, dict)
        ):
            return {
                "version": PAIRING_STORE_VERSION,
                "device_id": device_id,
                "pairings": {},
            }

        return payload

    def _save_mobile_pairings(self, payload: dict):
        pairings = payload.get("pairings")
        if not isinstance(pairings, dict):
            pairings = {}

        if len(pairings) > MAX_PERSISTED_PAIRINGS:
            ordered_keys = sorted(
                pairings.keys(),
                key=lambda key: (
                    int((pairings.get(key) or {}).get("last_resolved_at") or 0),
                    int((pairings.get(key) or {}).get("created_at") or 0),
                ),
                reverse=True,
            )
            pairings = {key: pairings[key] for key in ordered_keys[:MAX_PERSISTED_PAIRINGS]}

        _write_json_atomic(
            _pairings_file_path(),
            {
                "version": PAIRING_STORE_VERSION,
                "device_id": self._get_device_id(),
                "pairings": pairings,
            },
        )

    def create_mobile_pairing(
        self,
        client_name: str = "",
        client_device_id: str = "",
        user_agent: str = "",
    ) -> dict:
        payload = self._load_mobile_pairings()
        pairings = payload["pairings"]
        pairing_id = f"pair_{secrets.token_urlsafe(9)}"
        pairing_secret = secrets.token_urlsafe(32)
        now = int(time.time())

        pairings[pairing_id] = {
            "secret_sha256": hashlib.sha256(pairing_secret.encode("utf-8")).hexdigest(),
            "client_name": str(client_name or "")[:128],
            "client_device_id": str(client_device_id or "")[:128],
            "user_agent": str(user_agent or "")[:256],
            "created_at": now,
            "last_resolved_at": 0,
        }
        self._save_mobile_pairings(payload)

        return {
            "pairing_id": pairing_id,
            "pairing_secret": pairing_secret,
            "device_id": self._get_device_id(),
            "register_path": "/pairing/register",
            "resolve_path": "/pairing/resolve",
            "created_at": now,
        }

    def resolve_mobile_pairing(
        self,
        pairing_id: str,
        pairing_secret: str,
        client_name: str = "",
        client_device_id: str = "",
        user_agent: str = "",
    ) -> Optional[dict]:
        if not pairing_id or not pairing_secret:
            return None

        payload = self._load_mobile_pairings()
        pairings = payload["pairings"]
        entry = pairings.get(pairing_id)
        if not isinstance(entry, dict):
            return None

        expected_hash = str(entry.get("secret_sha256") or "")
        actual_hash = hashlib.sha256(str(pairing_secret).encode("utf-8")).hexdigest()
        if not expected_hash or not hmac.compare_digest(expected_hash, actual_hash):
            return None

        entry["last_resolved_at"] = int(time.time())
        if client_name:
            entry["client_name"] = str(client_name)[:128]
        if client_device_id:
            entry["client_device_id"] = str(client_device_id)[:128]
        if user_agent:
            entry["user_agent"] = str(user_agent)[:256]
        self._save_mobile_pairings(payload)

        return self.get_connection_info()

    async def handle_pairing_register(self, request: web.Request) -> web.Response:
        """Register a durable mobile pairing after the first QR/token bootstrap."""
        payload = {}
        if request.can_read_body:
            try:
                payload = await request.json()
            except Exception:
                payload = {}

        pairing = self.create_mobile_pairing(
            client_name=str(payload.get("client_name") or ""),
            client_device_id=str(payload.get("client_device_id") or ""),
            user_agent=request.headers.get("User-Agent", ""),
        )

        return web.json_response({
            "success": True,
            **self.get_connection_info(),
            "pairing": pairing,
        })

    async def handle_pairing_resolve(self, request: web.Request) -> web.Response:
        """Resolve a fresh runtime token from a previously stored durable pairing."""
        payload = {}
        if request.method != "GET" and request.can_read_body:
            try:
                payload = await request.json()
            except Exception:
                payload = {}

        pairing_id = str(payload.get("pairing_id") or request.query.get("pairing_id") or "").strip()
        pairing_secret = str(
            payload.get("pairing_secret")
            or request.query.get("pairing_secret")
            or ""
        ).strip()
        client_name = str(payload.get("client_name") or request.query.get("client_name") or "")
        client_device_id = str(
            payload.get("client_device_id")
            or request.query.get("client_device_id")
            or ""
        )

        if not pairing_id or not pairing_secret:
            return web.json_response(
                {"success": False, "error": "Missing pairing credentials"},
                status=400,
            )

        info = self.resolve_mobile_pairing(
            pairing_id=pairing_id,
            pairing_secret=pairing_secret,
            client_name=client_name,
            client_device_id=client_device_id,
            user_agent=request.headers.get("User-Agent", ""),
        )
        if not info:
            return web.json_response(
                {"success": False, "error": "Invalid pairing credentials"},
                status=403,
            )

        return web.json_response({
            "success": True,
            **info,
            "pairing": {
                "pairing_id": pairing_id,
                "device_id": info.get("device_id", ""),
                "resolve_path": "/pairing/resolve",
            },
        })

    # ── Token 鉴权中间件 ──
    @web.middleware
    async def token_middleware(self, request: web.Request, handler):
        """验证 URL query 或 Header 中的 token"""
        # 跳过 OPTIONS 请求（CORS 预检）
        if request.method == "OPTIONS":
            return await handler(request)

        # 健康检查公开；连接信息/二维码只允许本机主服务代取，避免同 LAN
        # 其他设备在配对前直接拿到 proxy token。
        if request.path == '/health':
            return await handler(request)

        # 扫过一次码后，移动端后续只靠 durable pairing secret 换新 token。
        if request.path == '/pairing/resolve':
            return await handler(request)

        local_pairing_paths = {'/p2p-info', '/lanproxyqrcode'}
        if request.path in local_pairing_paths and self._is_loopback_request(request):
            return await handler(request)

        # 获取 token（优先 query，其次 header）
        token = request.query.get('token')
        if not token:
            token = request.headers.get('X-Proxy-Token')

        if token != self.token:
            raise web.HTTPForbidden(text='Invalid token')

        return await handler(request)

    # ── CORS 中间件 ──
    @web.middleware
    async def cors_middleware(self, request: web.Request, handler):
        """处理 CORS 跨域"""
        if request.method == "OPTIONS":
            # 预检请求响应
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS, PATCH',
                'Access-Control-Allow-Headers': 'Content-Type, X-Proxy-Token, Authorization',
                'Access-Control-Max-Age': '86400',
            }
            return web.Response(status=204, headers=headers)

        response = await handler(request)

        # 添加 CORS 头
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Proxy-Token, Authorization'
        return response

    # ── WebSocket 反向代理 ──
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """处理 WebSocket 连接，转发到主服务"""
        ws_client = web.WebSocketResponse(
            heartbeat=30.0,  # 30秒心跳
            autoping=True,
        )
        await ws_client.prepare(request)

        # 构建目标 URL（移除 token 参数，保留其他参数）
        target_params = {k: v for k, v in request.query.items() if k != 'token'}
        target_url = f"{TARGET_WS_BASE}{request.path}"
        if target_params:
            query_string = '&'.join(f'{k}={v}' for k, v in target_params.items())
            target_url = f"{target_url}?{query_string}"

        client_addr = request.remote
        print(f"[LAN Proxy] WebSocket connection from {client_addr} -> {request.path}")

        try:
            async with ClientSession() as session:
                async with session.ws_connect(
                    target_url,
                    heartbeat=30.0,
                    autoping=True,
                ) as ws_server:
                    # 双向转发
                    await asyncio.gather(
                        self._pipe_ws_client_to_server(ws_client, ws_server),
                        self._pipe_ws_server_to_client(ws_server, ws_client),
                        return_exceptions=True,
                    )
        except Exception as e:
            print(f"[LAN Proxy] WebSocket error: {e}")
        finally:
            print(f"[LAN Proxy] WebSocket disconnected from {client_addr}")

        return ws_client

    async def _pipe_ws_client_to_server(self, src: web.WebSocketResponse, dst):
        """客户端 -> 服务端"""
        async for msg in src:
            if msg.type == WSMsgType.TEXT:
                await dst.send_str(msg.data)
            elif msg.type == WSMsgType.BINARY:
                await dst.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR, WSMsgType.CLOSED):
                break

    async def _pipe_ws_server_to_client(self, src, dst: web.WebSocketResponse):
        """服务端 -> 客户端"""
        async for msg in src:
            if msg.type == WSMsgType.TEXT:
                await dst.send_str(msg.data)
            elif msg.type == WSMsgType.BINARY:
                await dst.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR, WSMsgType.CLOSED):
                break

    # ── HTTP 反向代理 ──
    async def handle_http(self, request: web.Request) -> web.Response:
        """处理 HTTP 请求，转发到主服务"""
        # 构建目标 URL（移除 token 参数）
        target_params = {k: v for k, v in request.query.items() if k != 'token'}
        target_url = f"{TARGET_BASE}{request.path}"

        # 复制 headers（移除敏感头）
        headers = {}
        for k, v in request.headers.items():
            k_lower = k.lower()
            if k_lower not in ('host', 'x-proxy-token', 'connection', 'transfer-encoding'):
                headers[k] = v

        # 转发请求
        async with ClientSession() as session:
            try:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=target_params if target_params else None,
                    data=await request.read(),
                    timeout=30,
                ) as resp:
                    # 复制响应 headers
                    response_headers = {}
                    for k, v in resp.headers.items():
                        k_lower = k.lower()
                        if k_lower not in ('transfer-encoding', 'content-encoding', 'connection'):
                            response_headers[k] = v

                    # 添加 CORS 头
                    response_headers['Access-Control-Allow-Origin'] = '*'

                    body = await resp.read()
                    return web.Response(
                        body=body,
                        status=resp.status,
                        headers=response_headers,
                    )
            except Exception as e:
                print(f"[LAN Proxy] HTTP proxy error: {e}")
                return web.Response(
                    status=502,
                    text=f"Proxy error: {e}",
                    headers={'Access-Control-Allow-Origin': '*'},
                )

    # ── 健康检查 ──
    async def handle_health(self, request: web.Request) -> web.Response:
        """健康检查端点"""
        self._refresh_lan_ip()
        return web.json_response({
            "status": "ok",
            "service": "lan_proxy",
            "mobile_backend": True,
            "mobile_api_version": MOBILE_API_VERSION,
            "lan_ip": self.lan_ip,
            "port": PROXY_PORT,
            "pairing_supported": True,
        })

    # ── P2P 连接信息 ──
    async def handle_p2p_info(self, request: web.Request) -> web.Response:
        """返回 P2P 连接信息（用于生成二维码）"""
        info = self.get_connection_info()
        return web.json_response(info)

    # ── P2P 二维码图片 ──
    async def handle_lanproxy_qrcode(self, request: web.Request) -> web.Response:
        """生成并返回 P2P 连接二维码图片"""
        if not QR_AVAILABLE:
            return web.json_response(
                {"error": "QR code library not available"},
                status=503
            )

        try:
            # 获取连接信息
            info = self.get_connection_info()

            # 生成二维码
            import io
            qr_data = json.dumps(info)
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=2,
            )
            qr.add_data(qr_data)
            qr.make(fit=True)

            # 创建图片
            img = qr.make_image(fill_color="black", back_color="white")

            # 转换为字节流
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # 返回图片响应
            return web.Response(
                body=img_byte_arr.getvalue(),
                content_type='image/png',
                headers={
                    'X-Lan-Ip': info.get('lan_ip', ''),
                    'X-Port': str(info.get('port', '')),
                    'X-Token': info.get('token', ''),
                    'X-Device-Id': info.get('device_id', ''),
                    'X-Pairing-Supported': '1' if info.get('pairing_supported') else '0',
                    'X-Pairing-Register-Path': info.get('pairing_register_path', ''),
                    'X-Pairing-Resolve-Path': info.get('pairing_resolve_path', ''),
                }
            )
        except Exception as e:
            print(f"[LAN Proxy] Error generating QR code: {e}")
            return web.json_response(
                {"error": f"Failed to generate QR code: {str(e)}"},
                status=500
            )

    # ── 启动 / 停止 ──
    async def start(self):
        """启动代理服务器"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for LAN proxy v2")

        # 获取当前角色名
        self.character = await self._get_current_character()

        # 创建应用
        app = web.Application(middlewares=[
            self.cors_middleware,
            self.token_middleware,
        ])

        # 路由：WebSocket 路径
        app.router.add_route('GET', '/ws/{name}', self.handle_websocket)

        # 路由：健康检查（不需要 token）
        app.router.add_route('GET', '/health', self.handle_health)

        # 路由：P2P 连接信息（仅本机免 token，供 Main Server / 桌面 UI 取）
        app.router.add_route('GET', '/p2p-info', self.handle_p2p_info)

        # 路由：P2P 连接二维码图片（仅本机免 token）
        app.router.add_route('GET', '/lanproxyqrcode', self.handle_lanproxy_qrcode)

        # 路由：移动端持久配对（首次仍经 token/二维码，后续 resolve 不再依赖二维码）
        app.router.add_route('POST', '/pairing/register', self.handle_pairing_register)
        app.router.add_route('*', '/pairing/resolve', self.handle_pairing_resolve)

        # 路由：所有其他 HTTP 请求
        app.router.add_route('*', '/{path:.*}', self.handle_http)

        # 启动服务器
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        # 默认绑定所有 IPv4 接口，避免 Wi-Fi / LAN IP 变化后仍卡在旧地址。
        bind_addr = self.bind_host or '0.0.0.0'
        self.site = web.TCPSite(self.runner, bind_addr, PROXY_PORT)
        await self.site.start()
        # 显式绑定某个非 loopback 地址时，额外保留本机调试 / 主服务转接入口。
        if bind_addr not in {'0.0.0.0', '127.0.0.1'}:
            self.site_loopback = web.TCPSite(self.runner, '127.0.0.1', PROXY_PORT)
            await self.site_loopback.start()

        print(f"[LAN Proxy] v2 started on {bind_addr}:{PROXY_PORT} (advertising {self.lan_ip}:{PROXY_PORT})")
        print(f"[LAN Proxy] Token: {self.token}")
        print(f"[LAN Proxy] Character: {self.character}")
        print(f"[LAN Proxy] Target: {TARGET_BASE}")

        # ── STUN 公网 endpoint 发现 ──
        if self.enable_stun:
            print("[LAN Proxy] 正在通过 STUN 获取公网 endpoint...")
            try:
                stun_server, stun_port, stun_source = _resolve_stun_endpoint()
                print(f"[LAN Proxy] STUN server: {stun_server}:{stun_port} ({stun_source})")
                self.stun_client = STUNClient(stun_server, stun_port)
                endpoint = await self.stun_client.get_public_endpoint()
                if endpoint:
                    self.stun_ip, self.stun_port = endpoint
                    print(f"[LAN Proxy] ✅ STUN endpoint: {self.stun_ip}:{self.stun_port}")
                else:
                    print("[LAN Proxy] ⚠️ STUN 获取失败")
            except Exception as e:
                print(f"[LAN Proxy] ⚠️ STUN 配置失败: {e}")
        else:
            print("[LAN Proxy] STUN disabled; running in local LAN mode")

        # ── UDP P2P 服务器 ──
        if UDP_SERVER_AVAILABLE:
            print("[LAN Proxy] 正在启动 UDP P2P 服务器...")
            try:
                # UDP 使用独立的端口（与 TCP HTTP proxy 分开）
                # 如果 STUN 成功，使用 STUN 端口；否则使用 PROXY_PORT + 1
                udp_port = self.stun_port if self.stun_port else (PROXY_PORT + 1)

                # 确定 TCP IP：
                # 1. 如果有 STUN IP，使用 STUN IP（因为 UDP 客户端会从外网连接）
                # 2. 否则使用 LAN IP
                tcp_ip = self.stun_ip if self.stun_ip else self.lan_ip

                # 传递 TCP 端口（HTTP 代理端口），供客户端后续连接
                # 传入 device_id 和云端 URL，启用双向打洞
                cloud_url = _get_cloud_registry_url() if self.enable_cloud else None
                device_id = self._get_device_id() if self.enable_cloud else None
                self.udp_server = UDPP2PServer(
                    port=udp_port,
                    token=self.token,
                    tcp_port=PROXY_PORT,
                    tcp_ip=tcp_ip,
                    device_id=device_id,
                    cloud_registry_url=cloud_url,
                )
                await self.udp_server.start()
                print(f"[LAN Proxy] ✅ UDP P2P 服务器已启动，端口: {udp_port}")
                print(f"[LAN Proxy] UDP 客户端将连接到 TCP 端口: {PROXY_PORT}")
            except Exception as e:
                print(f"[LAN Proxy] ⚠️ UDP 服务器启动失败: {e}")
        else:
            print("[LAN Proxy] ⚠️ UDP 服务器不可用，跳过启动")

        # ── 云端注册 ──
        if self.enable_cloud:
            print("[LAN Proxy] 正在注册到云端...")
            try:
                self.cloud_client = CloudRegistryClient()
                device_id = self._get_device_id()  # 同步调用

                success = await self.cloud_client.register(
                    device_id=device_id,
                    lan_ip=self.lan_ip,
                    token=self.token,
                    stun_ip=self.stun_ip,
                    stun_port=self.stun_port,
                    character=self.character
                )

                if success:
                    print(f"[LAN Proxy] ✅ 云端注册成功")
                    print(f"[LAN Proxy] Device ID: {device_id}")
                    # 启动定期刷新
                    self._start_cloud_refresh()
                else:
                    print("[LAN Proxy] ⚠️ 云端注册失败")
            except Exception as e:
                print(f"[LAN Proxy] ⚠️ 云端注册失败: {e}")
        else:
            print("[LAN Proxy] Cloud registry disabled; QR/LAN connection remains available")

        # 保存状态
        _save_status(self.get_connection_info())

    async def stop(self):
        """停止代理服务器"""
        print("[LAN Proxy] Stopping...")

        # 停止云端刷新任务
        if self._cloud_refresh_task:
            self._cloud_refresh_task.cancel()
            try:
                await self._cloud_refresh_task
            except asyncio.CancelledError:
                pass

        # 关闭云端客户端
        if self.cloud_client:
            try:
                await self.cloud_client.close()
                print("[LAN Proxy] 云端客户端已关闭")
            except Exception as e:
                print(f"[LAN Proxy] 云端客户端关闭失败: {e}")

        # 停止 UDP P2P 服务器
        if self.udp_server:
            try:
                await self.udp_server.stop()
                print("[LAN Proxy] UDP P2P 服务器已停止")
            except Exception as e:
                print(f"[LAN Proxy] UDP 服务器停止失败: {e}")

        if self.site:
            await self.site.stop()
        if hasattr(self, 'site_loopback') and self.site_loopback:
            await self.site_loopback.stop()
        if self.runner:
            await self.runner.cleanup()

        _clear_status()
        print("[LAN Proxy] Stopped")

    def _get_device_id(self) -> str:
        """
        获取设备 ID（持久化）

        Returns:
            设备 ID
        """
        device_file = _device_id_file_path()

        # 尝试读取已存在的 ID
        if device_file.exists():
            device_id = device_file.read_text().strip()
            print(f"[LAN Proxy] 使用已存在的 Device ID: {device_id}")
            return device_id

        # 生成新的 ID
        import uuid
        device_id = f"neko-{uuid.uuid4().hex[:16]}"

        # 保存
        device_file.parent.mkdir(parents=True, exist_ok=True)
        device_file.write_text(device_id)
        print(f"[LAN Proxy] 生成新的 Device ID: {device_id}")

        return device_id

    def _start_cloud_refresh(self):
        """启动云端注册刷新后台任务"""
        async def refresh_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # 每60秒刷新一次

                    # 刷新 STUN endpoint
                    if self.enable_stun and self.stun_client:
                        try:
                            endpoint = await self.stun_client.get_public_endpoint()
                            if endpoint:
                                self.stun_ip, self.stun_port = endpoint
                        except Exception as e:
                            print(f"[LAN Proxy] STUN 刷新失败: {e}")

                    device_id = self._get_device_id()  # 同步调用
                    await self.cloud_client.register(
                        device_id=device_id,
                        lan_ip=self.lan_ip,
                        token=self.token,
                        stun_ip=self.stun_ip,
                        stun_port=self.stun_port,
                        character=self.character
                    )
                    print("[LAN Proxy] 云端注册已刷新")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[LAN Proxy] 云端刷新失败: {e}")

        self._cloud_refresh_task = asyncio.create_task(refresh_loop())

    def get_connection_info(self) -> dict:
        """获取连接信息（用于生成二维码）"""
        self._refresh_lan_ip()

        # 计算 UDP 端口（与启动时使用的相同逻辑）
        udp_port = self.stun_port if self.stun_port else (PROXY_PORT + 1)

        info = {
            "schema": "neko.mobile.p2p.v1",
            "service": "lan_proxy",
            "mobile_backend": True,
            "mobile_api_version": MOBILE_API_VERSION,
            "lan_ip": self.lan_ip,
            "port": PROXY_PORT,  # TCP HTTP 端口（供 WebSocket/HTTP 连接）
            "udp_port": udp_port,  # UDP P2P 端口（用于 UDP 打洞）
            "token": self.token,
            "character": self.character,
            "device_id": self._get_device_id(),
            "pairing_supported": True,
            "pairing_register_path": "/pairing/register",
            "pairing_resolve_path": "/pairing/resolve",
        }

        # 添加 STUN 信息
        if self.stun_ip and self.stun_port:
            info["stun_ip"] = self.stun_ip
            info["stun_port"] = self.stun_port

        return info

    def get_qr_data(self) -> str:
        """生成二维码数据"""
        info = self.get_connection_info()
        return json.dumps(info, separators=(',', ':'))


# 全局实例（用于 launcher 管理）
_proxy_instance: Optional[LanProxy] = None


def get_proxy() -> Optional[LanProxy]:
    """获取当前代理实例"""
    return _proxy_instance


def get_proxy_info() -> Optional[dict]:
    """获取代理连接信息"""
    if _proxy_instance:
        return _proxy_instance.get_connection_info()
    return None


def get_proxy_qr_data() -> Optional[str]:
    """获取代理二维码数据"""
    if _proxy_instance:
        return _proxy_instance.get_qr_data()
    return None


async def run_lan_proxy(
    stop_event=None,
    start_event=None,
    enable_cloud: Optional[bool] = None,
    enable_stun: Optional[bool] = None,
    character: str = "test"
):
    """
    运行 LAN 代理（供 launcher 调用）

    Args:
        stop_event: 停止事件，当设置时代理会优雅退出 (multiprocessing.Event)
        start_event: 启动完成事件，用于通知父进程已启动 (multiprocessing.Event)
        enable_cloud: 是否启用云端注册（None 时读取环境变量，默认关闭）
        enable_stun: 是否启用 STUN（None 时读取环境变量，默认关闭）
        character: 角色名称（默认 "test"）
    """
    global _proxy_instance

    if not AIOHTTP_AVAILABLE:
        print("[LAN Proxy] Error: aiohttp not installed. Please install it first.")
        return

    _proxy_instance = LanProxy(
        enable_cloud=enable_cloud,
        enable_stun=enable_stun,
        character=character
    )

    try:
        await _proxy_instance.start()

        # 通知父进程已启动
        if start_event:
            start_event.set()

        # 保持运行，直到 stop_event 被设置
        if stop_event:
            while not stop_event.is_set():
                await asyncio.sleep(0.5)
        else:
            # 如果没有 stop_event，无限运行
            while True:
                await asyncio.sleep(3600)

    except KeyboardInterrupt:
        print("\n[LAN Proxy] Received KeyboardInterrupt")
    except Exception as e:
        print(f"[LAN Proxy] Error: {e}")
    finally:
        if _proxy_instance:
            await _proxy_instance.stop()
        _proxy_instance = None


if __name__ == "__main__":
    asyncio.run(run_lan_proxy())
