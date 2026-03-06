# -*- coding: utf-8 -*-
"""
LAN Proxy - v2 架构 (HTTP/WebSocket 反向代理)
同WiFi直连P2P连接支持，使用 aiohttp 实现
"""

import asyncio
import json
import secrets
import socket
import sys
import os
from typing import Optional, Dict, Any

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

# 尝试导入二维码库
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# 配置
PROXY_PORT = LAN_PROXY_PORT
TARGET_BASE = f"http://127.0.0.1:{MAIN_SERVER_PORT}"
TARGET_WS_BASE = f"ws://127.0.0.1:{MAIN_SERVER_PORT}"

# 状态文件路径
STATUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lan_proxy_status.json")


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
    """LAN 代理服务器 - v2 HTTP/WebSocket 反向代理"""

    def __init__(self, bind_host: Optional[str] = None):
        self.token: str = secrets.token_urlsafe(32)
        self.lan_ip: str = bind_host or self._get_lan_ip()
        self.character: str = "test"
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

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

    # ── Token 鉴权中间件 ──
    @web.middleware
    async def token_middleware(self, request: web.Request, handler):
        """验证 URL query 或 Header 中的 token"""
        # 跳过 OPTIONS 请求（CORS 预检）
        if request.method == "OPTIONS":
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
        return web.json_response({
            "status": "ok",
            "lan_ip": self.lan_ip,
            "port": PROXY_PORT,
        })

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

        # 路由：所有其他 HTTP 请求
        app.router.add_route('*', '/{path:.*}', self.handle_http)

        # 启动服务器
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.lan_ip, PROXY_PORT)
        await self.site.start()

        print(f"[LAN Proxy] v2 started on {self.lan_ip}:{PROXY_PORT}")
        print(f"[LAN Proxy] Token: {self.token}")
        print(f"[LAN Proxy] Character: {self.character}")
        print(f"[LAN Proxy] Target: {TARGET_BASE}")

        # 保存状态
        _save_status(self.get_connection_info())

    async def stop(self):
        """停止代理服务器"""
        print("[LAN Proxy] Stopping...")

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        _clear_status()
        print("[LAN Proxy] Stopped")

    def get_connection_info(self) -> dict:
        """获取连接信息（用于生成二维码）"""
        return {
            "lan_ip": self.lan_ip,
            "port": PROXY_PORT,
            "token": self.token,
            "character": self.character,
        }

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


async def run_lan_proxy(stop_event=None, start_event=None):
    """
    运行 LAN 代理（供 launcher 调用）

    Args:
        stop_event: 停止事件，当设置时代理会优雅退出 (multiprocessing.Event)
        start_event: 启动完成事件，用于通知父进程已启动 (multiprocessing.Event)
    """
    global _proxy_instance

    if not AIOHTTP_AVAILABLE:
        print("[LAN Proxy] Error: aiohttp not installed. Please install it first.")
        return

    _proxy_instance = LanProxy()

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
