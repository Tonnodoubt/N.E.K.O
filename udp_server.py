"""
UDP P2P 服务器
用于移动端 UDP 打洞连接
"""

import asyncio
import logging
import socket
import json
import aiohttp
from typing import Optional, Tuple, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class UDPP2PServer:
    """UDP P2P 服务器"""

    def __init__(self, port: int, token: str, tcp_port: Optional[int] = None, tcp_ip: Optional[str] = None,
                 device_id: Optional[str] = None, cloud_registry_url: Optional[str] = None):
        """
        初始化 UDP 服务器

        Args:
            port: 监听端口（应该与 STUN 获取的端口一致）
            token: 连接 token（用于验证）
            tcp_port: TCP 服务端口（HTTP 代理端口，用于返回给客户端）
            tcp_ip: TCP 服务 IP（可选，默认自动检测）
            device_id: 设备 ID（用于云端打洞协调）
            cloud_registry_url: 云注册中心地址（用于轮询手机公网地址）
        """
        self.port = port
        self.token = token
        self.tcp_port = tcp_port or port
        self.tcp_ip = tcp_ip
        self.device_id = device_id
        self.cloud_registry_url = cloud_registry_url
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.clients = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._punch_task: Optional[asyncio.Task] = None

        logger.info(f"[UDP P2P] 初始化服务器，端口: {port}, TCP 端口: {self.tcp_port}")

    async def start(self):
        """启动 UDP 服务器"""
        try:
            # 创建 UDP socket（使用阻塞模式，配合 asyncio executor）
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 绑定到指定端口
            self.socket.bind(('0.0.0.0', self.port))
            # 不设置 setblocking(False)，保持默认的阻塞模式

            self.running = True
            logger.info(f"[UDP P2P] 服务器启动，监听端口: {self.port}")

            # 启动接收任务
            self._receive_task = asyncio.create_task(self._receive_loop())

            # 启动心跳检查任务
            asyncio.create_task(self._heartbeat_check_loop())

            # 如果配置了云端，启动打洞轮询任务
            if self.device_id and self.cloud_registry_url:
                self._punch_task = asyncio.create_task(self._punch_poll_loop())

        except Exception as e:
            logger.error(f"[UDP P2P] 启动失败: {e}")
            raise

    async def stop(self):
        """停止 UDP 服务器"""
        logger.info("[UDP P2P] 正在停止服务器...")
        self.running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._punch_task:
            self._punch_task.cancel()
            try:
                await self._punch_task
            except asyncio.CancelledError:
                pass

        if self.socket:
            self.socket.close()
            self.socket = None

        logger.info("[UDP P2P] 服务器已停止")

    async def _receive_loop(self):
        """接收消息循环"""
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                # 使用阻塞模式接收数据（在 executor 中运行）
                data, addr = await loop.run_in_executor(
                    None,
                    self.socket.recvfrom,
                    4096
                )

                # 处理消息
                await self._handle_message(data, addr)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.error(f"[UDP P2P] 接收消息错误: {e}")
                await asyncio.sleep(0.1)

    async def _handle_message(self, data: bytes, addr: Tuple[str, int]):
        """
        处理接收到的消息

        Args:
            data: 消息数据
            addr: 客户端地址 (ip, port)
        """
        try:
            # 解析 JSON 消息
            message = json.loads(data.decode('utf-8'))
            msg_type = message.get('type')
            msg_token = message.get('token')

            logger.info(f"[UDP P2P] 收到消息 {addr}: {msg_type}")

            # 验证 token
            if msg_token != self.token:
                logger.warning(f"[UDP P2P] Token 验证失败，来自 {addr}")
                await self._send_error(addr, "Invalid token")
                return

            # 处理不同类型的消息
            if msg_type == 'HELLO':
                # 握手消息
                await self._handle_hello(addr, message)

            elif msg_type == 'HEARTBEAT':
                # 心跳消息
                await self._handle_heartbeat(addr, message)

            elif msg_type == 'DATA':
                # 数据消息
                await self._handle_data(addr, message)

            else:
                logger.warning(f"[UDP P2P] 未知消息类型: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"[UDP P2P] 消息格式错误，来自 {addr}")
        except Exception as e:
            logger.error(f"[UDP P2P] 处理消息失败: {e}")

    async def _handle_hello(self, addr: Tuple[str, int], message: dict):
        """
        处理握手消息

        Args:
            addr: 客户端地址
            message: 消息内容
        """
        logger.info(f"[UDP P2P] 客户端握手成功: {addr}")

        # 记录客户端
        self.clients[addr] = datetime.now()

        # 确定 TCP endpoint（供客户端后续连接）
        # 如果有指定的 TCP IP，使用它；否则获取本机对外 IP
        if self.tcp_ip:
            tcp_ip = self.tcp_ip
        else:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 80))
                tcp_ip = s.getsockname()[0]
                s.close()
            except Exception:
                tcp_ip = '127.0.0.1'
        tcp_port = self.tcp_port

        # 发送 ACK（包含 TCP endpoint 信息）
        response = {
            'type': 'ACK',
            'token': self.token,
            'message': 'Connection established',
            'tcp_endpoint': {
                'ip': tcp_ip,
                'port': tcp_port
            },
            'timestamp': int(datetime.now().timestamp())
        }
        await self._send_message(addr, response)
        logger.info(f"[UDP P2P] 已返回 TCP endpoint: {tcp_ip}:{tcp_port}")

    async def _handle_heartbeat(self, addr: Tuple[str, int], message: dict):
        """
        处理心跳消息

        Args:
            addr: 客户端地址
            message: 消息内容
        """
        # 更新客户端心跳时间
        if addr in self.clients:
            self.clients[addr] = datetime.now()

        # 发送心跳响应
        response = {
            'type': 'HEARTBEAT_ACK',
            'token': self.token,
            'timestamp': int(datetime.now().timestamp())
        }
        await self._send_message(addr, response)

    async def _handle_data(self, addr: Tuple[str, int], message: dict):
        """
        处理数据消息

        Args:
            addr: 客户端地址
            message: 消息内容
        """
        payload = message.get('payload')
        logger.info(f"[UDP P2P] 收到数据: {payload}")

        # 这里可以添加自定义的数据处理逻辑
        # 例如转发到主服务器等

        # 发送确认
        response = {
            'type': 'DATA_ACK',
            'token': self.token,
            'timestamp': int(datetime.now().timestamp())
        }
        await self._send_message(addr, response)

    async def _send_message(self, addr: Tuple[str, int], message: dict):
        """
        发送消息

        Args:
            addr: 目标地址
            message: 消息内容
        """
        try:
            data = json.dumps(message).encode('utf-8')
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.socket.sendto,
                data,
                addr
            )
            logger.debug(f"[UDP P2P] 发送消息到 {addr}: {message.get('type')}")
        except Exception as e:
            logger.error(f"[UDP P2P] 发送消息失败: {e}")

    async def _send_error(self, addr: Tuple[str, int], error_msg: str):
        """
        发送错误消息

        Args:
            addr: 目标地址
            error_msg: 错误信息
        """
        response = {
            'type': 'ERROR',
            'token': self.token,
            'message': error_msg,
            'timestamp': int(datetime.now().timestamp())
        }
        await self._send_message(addr, response)

    async def _heartbeat_check_loop(self):
        """心跳检查循环（清理超时客户端）"""
        while self.running:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次

                # 清理超过 90 秒没有心跳的客户端
                now = datetime.now()
                timeout_clients = [
                    addr for addr, last_time in self.clients.items()
                    if (now - last_time).seconds > 90
                ]

                for addr in timeout_clients:
                    logger.info(f"[UDP P2P] 客户端超时断开: {addr}")
                    del self.clients[addr]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[UDP P2P] 心跳检查错误: {e}")

    def get_connected_clients(self) -> list:
        """获取已连接的客户端列表"""
        return list(self.clients.keys())

    async def _punch_poll_loop(self):
        """
        轮询云端，等待手机上报公网地址，然后主动发 PUNCH 包打洞
        每2秒轮询一次，最多等待 30 秒
        """
        logger.info("[UDP P2P] 开始轮询云端等待手机公网地址...")
        url = f"{self.cloud_registry_url}/api/punch"
        deadline = asyncio.get_event_loop().time() + 30

        async with aiohttp.ClientSession() as session:
            while self.running and asyncio.get_event_loop().time() < deadline:
                try:
                    async with session.get(
                        url,
                        params={"device_id": self.device_id, "token": self.token},
                        timeout=aiohttp.ClientTimeout(total=3)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            client_ip = data.get("client_ip")
                            client_port = data.get("client_port")
                            if client_ip and client_port:
                                logger.info(f"[UDP P2P] 拿到手机公网地址: {client_ip}:{client_port}，发送 PUNCH 包")
                                await self._send_punch(client_ip, client_port)
                                return  # 打洞包已发，退出轮询
                except asyncio.CancelledError:
                    return
                except Exception:
                    pass  # 404 或网络错误，继续等待

                await asyncio.sleep(2)

        logger.info("[UDP P2P] 打洞轮询超时，等待手机直接发 HELLO")

    async def _send_punch(self, ip: str, port: int):
        """向手机公网地址发送 PUNCH 包，打开本端 NAT 洞"""
        punch_msg = json.dumps({
            "type": "PUNCH",
            "token": self.token,
            "timestamp": int(datetime.now().timestamp())
        }).encode("utf-8")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.socket.sendto, punch_msg, (ip, port))
            logger.info(f"[UDP P2P] PUNCH 包已发送到 {ip}:{port}")
        except Exception as e:
            logger.error(f"[UDP P2P] 发送 PUNCH 包失败: {e}")


# 测试代码
async def test_udp_server():
    """测试 UDP 服务器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    server = UDPP2PServer(port=48920, token="test_token_123")

    try:
        await server.start()

        print("UDP P2P 服务器已启动")
        print("按 Ctrl+C 停止...")

        # 保持运行
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(test_udp_server())
