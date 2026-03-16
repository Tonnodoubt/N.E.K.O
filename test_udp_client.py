"""
UDP P2P 客户端（模拟移动端）
用于测试 UDP 打洞功能
"""

import asyncio
import logging
import socket
import json
import aiohttp
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class UDPP2PClient:
    """UDP P2P 客户端（模拟移动端）"""

    def __init__(self, device_id: str, registry_url: str):
        """
        初始化客户端

        Args:
            device_id: 桌面端设备 ID
            registry_url: 云注册服务地址
        """
        self.device_id = device_id
        self.registry_url = registry_url
        self.socket: Optional[socket.socket] = None
        self.token: Optional[str] = None
        self.server_addr: Optional[Tuple[str, int]] = None
        self.connected = False

        logger.info(f"[UDP Client] 初始化，目标设备: {device_id}")

    async def connect(self, timeout: int = 10) -> bool:
        """
        三层连接策略

        Args:
            timeout: 连接超时时间

        Returns:
            是否连接成功
        """
        logger.info("[UDP Client] 开始三层连接...")

        # 第1层：LAN 直连
        if await self._try_lan_connect(timeout):
            logger.info("[UDP Client] [OK] 第1层成功：LAN 直连")
            return True

        logger.info("[UDP Client] 第1层失败，尝试第2层...")

        # 第2层：STUN 打洞
        if await self._try_stun_hole_punching(timeout):
            logger.info("[UDP Client] [OK] 第2层成功：STUN 打洞")
            return True

        logger.error("[UDP Client] [FAIL] 所有连接方式失败")
        return False

    async def _try_lan_connect(self, timeout: int) -> bool:
        """
        第1层：LAN 直连

        Returns:
            是否成功
        """
        try:
            logger.info("[UDP Client] 第1层：尝试 LAN 直连...")

            # 查询云端获取设备信息
            device_info = await self._query_cloud()
            if not device_info:
                return False

            lan_ip = device_info.get('lan_ip')
            # 使用云端返回的 stun_port 作为 UDP 服务器端口
            lan_port = device_info.get('stun_port', 48920)
            self.token = device_info.get('token')

            if not lan_ip or not self.token:
                return False

            # 尝试连接 LAN IP
            self.server_addr = (lan_ip, lan_port)
            logger.info(f"[UDP Client] LAN endpoint: {lan_ip}:{lan_port}")
            return await self._send_hello(timeout)

        except Exception as e:
            logger.debug(f"[UDP Client] LAN 连接失败: {e}")
            return False

    async def _try_stun_hole_punching(self, timeout: int) -> bool:
        """
        第2层：STUN 打洞

        Returns:
            是否成功
        """
        try:
            logger.info("[UDP Client] 第2层：尝试 STUN 打洞...")

            # 查询云端获取设备信息
            device_info = await self._query_cloud()
            if not device_info:
                return False

            stun_ip = device_info.get('stun_ip')
            stun_port = device_info.get('stun_port')
            self.token = device_info.get('token')

            if not stun_ip or not stun_port or not self.token:
                logger.warning("[UDP Client] STUN 信息缺失")
                return False

            logger.info(f"[UDP Client] STUN endpoint: {stun_ip}:{stun_port}")

            # 连接 STUN endpoint
            self.server_addr = (stun_ip, stun_port)
            return await self._send_hello(timeout)

        except Exception as e:
            logger.error(f"[UDP Client] STUN 打洞失败: {e}")
            return False

    async def _query_cloud(self) -> Optional[dict]:
        """
        查询云端获取设备信息

        Returns:
            设备信息字典
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.registry_url}/api/lookup"
                params = {"device_id": self.device_id}

                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"[UDP Client] 云端查询成功")
                        return data
                    else:
                        logger.error(f"[UDP Client] 云端查询失败: {resp.status}")
                        return None

        except Exception as e:
            logger.error(f"[UDP Client] 云端查询异常: {e}")
            return None

    async def _send_hello(self, timeout: int) -> bool:
        """
        发送握手消息

        Args:
            timeout: 超时时间

        Returns:
            是否成功
        """
        try:
            # 创建 UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(timeout)
            self.socket.bind(('0.0.0.0', 0))

            local_port = self.socket.getsockname()[1]
            logger.info(f"[UDP Client] 本地端口: {local_port}")

            # 发送 HELLO 消息
            hello_msg = {
                'type': 'HELLO',
                'token': self.token,
                'timestamp': int(datetime.now().timestamp())
            }

            data = json.dumps(hello_msg).encode('utf-8')
            logger.info(f"[UDP Client] 发送 HELLO 到 {self.server_addr}")
            self.socket.sendto(data, self.server_addr)

            # 等待 ACK
            loop = asyncio.get_event_loop()
            response_data, _ = await loop.run_in_executor(
                None,
                self.socket.recvfrom,
                4096
            )

            response = json.loads(response_data.decode('utf-8'))

            if response.get('type') == 'ACK':
                logger.info("[UDP Client] 收到 ACK，连接建立成功！")
                self.connected = True
                return True
            else:
                logger.error(f"[UDP Client] 收到非预期响应: {response}")
                return False

        except socket.timeout:
            logger.error("[UDP Client] 握手超时")
            return False
        except Exception as e:
            logger.error(f"[UDP Client] 握手失败: {e}")
            return False

    async def send_data(self, payload: str) -> bool:
        """
        发送数据

        Args:
            payload: 数据内容

        Returns:
            是否成功
        """
        if not self.connected or not self.socket:
            logger.error("[UDP Client] 未连接")
            return False

        try:
            message = {
                'type': 'DATA',
                'token': self.token,
                'payload': payload,
                'timestamp': int(datetime.now().timestamp())
            }

            data = json.dumps(message).encode('utf-8')
            self.socket.sendto(data, self.server_addr)
            logger.info(f"[UDP Client] 数据已发送: {payload}")

            return True

        except Exception as e:
            logger.error(f"[UDP Client] 发送数据失败: {e}")
            return False

    async def send_heartbeat(self) -> bool:
        """
        发送心跳

        Returns:
            是否成功
        """
        if not self.connected or not self.socket:
            return False

        try:
            message = {
                'type': 'HEARTBEAT',
                'token': self.token,
                'timestamp': int(datetime.now().timestamp())
            }

            data = json.dumps(message).encode('utf-8')
            self.socket.sendto(data, self.server_addr)
            logger.debug("[UDP Client] 心跳已发送")

            return True

        except Exception as e:
            logger.error(f"[UDP Client] 发送心跳失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            self.socket = None

        self.connected = False
        logger.info("[UDP Client] 连接已关闭")


# 测试代码
async def test_udp_client():
    """测试 UDP 客户端"""
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 配置
    device_id = "neko-3acee8daac4c4dea"  # 桌面端设备 ID
    registry_url = "http://47.117.174.64:8000"

    print("\n" + "="*60)
    print("UDP P2P 客户端测试（模拟移动端）")
    print("="*60)

    client = UDPP2PClient(device_id, registry_url)

    try:
        # 连接
        print(f"\n正在连接设备: {device_id}")
        if await client.connect():
            print("[OK] 连接成功！\n")

            # 发送测试数据
            print("发送测试数据...")
            await client.send_data("Hello from mobile!")
            await asyncio.sleep(1)

            # 发送心跳
            print("发送心跳...")
            await client.send_heartbeat()
            await asyncio.sleep(1)

            # 再发送一些数据
            print("发送更多数据...")
            await client.send_data("This is a test message")
            await asyncio.sleep(1)

            print("\n测试完成！")
        else:
            print("[FAIL] 连接失败")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        client.close()

    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(test_udp_client())
