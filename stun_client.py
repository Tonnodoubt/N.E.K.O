"""
STUN 客户端
用于通过 STUN 服务器获取公网 endpoint (IP:端口)
"""

import asyncio
import logging
import socket
import struct
import random
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class STUNClient:
    """STUN 客户端"""

    def __init__(self, stun_server: str, stun_port: int = 3478):
        """
        初始化 STUN 客户端

        Args:
            stun_server: STUN 服务器地址
            stun_port: STUN 服务器端口（默认 3478）
        """
        self.stun_server = stun_server
        self.stun_port = stun_port
        self.socket: Optional[socket.socket] = None
        logger.info(f"[STUN] 初始化客户端，服务器: {stun_server}:{stun_port}")

    async def get_public_endpoint(self, timeout: int = 5) -> Optional[Tuple[str, int]]:
        """
        获取公网 endpoint

        Returns:
            (公网IP, 端口) 或 None（失败）
        """
        try:
            # 创建 UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(timeout)

            # 绑定到随机端口
            self.socket.bind(('0.0.0.0', 0))
            local_port = self.socket.getsockname()[1]
            logger.info(f"[STUN] 本地端口: {local_port}")

            # 构建 STUN Binding Request
            request = self._build_binding_request()

            # 发送到 STUN 服务器
            server_addr = (self.stun_server, self.stun_port)
            logger.info(f"[STUN] 发送请求到 {server_addr}")
            self.socket.sendto(request, server_addr)

            # 接收响应
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.socket.recvfrom,
                1024
            )

            data, _ = response

            # 解析响应，提取公网 endpoint
            public_ip, public_port = self._parse_binding_response(data)

            logger.info(f"[STUN] 公网 endpoint: {public_ip}:{public_port}")
            return (public_ip, public_port)

        except socket.timeout:
            logger.error("[STUN] 请求超时")
            return None
        except Exception as e:
            logger.error(f"[STUN] 获取公网 endpoint 失败: {e}")
            return None
        finally:
            if self.socket:
                self.socket.close()
                self.socket = None

    def _build_binding_request(self) -> bytes:
        """
        构建 STUN Binding Request

        STUN Header 格式：
        - 2 bytes: Message Type (0x0001 = Binding Request)
        - 2 bytes: Message Length
        - 4 bytes: Magic Cookie (0x2112A442)
        - 12 bytes: Transaction ID (随机)
        """
        # Message Type: Binding Request (0x0001)
        message_type = struct.pack('!H', 0x0001)

        # Message Length: 0 (no attributes)
        message_length = struct.pack('!H', 0x0000)

        # Magic Cookie (RFC 5389)
        magic_cookie = struct.pack('!I', 0x2112A442)

        # Transaction ID (12 bytes random)
        transaction_id = struct.pack('!12s', random.randbytes(12))

        # 拼接请求
        request = message_type + message_length + magic_cookie + transaction_id
        return request

    def _parse_binding_response(self, data: bytes) -> Tuple[str, int]:
        """
        解析 STUN Binding Response

        Args:
            data: STUN 响应数据

        Returns:
            (公网IP, 端口)

        Raises:
            ValueError: 解析失败
        """
        if len(data) < 20:
            raise ValueError("响应数据太短")

        # 解析 Header
        message_type = struct.unpack('!H', data[0:2])[0]
        message_length = struct.unpack('!H', data[2:4])[0]
        magic_cookie = struct.unpack('!I', data[4:8])[0]

        # 验证 Magic Cookie
        if magic_cookie != 0x2112A442:
            raise ValueError(f"无效的 Magic Cookie: {hex(magic_cookie)}")

        # 验证 Message Type (0x0101 = Binding Response)
        if message_type != 0x0101:
            raise ValueError(f"不是 Binding Response: {hex(message_type)}")

        # 解析 Attributes
        offset = 20  # Header 长度
        while offset < len(data):
            # Attribute Header
            attr_type = struct.unpack('!H', data[offset:offset+2])[0]
            attr_length = struct.unpack('!H', data[offset+2:offset+4])[0]

            # XOR-MAPPED-ADDRESS (0x0020) 或 MAPPED-ADDRESS (0x0001)
            if attr_type == 0x0020:
                # 解析 XOR-MAPPED-ADDRESS
                return self._parse_xor_mapped_address(
                    data[offset+4:offset+4+attr_length],
                    magic_cookie,
                    data[8:20]  # Transaction ID
                )
            elif attr_type == 0x0001:
                # 解析 MAPPED-ADDRESS
                return self._parse_mapped_address(
                    data[offset+4:offset+4+attr_length]
                )

            # 移动到下一个 Attribute
            offset += 4 + attr_length
            # Padding to 4-byte boundary
            if attr_length % 4 != 0:
                offset += 4 - (attr_length % 4)

        raise ValueError("未找到 MAPPED-ADDRESS 属性")

    def _parse_xor_mapped_address(
        self,
        data: bytes,
        magic_cookie: int,
        transaction_id: bytes
    ) -> Tuple[str, int]:
        """
        解析 XOR-MAPPED-ADDRESS

        Args:
            data: 属性值
            magic_cookie: Magic Cookie
            transaction_id: Transaction ID

        Returns:
            (IP, 端口)
        """
        # 第一个字节: 保留 (0x00)
        # 第二个字节: 地址族 (0x01 = IPv4, 0x02 = IPv6)
        family = struct.unpack('!B', data[1:2])[0]

        if family == 0x01:
            # IPv4
            # 端口：XOR with Magic Cookie 高 16 位
            xored_port = struct.unpack('!H', data[2:4])[0]
            port = xored_port ^ (magic_cookie >> 16)

            # IP: XOR with Magic Cookie
            xored_ip = struct.unpack('!I', data[4:8])[0]
            ip_int = xored_ip ^ magic_cookie
            ip = socket.inet_ntoa(struct.pack('!I', ip_int))

            return (ip, port)
        else:
            raise ValueError(f"不支持的地址族: {family}")

    def _parse_mapped_address(self, data: bytes) -> Tuple[str, int]:
        """
        解析 MAPPED-ADDRESS

        Args:
            data: 属性值

        Returns:
            (IP, 端口)
        """
        # 第一个字节: 保留 (0x00)
        # 第二个字节: 地址族 (0x01 = IPv4, 0x02 = IPv6)
        family = struct.unpack('!B', data[1:2])[0]

        if family == 0x01:
            # IPv4
            port = struct.unpack('!H', data[2:4])[0]
            ip = socket.inet_ntoa(data[4:8])
            return (ip, port)
        else:
            raise ValueError(f"不支持的地址族: {family}")

    async def close(self):
        """关闭客户端"""
        if self.socket:
            self.socket.close()
            self.socket = None


# 测试代码
async def test_stun():
    """测试 STUN 客户端"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 使用 Google STUN 服务器测试
    print("测试 Google STUN 服务器...")
    client = STUNClient("stun.l.google.com", 19302)
    endpoint = await client.get_public_endpoint()
    if endpoint:
        print(f"✅ 公网 endpoint: {endpoint[0]}:{endpoint[1]}")
    else:
        print("❌ 获取失败")

    await client.close()

    # 测试阿里云 STUN 服务器（如果已部署）
    print("\n测试阿里云 STUN 服务器...")
    client = STUNClient("47.117.174.64", 3478)
    endpoint = await client.get_public_endpoint()
    if endpoint:
        print(f"✅ 公网 endpoint: {endpoint[0]}:{endpoint[1]}")
    else:
        print("❌ 获取失败（服务器可能未部署）")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_stun())
