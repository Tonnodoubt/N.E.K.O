"""
云端地址注册客户端
用于将设备信息上报到云端注册服务，实现跨网 P2P 连接
"""

import asyncio
import os
import logging
from typing import Optional, Dict, Any
import aiohttp

logger = logging.getLogger(__name__)


class CloudRegistryClient:
    """云端地址注册客户端"""

    def __init__(self, registry_url: Optional[str] = None):
        """
        初始化客户端

        Args:
            registry_url: 云端注册服务地址（不指定则从环境变量读取）
        """
        self.registry_url = (
            registry_url
            or os.getenv("NEKO_CLOUD_REGISTRY_URL")
            or os.getenv("CLOUD_REGISTRY_URL")
        )
        if not self.registry_url:
            raise ValueError(
                "未配置云端注册服务地址，请设置 NEKO_CLOUD_REGISTRY_URL 或 CLOUD_REGISTRY_URL 环境变量"
            )
        self.registry_url = self.registry_url.strip().rstrip("/")

        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"[CloudRegistry] 初始化完成，服务地址: {self.registry_url}")

    async def register(
        self,
        device_id: str,
        lan_ip: str,
        token: str,
        stun_ip: Optional[str] = None,
        stun_port: Optional[int] = None,
        character: Optional[str] = None
    ) -> bool:
        """
        注册设备信息到云端

        Args:
            device_id: 设备唯一 ID
            lan_ip: LAN IP 地址
            token: 连接 token
            stun_ip: STUN 获取的公网 IP（可选）
            stun_port: STUN 获取的公网端口（可选）
            character: 角色名（可选）

        Returns:
            是否成功
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            payload = {
                "device_id": device_id,
                "lan_ip": lan_ip,
                "token": token,
            }

            # 添加可选字段
            if stun_ip:
                payload["stun_ip"] = stun_ip
            if stun_port:
                payload["stun_port"] = stun_port
            if character:
                payload["character"] = character

            url = f"{self.registry_url}/api/register"
            logger.info(f"[CloudRegistry] 注册设备到云端: {device_id}")

            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"[CloudRegistry] 注册成功，TTL: {data.get('ttl')} 秒")
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"[CloudRegistry] 注册失败: {resp.status} - {error}")
                    return False

        except asyncio.TimeoutError:
            logger.error("[CloudRegistry] 注册超时")
            return False
        except Exception as e:
            logger.error(f"[CloudRegistry] 注册异常: {e}")
            return False

    async def lookup(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        从云端查询设备信息（阅后即焚）

        Args:
            device_id: 设备唯一 ID

        Returns:
            设备信息字典（如果查询成功）
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            url = f"{self.registry_url}/api/lookup"
            params = {"device_id": device_id}

            logger.info(f"[CloudRegistry] 查询设备: {device_id}")

            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"[CloudRegistry] 查询成功: {data.get('lan_ip')}")
                    return data
                elif resp.status == 404:
                    logger.warning("[CloudRegistry] 设备未找到或已过期")
                    return None
                else:
                    error = await resp.text()
                    logger.error(f"[CloudRegistry] 查询失败: {resp.status} - {error}")
                    return None

        except asyncio.TimeoutError:
            logger.error("[CloudRegistry] 查询超时")
            return None
        except Exception as e:
            logger.error(f"[CloudRegistry] 查询异常: {e}")
            return None

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            服务是否正常
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            url = f"{self.registry_url}/api/health"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"[CloudRegistry] 服务正常: {data}")
                    return True
                return False
        except Exception as e:
            logger.error(f"[CloudRegistry] 健康检查失败: {e}")
            return False

    async def close(self):
        """关闭客户端连接"""
        if self.session:
            await self.session.close()
            self.session = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# 测试代码
async def test_client():
    """测试云端注册客户端"""
    # 加载 .env 文件
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 从环境变量读取或直接指定
    # os.environ["NEKO_CLOUD_REGISTRY_URL"] = "http://<your-cloud-host>:8000"

    async with CloudRegistryClient() as client:
        # 1. 健康检查
        print("1. 健康检查...")
        if await client.health_check():
            print("✅ 服务正常\n")

        # 2. 注册设备
        print("2. 注册设备...")
        success = await client.register(
            device_id="test_device_001",
            lan_ip="192.168.1.100",
            token="test_token_123",
            stun_ip="1.2.3.4",
            stun_port=48920
        )
        print(f"{'✅' if success else '❌'} 注册{'成功' if success else '失败'}\n")

        # 3. 查询设备
        print("3. 查询设备...")
        data = await client.lookup("test_device_001")
        if data:
            print(f"✅ 查询成功: {data}\n")
        else:
            print("❌ 设备未找到\n")

        # 4. 再次查询（应该找不到，阅后即焚）
        print("4. 再次查询（阅后即焚）...")
        data = await client.lookup("test_device_001")
        print(f"{'✅ 预期：未找到' if not data else '❌ 意外：找到数据'}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_client())
