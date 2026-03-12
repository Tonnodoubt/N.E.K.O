"""
UPnP 端口映射管理器
自动在路由器上创建端口映射，实现跨网 P2P 连接
"""

import asyncio
import logging
from typing import Optional, Tuple
from async_upnp_client.client_factory import UpnpFactory
from async_upnp_client.aiohttp import AiohttpRequester
from async_upnp_client.search import async_search
from async_upnp_client.client import UpnpDevice

logger = logging.getLogger(__name__)


class UPnPManager:
    """UPnP 端口映射管理器"""

    def __init__(self, local_port: int, external_port: Optional[int] = None):
        """
        初始化 UPnP 管理器

        Args:
            local_port: 本地服务端口
            external_port: 外部端口（不指定则自动选择）
        """
        self.local_port = local_port
        self.external_port = external_port or local_port
        self.device: Optional[UpnpDevice] = None
        self.external_ip: Optional[str] = None
        self.mapping_enabled = False

    async def discover(self) -> bool:
        """
        发现局域网中的 UPnP 设备（路由器）

        Returns:
            是否发现成功
        """
        try:
            logger.info("[UPnP] 正在发现 UPnP 设备...")

            # 搜索 UPnP 设备
            discoveries = []

            async def on_discovery(discovery):
                """UPnP 设备发现回调"""
                discoveries.append(discovery)

            await async_search(
                async_callback=on_discovery,
                search_target="urn:schemas-upnp-org:device:InternetGatewayDevice:1",
                timeout=5
            )

            if not discoveries:
                logger.warning("[UPnP] 未发现 UPnP 设备")
                return False

            # 使用第一个发现的设备
            discovery = discoveries[0]
            logger.info(f"[UPnP] 发现设备: {discovery.location}")

            # 创建设备客户端
            requester = AiohttpRequester()
            factory = UpnpFactory(requester)
            self.device = await factory.async_create_device(discovery.location)

            logger.info(f"[UPnP] 连接到路由器: {self.device.name}")
            return True

        except Exception as e:
            logger.error(f"[UPnP] 发现设备失败: {e}")
            return False

    async def get_external_ip(self) -> Optional[str]:
        """
        获取路由器的外网 IP

        Returns:
            外网 IP 地址
        """
        if not self.device:
            logger.warning("[UPnP] 设备未连接")
            return None

        try:
            # 查找 WAN IP 连接服务
            service = self.device.service("urn:schemas-upnp-org:service:WANIPConnection:1")
            if not service:
                logger.warning("[UPnP] 未找到 WAN IP 连接服务")
                return None

            # 调用 GetExternalIPAddress
            action = service.action("GetExternalIPAddress")
            response = await action.async_call()

            self.external_ip = response.get("NewExternalIPAddress")
            logger.info(f"[UPnP] 外网 IP: {self.external_ip}")
            return self.external_ip

        except Exception as e:
            logger.error(f"[UPnP] 获取外网 IP 失败: {e}")
            return None

    async def add_port_mapping(
        self,
        protocol: str = "TCP",
        lease_duration: int = 3600
    ) -> bool:
        """
        添加端口映射

        Args:
            protocol: 协议类型（TCP/UDP）
            lease_duration: 租约时长（秒）

        Returns:
            是否成功
        """
        if not self.device:
            logger.warning("[UPnP] 设备未连接")
            return False

        try:
            # 获取本地 IP
            local_ip = self._get_local_ip()
            if not local_ip:
                return False

            # 查找端口映射服务
            service = self.device.service("urn:schemas-upnp-org:service:WANIPConnection:1")
            if not service:
                logger.warning("[UPnP] 未找到端口映射服务")
                return False

            # 添加端口映射
            action = service.action("AddPortMapping")
            await action.async_call(
                NewRemoteHost="",  # 空表示任意远程主机
                NewExternalPort=self.external_port,
                NewProtocol=protocol,
                NewInternalPort=self.local_port,
                NewInternalClient=local_ip,
                NewEnabled="1",
                NewPortMappingDescription=f"N.E.K.O-P2P-{self.local_port}",
                NewLeaseDuration=lease_duration
            )

            self.mapping_enabled = True
            logger.info(
                f"[UPnP] 端口映射成功: "
                f"{self.external_ip}:{self.external_port} → {local_ip}:{self.local_port}"
            )
            return True

        except Exception as e:
            logger.error(f"[UPnP] 添加端口映射失败: {e}")
            return False

    async def delete_port_mapping(self, protocol: str = "TCP") -> bool:
        """
        删除端口映射

        Args:
            protocol: 协议类型

        Returns:
            是否成功
        """
        if not self.device:
            return False

        try:
            service = self.device.service("urn:schemas-upnp-org:service:WANIPConnection:1")
            if not service:
                return False

            action = service.action("DeletePortMapping")
            await action.async_call(
                NewRemoteHost="",
                NewExternalPort=self.external_port,
                NewProtocol=protocol
            )

            self.mapping_enabled = False
            logger.info(f"[UPnP] 端口映射已删除: {self.external_port}")
            return True

        except Exception as e:
            logger.error(f"[UPnP] 删除端口映射失败: {e}")
            return False

    def get_connection_info(self) -> Optional[dict]:
        """
        获取连接信息

        Returns:
            连接信息字典
        """
        if not self.mapping_enabled or not self.external_ip:
            return None

        return {
            "upnp_ip": self.external_ip,
            "upnp_port": self.external_port,
            "local_port": self.local_port,
        }

    def _get_local_ip(self) -> Optional[str]:
        """
        获取本地 IP 地址

        Returns:
            本地 IP
        """
        import socket
        try:
            # 创建 UDP socket 连接到外网地址（不会真正发送数据）
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            logger.error(f"[UPnP] 获取本地 IP 失败: {e}")
            return None

    async def setup(self) -> bool:
        """
        一键设置 UPnP：发现设备 → 获取外网 IP → 添加端口映射

        Returns:
            是否成功
        """
        # 1. 发现设备
        if not await self.discover():
            return False

        # 2. 获取外网 IP
        if not await self.get_external_ip():
            return False

        # 3. 添加端口映射
        if not await self.add_port_mapping():
            return False

        logger.info(f"[UPnP] 设置完成，外网地址: {self.external_ip}:{self.external_port}")
        return True

    async def cleanup(self):
        """清理资源，删除端口映射"""
        if self.mapping_enabled:
            await self.delete_port_mapping()
        logger.info("[UPnP] 资源已清理")


# 测试代码
async def test_upnp():
    """测试 UPnP 功能"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    manager = UPnPManager(local_port=48920, external_port=48920)

    # 设置
    if await manager.setup():
        info = manager.get_connection_info()
        print(f"连接信息: {info}")

        # 保持运行
        print("按 Ctrl+C 退出...")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass

    # 清理
    await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(test_upnp())
