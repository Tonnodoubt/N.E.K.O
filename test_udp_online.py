"""
UDP P2P 客户端测试（最小化版本）
可在 Repl.it 或 Google Colab 运行
"""

import asyncio
import socket
import json
import aiohttp
from datetime import datetime

# ============ 配置 ============
DEVICE_ID = "neko-3acee8daac4c4dea"  # 替换为你的设备 ID
REGISTRY_URL = "http://47.117.174.64:8000"
# ==============================

async def test_udp_client():
    """测试 UDP 客户端"""

    print("\n" + "="*70)
    print("UDP P2P 客户端测试（在线版）")
    print("="*70)
    print(f"目标设备: {DEVICE_ID}")
    print(f"云端服务: {REGISTRY_URL}")
    print("-" * 70)

    # 查询云端
    print("\n[步骤 1] 查询云端...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{REGISTRY_URL}/api/lookup",
                params={"device_id": DEVICE_ID},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    print(f"❌ 查询失败: {resp.status}")
                    return

                device_info = await resp.json()
                print(f"✅ 查询成功")
                print(f"   LAN IP: {device_info.get('lan_ip')}")
                print(f"   STUN IP: {device_info.get('stun_ip')}")
                print(f"   STUN Port: {device_info.get('stun_port')}")
                print(f"   FRP IP: {device_info.get('frp_ip')}")
                print(f"   FRP Port: {device_info.get('frp_port')}")
                print(f"   FRP IP: {device_info.get('frp_ip')}")
                print(f"   FRP Port: {device_info.get('frp_port')}")
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return

    # 测试第1层（LAN）
    print("\n[步骤 2] 尝试第1层：LAN 直连...")
    lan_ip = device_info.get('lan_ip')
    lan_port = device_info.get('stun_port', 48920)
    token = device_info.get('token')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)

    try:
        # 发送 HELLO
        hello = {
            'type': 'HELLO',
            'token': token,
            'timestamp': int(datetime.now().timestamp())
        }
        sock.sendto(json.dumps(hello).encode(), (lan_ip, lan_port))
        print(f"   发送 HELLO 到 {lan_ip}:{lan_port}")

        # 等待 ACK
        data, _ = sock.recvfrom(4096)
        response = json.loads(data.decode())

        if response.get('type') == 'ACK':
            print("\n✅ 第1层成功：LAN 直连")
            return
    except socket.timeout:
        print("   ⏱️  第1层超时")
    except Exception as e:
        print(f"   ❌ 第1层失败: {e}")
    finally:
        sock.close()

    # 测试第2层（STUN）
    print("\n[步骤 3] 尝试第2层：STUN 打洞...")
    stun_ip = device_info.get('stun_ip')
    stun_port = device_info.get('stun_port')

    if not stun_ip or not stun_port:
        print("❌ STUN 信息缺失")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10)

    try:
        # 发送 HELLO 到 STUN endpoint
        hello = {
            'type': 'HELLO',
            'token': token,
            'timestamp': int(datetime.now().timestamp())
        }
        sock.sendto(json.dumps(hello).encode(), (stun_ip, stun_port))
        print(f"   发送 HELLO 到 {stun_ip}:{stun_port}")

        # 等待 ACK
        data, _ = sock.recvfrom(4096)
        response = json.loads(data.decode())

        if response.get('type') == 'ACK':
            print("\n✅ 第2层成功：STUN 打洞")
            return
    except socket.timeout:
        print("   ⏱️  第2层超时")
        print("   可能原因：NAT 不支持打洞，或防火墙阻止")
    except Exception as e:
        print(f"   ❌ 第2层失败: {e}")
    finally:
        sock.close()

    # 测试第3层（FRP 中转）
    print("\n[步骤 4] 尝试第3层：FRP 中转...")
    frp_ip = device_info.get('frp_ip')
    frp_port = device_info.get('frp_port')

    if not frp_ip or not frp_port:
        print("❌ FRP 信息缺失")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(10)

    try:
        # 发送 HELLO 到 FRP 中转
        hello = {
            'type': 'HELLO',
            'token': token,
            'timestamp': int(datetime.now().timestamp())
        }
        sock.sendto(json.dumps(hello).encode(), (frp_ip, frp_port))
        print(f"   发送 HELLO 到 {frp_ip}:{frp_port}")

        # 等待 ACK
        data, _ = sock.recvfrom(4096)
        response = json.loads(data.decode())

        if response.get('type') == 'ACK':
            print("\n✅ 第3层成功：FRP 中转")
            print("   通过 FRP 中转服务器连接成功")
            return
    except socket.timeout:
        print("\n❌ 第3层超时")
        print("   可能原因：FRP 服务未启动")
    except Exception as e:
        print(f"\n❌ 第3层失败: {e}")
    finally:
        sock.close()

    print("\n❌ 所有连接方式失败")
    print("\n" + "="*70 + "\n")

# 在线运行入口
if __name__ == "__main__":
    # 安装依赖
    import subprocess
    subprocess.run(["pip", "install", "-q", "aiohttp"])

    # 运行测试
    asyncio.run(test_udp_client())
