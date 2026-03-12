"""
N.E.K.O 云端地址注册服务 - FastAPI 版本
用于跨网 P2P 打洞的设备地址交换服务

部署方式：
1. 安装依赖：pip install fastapi uvicorn redis python-dotenv
2. 运行：python main.py
3. 后台运行：nohup python main.py > app.log 2>&1 &
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import redis
import json
import os
from datetime import datetime

# 初始化 FastAPI
app = FastAPI(title="N.E.K.O Cloud Registry")

# CORS 配置（允许跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 连接 Redis
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

# 数据模型
class DeviceRegister(BaseModel):
    device_id: str
    lan_ip: str
    token: str
    upnp_ip: Optional[str] = None
    upnp_port: Optional[int] = None
    character: Optional[str] = "default"

class DeviceInfo(BaseModel):
    device_id: str
    lan_ip: str
    token: str
    upnp_ip: Optional[str] = None
    upnp_port: Optional[int] = None
    character: Optional[str] = None
    created_at: int

# 健康检查
@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "status": "ok",
        "service": "N.E.K.O Cloud Registry",
        "timestamp": int(datetime.now().timestamp())
    }

@app.get("/api/health")
async def health():
    """API 健康检查"""
    try:
        redis_status = "connected" if redis_client.ping() else "disconnected"
    except:
        redis_status = "error"

    return {
        "status": "ok",
        "redis": redis_status,
        "timestamp": int(datetime.now().timestamp())
    }

# 注册设备
@app.post("/api/register")
async def register(device: DeviceRegister):
    """
    注册设备信息到云端
    TTL: 120 秒
    """
    try:
        # 构建设备信息
        device_info = {
            "device_id": device.device_id,
            "lan_ip": device.lan_ip,
            "token": device.token,
            "upnp_ip": device.upnp_ip,
            "upnp_port": device.upnp_port,
            "character": device.character,
            "created_at": int(datetime.now().timestamp())
        }

        # 存储到 Redis，TTL 120 秒
        key = f"device:{device.device_id}"
        redis_client.setex(
            key,
            120,
            json.dumps(device_info)
        )

        print(f"[register] Device {device.device_id} registered successfully")

        return {
            "success": True,
            "ttl": 120,
            "message": "Device registered successfully"
        }

    except Exception as e:
        print(f"[register] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询设备（阅后即焚）
@app.get("/api/lookup")
async def lookup(device_id: str):
    """
    查询设备信息（阅后即焚）
    查询后立即删除数据
    """
    try:
        key = f"device:{device_id}"
        data = redis_client.get(key)

        if not data:
            raise HTTPException(
                status_code=404,
                detail="Device not found or expired"
            )

        # 阅后即焚：查询后立即删除
        redis_client.delete(key)

        device_info = json.loads(data)
        print(f"[lookup] Device {device_id} found and deleted")

        return device_info

    except HTTPException:
        raise
    except Exception as e:
        print(f"[lookup] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询设备（POST 方式）
@app.post("/api/lookup")
async def lookup_post(request: dict):
    """
    查询设备信息（POST 方式，阅后即焚）
    """
    device_id = request.get("device_id")
    if not device_id:
        raise HTTPException(status_code=400, detail="Missing device_id")

    return await lookup(device_id)

if __name__ == "__main__":
    import uvicorn
    # 监听所有网络接口，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
