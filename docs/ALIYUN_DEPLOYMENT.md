# 阿里云服务器部署方案

## 架构概览

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  桌面端     │ ──注册─→ │  阿里云 API  │ ←─查询── │  移动端     │
│  (Python)   │         │  (FastAPI)   │         │  (RN)       │
│             │         │              │         │             │
│  UPnP 映射  │         │  Redis 存储  │         │  P2P 连接   │
└─────────────┘         └──────────────┘         └─────────────┘
```

## 准备工作

### 1. 阿里云服务器要求

- **系统：** Ubuntu 20.04+ 或 CentOS 7+
- **配置：** 1核2G 内存即可（免费试用或最低配置）
- **网络：** 公网 IP，开放端口 8000（或自定义）
- **安全组：** 需要开放 HTTP 端口

### 2. 需要的域名（可选）

如果有域名，可以配置 HTTPS（推荐）。没有域名可以直接用 IP。

## 部署步骤

### 第一步：安装依赖

SSH 连接到服务器，执行：

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip -y

# 安装 Redis
sudo apt install redis-server -y

# 启动 Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# 验证 Redis 运行
redis-cli ping
# 应该返回: PONG
```

### 第二步：创建项目目录

```bash
# 创建项目目录
mkdir -p ~/neko-cloud-registry
cd ~/neko-cloud-registry

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 Python 依赖
pip install fastapi uvicorn redis python-dotenv
```

### 第三步：创建 API 代码

创建 `main.py` 文件：

```bash
nano main.py
```

粘贴以下代码：

```python
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
    return {
        "status": "ok",
        "service": "N.E.K.O Cloud Registry",
        "timestamp": int(datetime.now().timestamp())
    }

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "redis": "connected" if redis_client.ping() else "disconnected",
        "timestamp": int(datetime.now().timestamp())
    }

# 注册设备
@app.post("/api/register")
async def register(device: DeviceRegister):
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

        print(f"[register] Device {device.device_id} registered")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 第四步：创建环境变量文件（可选）

创建 `.env` 文件：

```bash
nano .env
```

内容：

```
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 第五步：启动服务

**测试运行：**

```bash
# 激活虚拟环境
source venv/bin/activate

# 启动服务
python main.py
```

**后台运行（生产环境）：**

```bash
# 使用 nohup 后台运行
nohup python main.py > app.log 2>&1 &

# 查看日志
tail -f app.log

# 停止服务
ps aux | grep main.py
kill <PID>
```

**或者使用 systemd（推荐）：**

创建服务文件：

```bash
sudo nano /etc/systemd/system/neko-registry.service
```

内容：

```ini
[Unit]
Description=N.E.K.O Cloud Registry API
After=network.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/home/你的用户名/neko-cloud-registry
Environment="PATH=/home/你的用户名/neko-cloud-registry/venv/bin"
ExecStart=/home/你的用户名/neko-cloud-registry/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
# 重新加载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start neko-registry

# 设置开机自启
sudo systemctl enable neko-registry

# 查看状态
sudo systemctl status neko-registry

# 查看日志
sudo journalctl -u neko-registry -f
```

### 第六步：配置阿里云安全组

1. 登录阿里云控制台
2. 找到你的 ECS 实例
3. 点击 **安全组** → **配置规则**
4. 添加入站规则：
   - **端口范围：** 8000（或你使用的端口）
   - **授权对象：** 0.0.0.0/0
   - **协议类型：** TCP

### 第七步：测试 API

**健康检查：**

```bash
curl http://你的服务器IP:8000/api/health
```

**注册设备：**

```bash
curl -X POST "http://你的服务器IP:8000/api/register" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test_device_001",
    "lan_ip": "192.168.1.100",
    "token": "test_token_123"
  }'
```

**查询设备：**

```bash
curl "http://你的服务器IP:8000/api/lookup?device_id=test_device_001"
```

## 可选：配置 HTTPS

### 使用 Nginx 反向代理 + Let's Encrypt

**1. 安装 Nginx：**

```bash
sudo apt install nginx -y
```

**2. 安装 Certbot：**

```bash
sudo apt install certbot python3-certbot-nginx -y
```

**3. 配置 Nginx：**

```bash
sudo nano /etc/nginx/sites-available/neko-registry
```

内容：

```nginx
server {
    listen 80;
    server_name 你的域名.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/neko-registry /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**4. 获取 SSL 证书：**

```bash
sudo certbot --nginx -d 你的域名.com
```

**5. 自动续期：**

```bash
sudo certbot renew --dry-run
```

## 桌面端配置

在 `.env` 文件中添加：

```bash
NEKO_CLOUD_REGISTRY_URL=http://你的服务器IP:8000
# 或使用 HTTPS
# NEKO_CLOUD_REGISTRY_URL=https://你的域名.com
```

## 成本估算

- **阿里云 ECS：**
  - 1核2G：约 50-80 元/月
  - 或使用免费试用（1个月）

- **域名（可选）：**
  - .com 域名：约 55 元/年
  - .top/.xyz 等：约 10 元/年

- **Redis：**
  - 本地安装：免费
  - 阿里云 Redis：约 100 元/月（不需要，用本地的即可）

**总成本：** 50-80 元/月（不含域名）或 0 元（使用免费试用）

## 常见问题

### 1. 端口被占用

```bash
# 查看端口占用
sudo lsof -i :8000

# 杀掉进程
sudo kill -9 <PID>
```

### 2. Redis 连接失败

```bash
# 检查 Redis 状态
sudo systemctl status redis-server

# 重启 Redis
sudo systemctl restart redis-server

# 测试连接
redis-cli ping
```

### 3. 防火墙问题

```bash
# Ubuntu UFW
sudo ufw allow 8000

# CentOS Firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### 4. 外网无法访问

检查：
1. 阿里云安全组是否开放端口
2. 服务器防火墙是否允许
3. 服务是否监听 0.0.0.0（不是 127.0.0.1）

## 监控和日志

**查看实时日志：**

```bash
# systemd 方式
sudo journalctl -u neko-registry -f

# nohup 方式
tail -f app.log
```

**监控 Redis：**

```bash
redis-cli monitor
```

## 备份和恢复

**Redis 数据备份：**

```bash
# 自动备份（Redis 默认开启 RDB）
# 数据文件位置：/var/lib/redis/dump.rdb

# 手动触发备份
redis-cli BGSAVE
```

## 下一步

1. 部署完成后，更新桌面端 `.env` 配置
2. 实现移动端连接逻辑
3. 测试完整的跨网 P2P 流程

---

**相关文档：**
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Redis 官方文档](https://redis.io/documentation)
- [阿里云 ECS 文档](https://help.aliyun.com/product/25365.html)
