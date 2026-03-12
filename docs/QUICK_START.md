# N.E.K.O 跨网 P2P 快速开始

## 🎯 当前状态

✅ **云端地址注册服务** 已部署（阿里云）
- 服务器地址：`http://47.117.174.64:8000`
- API 端点：
  - `POST /api/register` - 注册设备
  - `GET /api/lookup?device_id=xxx` - 查询设备（阅后即焚）
  - `GET /api/health` - 健康检查

✅ **桌面端模块** 已实现
- `upnp_manager.py` - UPnP 端口映射管理器
- `cloud_registry_client.py` - 云端注册客户端

🚧 **待完成**
- 桌面端集成（扩展 lan_proxy.py）
- 移动端三层连接逻辑
- 端到端测试

---

## 📋 快速测试

### 1. 测试云端 API

在本地 Mac 测试：

```bash
# 健康检查
curl http://47.117.174.64:8000/api/health

# 注册设备
curl -X POST "http://47.117.174.64:8000/api/register" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test_device",
    "lan_ip": "192.168.1.100",
    "token": "test_token"
  }'

# 查询设备（只能查询一次，第二次会404）
curl "http://47.117.174.64:8000/api/lookup?device_id=test_device"
```

### 2. 测试 UPnP 管理器

```bash
# 运行 UPnP 测试（需要路由器支持UPnP）
python3 upnp_manager.py
```

### 3. 测试云端注册客户端

```bash
# 测试客户端
python3 cloud_registry_client.py
```

---

## 🔧 下一步：集成到 lan_proxy.py

需要扩展 `lan_proxy.py`，添加：
1. UPnP 自动映射
2. 云端自动注册
3. 定期刷新注册信息

### 扩展后的使用流程

```python
# 启动 lan_proxy 时会自动：
# 1. 启动 HTTP 反向代理（同WiFi连接）
# 2. 尝试 UPnP 端口映射（跨网打洞）
# 3. 注册到云端（地址交换）

# 移动端连接流程：
# 1. 尝试 LAN 直连（同WiFi）
# 2. 失败则查询云端获取UPnP地址
# 3. 通过 UPnP 地址连接
```

---

## 🌐 架构说明

### 三层连接策略

```
移动端 → 桌面端

第1层：LAN 直连
  192.168.x.x:48920 (同WiFi)
  ↓ 失败
第2层：UPnP 打洞
  查询云端 → 获取公网IP:端口 → 连接
  ↓ 失败
第3层：FRP 穿透
  通过 FRP 服务器中转
```

### 云端注册流程

```
桌面端:
  启动 → UPnP 映射 → 注册到云端
           ↓
    {
      device_id: "neko-xxx",
      lan_ip: "192.168.1.100",
      upnp_ip: "1.2.3.4",
      upnp_port: 48920,
      token: "..."
    }

移动端:
  扫码获取 device_id → 查询云端 → 获取地址 → 连接
```

---

## ⚙️ 配置

### 环境变量（.env）

```bash
# 云端注册服务地址
NEKO_CLOUD_REGISTRY_URL=http://47.117.174.64:8000
```

### 安装依赖

```bash
# Python 依赖
pip install async-upnp-client aiohttp redis

# 或使用 uv
uv add async-upnp-client aiohttp redis
```

---

## 🐛 故障排查

### UPnP 映射失败

**可能原因：**
1. 路由器未启用 UPnP
2. 运营商使用 CGNAT（无法UPnP）
3. 防火墙阻止

**解决方法：**
- 登录路由器管理页面，启用 UPnP
- 联系运营商询问是否支持UPnP
- 使用 FRP 第三层方案

### 云端注册失败

**检查步骤：**
1. 测试云端 API：`curl http://47.117.174.64:8000/api/health`
2. 检查 `.env` 配置
3. 查看服务器日志：`sudo journalctl -u neko-registry -f`

---

## 📚 相关文档

- [阿里云部署指南](docs/ALIYUN_DEPLOYMENT.md)
- [Vercel 使用指南](cloud-registry/vercel-api/VERCEL_GUIDE.md)（已废弃）
- [API 文档](cloud-registry/aliyun-api/README.md)

---

## 📝 TODO

- [ ] 集成到 lan_proxy.py
- [ ] 移动端三层连接逻辑
- [ ] 端到端测试
- [ ] 性能优化
- [ ] 错误处理增强
