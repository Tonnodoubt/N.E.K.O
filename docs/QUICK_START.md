# N.E.K.O 跨网 P2P 快速开始

## 🎯 当前状态

✅ **云端地址注册服务** 已部署（阿里云）
- 服务器地址：`http://47.117.174.64:8000`
- API 端点：
  - `POST /api/register` - 注册设备
  - `GET /api/lookup?device_id=xxx` - 查询设备（阅后即焚）
  - `GET /api/health` - 健康检查

✅ **STUN 服务器** 已支持（阿里云）
- 服务器地址：`47.117.174.64:3478`
- 用于 NAT 穿透，获取公网 endpoint
- 部署指南：[docs/STUN_DEPLOYMENT.md](STUN_DEPLOYMENT.md)

✅ **桌面端模块** 已实现
- `upnp_manager.py` - UPnP 端口映射管理器
- `stun_client.py` - STUN 客户端（获取公网 endpoint）
- `cloud_registry_client.py` - 云端注册客户端
- `lan_proxy.py` - 完整集成（UPnP + STUN + 云注册）

🚧 **待完成**
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

### 4. 测试 STUN 客户端

```bash
# 测试 STUN 功能（使用 Google STUN 服务器）
.venv/bin/python3 stun_client.py
```

### 5. 测试完整 STUN 集成

```bash
# 测试 STUN + 云注册集成
.venv/bin/python3 test_stun.py
```

---

## 🔧 下一步：集成到 lan_proxy.py

✅ **已完成！** `lan_proxy.py` 已集成：
1. ✅ UPnP 自动映射（可选，需要路由器支持）
2. ✅ STUN 公网 endpoint 发现（推荐）
3. ✅ 云端自动注册（包含 UPnP 和 STUN 信息）
4. ✅ 定期刷新注册信息

### 使用流程

```python
# 启动 lan_proxy 时会自动：
# 1. 启动 HTTP 反向代理（同WiFi连接）
# 2. 尝试 UPnP 端口映射（如果路由器支持）
# 3. 通过 STUN 获取公网 endpoint（推荐）
# 4. 注册到云端（地址交换）

# 移动端连接流程：
# 1. 尝试 LAN 直连（同WiFi）
# 2. 失败则查询云端获取 STUN endpoint
# 3. 通过 STUN endpoint 连接（UDP 打洞）
```

---

## 🌐 架构说明

### 三层连接策略

```
移动端 → 桌面端

第1层：LAN 直连
  192.168.x.x:48920 (同WiFi)
  ↓ 失败
第2层：STUN 打洞 ← 推荐使用！
  通过 STUN 服务器发现公网 endpoint → UDP 打洞 → P2P 连接
  成功率：70-80%
  ↓ 失败
第3层：FRP 中转
  通过 FRP 服务器中继流量
  成功率：100%（但有延迟和带宽成本）
```

**STUN vs UPnP：**
- UPnP：需要路由器支持，成功率约 30%
- STUN：无需路由器配置，成功率 70-80%（推荐）

### 云端注册流程

```
桌面端:
  启动 → UPnP 映射（可选）→ STUN 获取公网 endpoint → 注册到云端
           ↓
    {
      device_id: "neko-xxx",
      lan_ip: "192.168.1.100",
      upnp_ip: "1.2.3.4",           // 可选
      upnp_port: 48920,             // 可选
      stun_ip: "5.6.7.8",           // 推荐
      stun_port: 12345,             // 推荐
      token: "..."
    }

移动端:
  扫码获取 device_id → 查询云端 → 获取 endpoint → 连接
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

- [STUN 服务器部署指南](docs/STUN_DEPLOYMENT.md)
- [阿里云部署指南](docs/ALIYUN_DEPLOYMENT.md)
- [Vercel 使用指南](cloud-registry/vercel-api/VERCEL_GUIDE.md)（已废弃）
- [API 文档](cloud-registry/aliyun-api/README.md)

---

## 📝 TODO

- [x] 集成 STUN 到 lan_proxy.py
- [ ] 部署 STUN 服务器到阿里云
- [ ] 移动端三层连接逻辑
- [ ] UDP 打洞实现（移动端）
- [ ] 端到端测试
- [ ] 性能优化
- [ ] 错误处理增强
