# 阿里云部署文件

这个目录包含在阿里云服务器上部署 N.E.K.O 云端注册服务的所有文件。

## 文件说明

- **main.py** - FastAPI 服务主程序（核心代码）
- **deploy.sh** - 一键部署脚本（Ubuntu）
- **neko-registry.service** - systemd 服务配置文件模板

## 快速部署

### 方法 1：使用部署脚本（推荐）

1. **在服务器上运行部署脚本：**

```bash
# 下载脚本
wget https://raw.githubusercontent.com/你的仓库地址/main/cloud-registry/aliyun-api/deploy.sh

# 添加执行权限
chmod +x deploy.sh

# 运行脚本
./deploy.sh
```

2. **上传 API 代码：**

```bash
# 在本地执行
scp main.py 你的用户名@服务器IP:~/neko-cloud-registry/
```

3. **启动服务：**

```bash
# 在服务器执行
sudo systemctl start neko-registry
sudo systemctl status neko-registry
```

### 方法 2：手动部署

参考详细文档：[docs/ALIYUN_DEPLOYMENT.md](../../docs/ALIYUN_DEPLOYMENT.md)

## 测试 API

```bash
# 健康检查
curl http://你的服务器IP:8000/api/health

# 注册设备
curl -X POST "http://你的服务器IP:8000/api/register" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test_device",
    "lan_ip": "192.168.1.100",
    "token": "test_token"
  }'

# 查询设备
curl "http://你的服务器IP:8000/api/lookup?device_id=test_device"
```

## 配置桌面端

在项目根目录的 `.env` 文件中添加：

```bash
NEKO_CLOUD_REGISTRY_URL=http://你的服务器IP:8000
```

## 故障排查

查看日志：
```bash
sudo journalctl -u neko-registry -f
```

检查 Redis：
```bash
redis-cli ping
```

检查端口：
```bash
sudo netstat -tulpn | grep 8000
```

## 成本

- 阿里云 ECS 1核2G：约 50-80 元/月
- Redis（本地）：免费
- 总计：50-80 元/月

## 相关文档

- [完整部署指南](../../docs/ALIYUN_DEPLOYMENT.md)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Redis 文档](https://redis.io/documentation)
