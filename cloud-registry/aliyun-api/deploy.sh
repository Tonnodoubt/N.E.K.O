#!/bin/bash
# N.E.K.O 云端注册服务快速部署脚本
# 适用于 CentOS 7+ / Rocky Linux / AlmaLinux

set -e

echo "==================================="
echo "N.E.K.O Cloud Registry 部署脚本"
echo "==================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否为 root 用户
if [ "$EUID" -eq 0 ]; then
  echo -e "${RED}请不要使用 root 用户运行此脚本${NC}"
  echo "使用普通用户: ./deploy.sh"
  exit 1
fi

# 获取当前用户名
USERNAME=$(whoami)
echo "当前用户: $USERNAME"

# 1. 更新系统
echo -e "${GREEN}[1/7] 更新系统...${NC}"
sudo dnf update -y

# 2. 安装依赖
echo -e "${GREEN}[2/7] 安装依赖...${NC}"
sudo dnf install -y python3 python3-pip redis

# 3. 启动 Redis
echo -e "${GREEN}[3/7] 启动 Redis...${NC}"
sudo systemctl start redis
sudo systemctl enable redis

# 验证 Redis
if redis-cli ping | grep -q "PONG"; then
    echo -e "${GREEN}Redis 启动成功${NC}"
else
    echo -e "${RED}Redis 启动失败${NC}"
    exit 1
fi

# 4. 创建项目目录
echo -e "${GREEN}[4/7] 创建项目目录...${NC}"
PROJECT_DIR="$HOME/neko-cloud-registry"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# 5. 创建虚拟环境并安装依赖
echo -e "${GREEN}[5/7] 安装 Python 依赖...${NC}"
python3 -m venv venv || {
    echo -e "${RED}虚拟环境创建失败，尝试安装 python3-venv${NC}"
    sudo dnf install -y python3-venv
    python3 -m venv venv
}
source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn redis python-dotenv

# 6. 下载 main.py（如果不存在）
if [ ! -f "main.py" ]; then
    echo -e "${GREEN}[6/7] 下载 main.py...${NC}"
    wget https://raw.githubusercontent.com/Tonnodoubt/N.E.K.O/feature/react_native/cloud-registry/aliyun-api/main.py -O main.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}main.py 下载成功${NC}"
    else
        echo -e "${RED}main.py 下载失败，请手动上传${NC}"
        echo "从本地电脑执行:"
        echo "scp cloud-registry/aliyun-api/main.py $USERNAME@服务器IP:$PROJECT_DIR/"
    fi
else
    echo -e "${GREEN}[6/7] main.py 已存在${NC}"
fi

# 7. 配置 systemd 服务
echo -e "${GREEN}[7/7] 配置 systemd 服务...${NC}"
SERVICE_FILE="/tmp/neko-registry.service"

cat > $SERVICE_FILE << EOF
[Unit]
Description=N.E.K.O Cloud Registry API
After=network.target redis.service

[Service]
Type=simple
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo cp $SERVICE_FILE /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable neko-registry

echo ""
echo -e "${GREEN}==================================="
echo "部署完成！"
echo "===================================${NC}"
echo ""
echo "接下来请执行："
echo ""
echo "1. 将 main.py 复制到: $PROJECT_DIR/"
echo "   scp main.py $USERNAME@服务器IP:$PROJECT_DIR/"
echo ""
echo "2. 启动服务:"
echo "   sudo systemctl start neko-registry"
echo ""
echo "3. 查看状态:"
echo "   sudo systemctl status neko-registry"
echo ""
echo "4. 查看日志:"
echo "   sudo journalctl -u neko-registry -f"
echo ""
echo "5. 测试 API:"
echo "   curl http://localhost:8000/api/health"
echo ""
echo "6. 配置阿里云安全组:"
echo "   开放端口 8000 (TCP)"
echo ""
