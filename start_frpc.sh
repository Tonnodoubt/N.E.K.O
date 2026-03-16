#!/bin/bash
# 启动 FRP 客户端（桌面端）

echo "================================"
echo "启动 FRP 客户端"
echo "================================"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRP_BIN="$PROJECT_DIR/bin/frpc"

# 检查配置文件
if [ ! -f "$PROJECT_DIR/frpc.toml" ]; then
    echo "❌ 配置文件 frpc.toml 不存在"
    exit 1
fi

# 检查是否已安装 frpc
if [ ! -f "$FRP_BIN" ]; then
    echo "⚠️  frpc 未安装，正在下载..."

    # macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        FRP_VERSION="0.52.3"
        mkdir -p "$PROJECT_DIR/bin"
        cd /tmp
        curl -L -o frp.tar.gz "https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_darwin_arm64.tar.gz"
        tar -xzf frp.tar.gz
        cp "frp_${FRP_VERSION}_darwin_arm64/frpc" "$FRP_BIN"
        chmod +x "$FRP_BIN"
        cd - > /dev/null
        echo "✅ frpc 已安装到 $FRP_BIN"
    else
        echo "❌ 请手动安装 frpc"
        echo "下载地址: https://github.com/fatedier/frp/releases"
        exit 1
    fi
fi

# 停止旧的 FRP 客户端
pkill -f "frpc.*frpc.toml" 2>/dev/null
sleep 1

# 启动 FRP 客户端
echo "启动 FRP 客户端..."
nohup "$FRP_BIN" -c "$PROJECT_DIR/frpc.toml" > /tmp/frpc_output.log 2>&1 &
FRP_PID=$!

sleep 2

# 检查是否启动成功
if ps -p $FRP_PID > /dev/null; then
    echo "✅ FRP 客户端已启动 (PID: $FRP_PID)"
    echo ""
    echo "日志文件: /tmp/frpc_output.log"
    echo "查看日志: tail -f /tmp/frpc_output.log"
    echo ""
    echo "================================"
    echo "FRP 连接信息"
    echo "================================"
    echo "服务器: 47.117.174.64:7000"
    echo "本地端口: 48920 (UDP)"
    echo "远程端口: 48920 (UDP)"
    echo ""
    echo "移动端连接地址: 47.117.174.64:48920"
    echo "================================"
else
    echo "❌ FRP 客户端启动失败"
    echo ""
    echo "查看错误日志:"
    cat /tmp/frpc_output.log
    exit 1
fi
