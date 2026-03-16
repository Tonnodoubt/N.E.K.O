#!/bin/bash
# 使用代理启动 N.E.K.O.TONG

export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

echo "🌐 代理已设置，当前 IP："
curl -s ipinfo.io | grep -E '"ip"|"country"'

# 调用原始启动脚本
cd "$(dirname "$0")"
./start.sh
