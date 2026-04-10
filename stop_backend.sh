#!/bin/bash
# 停止后端服务

cd "$(dirname "$0")"

if [ -f .backend.pid ]; then
    BACKEND_PID=$(cat .backend.pid)
    kill $BACKEND_PID 2>/dev/null && echo "✅ 后端已停止"
    rm .backend.pid
fi

if [ -f .frpc.pid ]; then
    FRP_PID=$(cat .frpc.pid)
    kill $FRP_PID 2>/dev/null && echo "✅ frp 已停止"
    rm .frpc.pid
fi

echo "✅ 服务已全部停止"
