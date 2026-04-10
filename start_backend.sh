#!/bin/bash
# 滑雪项目后端服务启动脚本

cd "$(dirname "$0")"

echo "🚀 启动滑雪分析后端服务..."

# 检查 frpc 是否安装
if ! command -v frpc &> /dev/null; then
    echo "❌ frpc 未安装，正在安装..."
    brew install frp
fi

# 启动后端 API
echo "📡 启动后端 API (端口 8765)..."
/Users/murphy/miniconda3/envs/mujoco/bin/python backend_api.py &
BACKEND_PID=$!
echo "✅ 后端 PID: $BACKEND_PID"

# 等待后端启动
sleep 3

# 启动 frp 客户端
echo "🔗 启动 frp 隧道连接香港服务器..."
frpc -c frpc.toml &
FRP_PID=$!
echo "✅ frp PID: $FRP_PID"

echo ""
echo "✅ 服务已启动！"
echo "   后端 API: http://localhost:8765"
echo "   香港访问: http://103.251.89.147:8765"
echo ""
echo "停止服务: kill $BACKEND_PID $FRP_PID"
echo "或运行: ./stop_backend.sh"

# 保存 PID
echo "$BACKEND_PID" > .backend.pid
echo "$FRP_PID" > .frpc.pid
