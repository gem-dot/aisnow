# 滑雪项目部署指南

## 架构说明

```
用户浏览器
    ↓
香港服务器 (103.251.89.147:3000) - Next.js 前端
    ↓ 转发 API 请求
香港服务器 (103.251.89.147:8765) - frp 隧道
    ↓
Mac mini (本地:8765) - Python 后端分析引擎
```

## 已完成配置

### 1. Mac mini 本地服务 ✅
- 后端 API: `/Users/murphy/Desktop/Ski_Avatar_Pro/backend_api.py`
- frp 配置: `/Users/murphy/Desktop/Ski_Avatar_Pro/frpc.toml`
- 启动脚本: `./start_backend.sh`
- 停止脚本: `./stop_backend.sh`

### 2. 前端 API 转发 ✅
- 已修改 `app/api/analyze/route.ts` 为转发模式
- 通过环境变量 `BACKEND_URL` 配置后端地址

### 3. frp 隧道 ✅
- 香港服务器 frps 已运行 (端口 7000)
- Mac mini frpc 已配置并连接成功
- 映射端口: 8765 (后端 API)

## 部署步骤

### 本地启动后端（Mac mini）
```bash
cd /Users/murphy/Desktop/Ski_Avatar_Pro
./start_backend.sh
```

### 部署前端到香港服务器
```bash
cd "/Users/murphy/Library/Mobile Documents/com~apple~CloudDocs/Projects/滑雪项目/网页前端设计"
./deploy.sh
```

## 访问地址
- 生产环境: http://103.251.89.147:3000
- 本地开发: http://localhost:3000

## 工作流程
1. 用户在网页上传视频
2. 前端发送到香港服务器 `/api/analyze`
3. 香港服务器转发到 `103.251.89.147:8765`
4. frp 隧道转发到 Mac mini `localhost:8765`
5. Mac mini 执行分析，实时推送进度
6. 结果返回给用户
