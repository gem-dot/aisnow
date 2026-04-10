# AI4Snow - 滑雪视频分析引擎

基于深度学习的高精度滑雪动作分析系统，支持 2D 骨架追踪和压力曲线分析。

## 核心技术

**检测与追踪流程：**
1. `YOLOv8x + ByteTrack` - 人物检测与追踪
2. `RTMPose-X` - 17 点关键点姿态估计
3. 时序修正 - 离线修正、EMA 融合、异常值处理
4. 自定义渲染 - 优化双腿重合场景显示

**为什么更稳定：**
- 先检测再裁剪，适合远景滑雪场景
- 追踪框提供时间连续性
- Anti-lag 修正抑制关键点拖尾
- Overlap-aware 渲染处理双腿重合

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 处理视频

```bash
# 基础用法
python run_ski_pipeline.py /path/to/your/video.mp4

# 指定任务名
python run_ski_pipeline.py /path/to/video.mp4 --name my_session

# 强制重新计算
python run_ski_pipeline.py /path/to/video.mp4 --name my_session --force
```

### 3. 查看结果

输出文件位于 `output/<任务名>/`：
- `<任务名>_overlay.mp4` - 骨架叠加视频
- `<任务名>_side_by_side.mp4` - 对比视频
- `<任务名>_pressure_sync.mp4` - 压力曲线同步视频
- `review/` - 分析报告和数据

## Web 界面

### 启动本地服务

```bash
python backend_api.py
```

访问 `http://localhost:8765` 使用 Web 界面：
- 拖拽上传视频
- 实时查看处理进度
- 在线预览和下载结果

