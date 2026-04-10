#!/usr/bin/env python3
"""
滑雪项目后端 API 服务
运行在 Mac mini 本地，通过 frp 暴露给香港前端
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import subprocess
import time
import json
from pathlib import Path
import threading

app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 配置路径
SKI_PROJECT = Path(__file__).parent
USER_ROOT = Path("/Volumes/extra/ski/users")
PYTHON_BIN = "/Users/murphy/miniconda3/envs/mujoco/bin/python"
MAX_VIDEO_DURATION = 40  # 最大视频时长（秒）

# 任务存储
tasks = {}

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "service": "ski-backend"})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """接收视频分析请求"""
    try:
        file = request.files.get('file')
        username = request.form.get('username')
        mode = request.form.get('mode', 'jsba')

        if not file or not username:
            return jsonify({"success": False, "error": "缺少参数"}), 400

        # 创建用户目录
        user_input = USER_ROOT / username / "input"
        user_output = USER_ROOT / username / "output"
        user_input.mkdir(parents=True, exist_ok=True)
        user_output.mkdir(parents=True, exist_ok=True)

        # 保存文件
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{file.filename}"
        filepath = user_input / filename
        file.save(filepath)

        file_size = filepath.stat().st_size
        print(f"[API] 文件已保存: {filepath}, 大小: {file_size / 1024 / 1024:.2f} MB")

        if file_size == 0:
            return jsonify({"success": False, "error": "上传的文件为空"}), 400

        # 检测视频时长
        duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{filepath}"'
        result = subprocess.run(duration_cmd, shell=True, capture_output=True, text=True)

        try:
            duration = float(result.stdout.strip())
            print(f"[API] 视频时长: {duration:.2f} 秒")

            # 如果超过40秒，截取前40秒
            if duration > MAX_VIDEO_DURATION:
                print(f"[API] 视频超过 {MAX_VIDEO_DURATION} 秒，正在截取...")
                trimmed_path = user_input / f"{timestamp}_trimmed_{file.filename}"
                # 使用 -ss 在输入前面，快速定位
                trim_cmd = f'ffmpeg -ss 0 -i "{filepath}" -t {MAX_VIDEO_DURATION} -c:v libx264 -preset ultrafast -crf 23 -c:a aac "{trimmed_path}" -y'
                subprocess.run(trim_cmd, shell=True, check=True)
                filepath.unlink()  # 删除原文件
                filepath = trimmed_path
                print(f"[API] 已截取为 {MAX_VIDEO_DURATION} 秒")
        except:
            print("[API] 无法检测视频时长，继续处理")

        task_id = f"task_{timestamp}"
        output_name = filepath.stem

        # 初始化任务
        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "stage": "准备分析",
            "username": username,
            "outputName": output_name,
            "logs": []
        }

        # 异步启动分析
        thread = threading.Thread(
            target=run_analysis,
            args=(task_id, str(filepath), output_name, mode, str(user_output))
        )
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "taskId": task_id})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def run_analysis(task_id, filepath, output_name, mode, user_output):
    """执行分析任务"""
    task = tasks[task_id]

    cmd = [
        PYTHON_BIN,
        "-u",  # 无缓冲输出
        str(SKI_PROJECT / "run_ski_pipeline.py"),
        filepath,
        "--name", output_name,
        "--output-root", user_output,
        "--pipeline-id", mode
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            text=True,
            bufsize=1,  # 行缓冲
            cwd=str(SKI_PROJECT)
        )

        # 实时读取输出并推送进度
        for line in process.stdout:
            task["logs"].append(line.strip())

            # 解析进度
            if "ByteTrack:" in line and "%" in line:
                percent = int(line.split("%")[0].split()[-1])
                task["progress"] = int(percent * 0.3)
                task["stage"] = "ByteTrack 人物追踪"
                socketio.emit('progress', {"taskId": task_id, "task": task})

            elif "RTMPose-X:" in line and "%" in line:
                percent = int(line.split("%")[0].split()[-1])
                task["progress"] = 30 + int(percent * 0.4)
                task["stage"] = "RTMPose-X 关键点检测"
                socketio.emit('progress', {"taskId": task_id, "task": task})

            elif "Render Overlay:" in line and "%" in line:
                percent = int(line.split("%")[0].split()[-1])
                task["progress"] = 70 + int(percent * 0.3)
                task["stage"] = "渲染分析视频"
                socketio.emit('progress', {"taskId": task_id, "task": task})

        process.wait()

        if process.returncode == 0:
            task["status"] = "completed"
            task["progress"] = 100
            task["stage"] = "分析完成"
            task["results"] = {
                "overlayVideo": f"{output_name}_overlay.mp4",
                "sideBySideVideo": f"{output_name}_side_by_side.mp4",
                "pressureVideo": f"{output_name}_pressure_sync.mp4"
            }
        else:
            task["status"] = "failed"
            task["error"] = f"分析失败 (退出码: {process.returncode})"

        socketio.emit('progress', {"taskId": task_id, "task": task})

    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        socketio.emit('progress', {"taskId": task_id, "task": task})

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """查询任务状态"""
    if task_id not in tasks:
        return jsonify({"success": False, "error": "任务不存在"}), 404
    return jsonify({"success": True, "task": tasks[task_id]})

@socketio.on('connect')
def handle_connect():
    print(f"客户端已连接: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"客户端已断开: {request.sid}")

if __name__ == '__main__':
    print("🚀 滑雪分析后端启动中...")
    print(f"📁 项目路径: {SKI_PROJECT}")
    print(f"👤 用户根目录: {USER_ROOT}")
    socketio.run(app, host='0.0.0.0', port=8765, debug=False, allow_unsafe_werkzeug=True)
