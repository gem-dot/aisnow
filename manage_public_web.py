#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from webface.config import (
    PORT,
    PROJ_ROOT,
    PUBLIC_URL_FILE,
    PYTHON_BIN,
    TUNNEL_LOG_FILE,
    TUNNEL_PID_FILE,
    WEB_LOG_FILE,
    WEB_PID_FILE,
    get_public_base_url,
    write_public_base_url,
)

LOCAL_HEALTH_URL = f'http://127.0.0.1:{PORT}/api/health'
QUICK_TUNNEL_RE = re.compile(r'https://[-a-zA-Z0-9.]+trycloudflare\.com')


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def write_pid(path: Path, pid: int) -> None:
    path.write_text(str(pid))


def remove_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def resolve_cloudflared(explicit: str | None) -> str | None:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend([
        os.getenv('CLOUDFLARED_BIN', '').strip(),
        str(Path.home() / '.local/bin/cloudflared'),
        '/opt/homebrew/bin/cloudflared',
        '/usr/local/bin/cloudflared',
        'cloudflared',
    ])
    for candidate in candidates:
        if not candidate:
            continue
        if '/' in candidate:
            if Path(candidate).exists():
                return candidate
        else:
            from shutil import which
            resolved = which(candidate)
            if resolved:
                return resolved
    return None


def wait_for_local_server(timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(LOCAL_HEALTH_URL, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except URLError:
            time.sleep(0.6)
        except Exception:
            time.sleep(0.6)
    return False


def start_process(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open('a', encoding='utf-8')
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJ_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    handle.close()
    return process.pid


def terminate_pid(pid: int | None, label: str) -> None:
    if not pid or not pid_running(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if not pid_running(pid):
            return
        time.sleep(0.2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def ensure_web_running() -> int:
    existing_pid = read_pid(WEB_PID_FILE)
    if existing_pid and pid_running(existing_pid):
        return existing_pid
    cmd = [str(PYTHON_BIN), str(PROJ_ROOT / 'run_web_app.py')]
    pid = start_process(cmd, WEB_LOG_FILE, env=os.environ.copy())
    write_pid(WEB_PID_FILE, pid)
    if not wait_for_local_server():
        terminate_pid(pid, 'web')
        remove_file(WEB_PID_FILE)
        raise RuntimeError('本地 Web 服务启动失败，请查看日志')
    return pid


def wait_for_quick_tunnel_url(timeout: float = 45.0) -> str:
    deadline = time.time() + timeout
    last_size = 0
    while time.time() < deadline:
        if TUNNEL_LOG_FILE.exists():
            text = TUNNEL_LOG_FILE.read_text(errors='replace')
            if len(text) != last_size:
                last_size = len(text)
            match = QUICK_TUNNEL_RE.search(text)
            if match:
                return match.group(0)
        time.sleep(0.5)
    raise RuntimeError('未在 cloudflared 日志中拿到公网 URL')


def start_tunnel(mode: str, cloudflared_bin: str, tunnel_token: str = '', public_url: str = '') -> tuple[int, str]:
    existing_pid = read_pid(TUNNEL_PID_FILE)
    if existing_pid and pid_running(existing_pid):
        return existing_pid, get_public_base_url()

    remove_file(TUNNEL_LOG_FILE)
    remove_file(PUBLIC_URL_FILE)
    local_url = f'http://127.0.0.1:{PORT}'
    if mode == 'named':
        if not tunnel_token:
            raise RuntimeError('named 模式缺少 tunnel token')
        cmd = [cloudflared_bin, 'tunnel', '--no-autoupdate', 'run', '--token', tunnel_token]
        pid = start_process(cmd, TUNNEL_LOG_FILE, env=os.environ.copy())
        write_pid(TUNNEL_PID_FILE, pid)
        if public_url:
            write_public_base_url(public_url)
        return pid, public_url

    cmd = [cloudflared_bin, 'tunnel', '--no-autoupdate', '--url', local_url]
    pid = start_process(cmd, TUNNEL_LOG_FILE, env=os.environ.copy())
    write_pid(TUNNEL_PID_FILE, pid)
    public_url = wait_for_quick_tunnel_url()
    write_public_base_url(public_url)
    return pid, public_url


def stop_stack() -> None:
    terminate_pid(read_pid(TUNNEL_PID_FILE), 'tunnel')
    terminate_pid(read_pid(WEB_PID_FILE), 'web')
    remove_file(TUNNEL_PID_FILE)
    remove_file(WEB_PID_FILE)
    write_public_base_url('')


def status() -> int:
    web_pid = read_pid(WEB_PID_FILE)
    tunnel_pid = read_pid(TUNNEL_PID_FILE)
    print(f'web_pid={web_pid} running={pid_running(web_pid or -1)}')
    print(f'tunnel_pid={tunnel_pid} running={pid_running(tunnel_pid or -1)}')
    print(f'local_url=http://127.0.0.1:{PORT}')
    print(f'public_url={get_public_base_url() or ""}')
    print(f'web_log={WEB_LOG_FILE}')
    print(f'tunnel_log={TUNNEL_LOG_FILE}')
    return 0


def start(args: argparse.Namespace) -> int:
    web_pid = ensure_web_running()
    print(f'[public-web] 本地 Web 已就绪，PID={web_pid}，地址 http://127.0.0.1:{PORT}')
    if args.mode == 'none':
        return 0

    cloudflared_bin = resolve_cloudflared(args.cloudflared)
    if not cloudflared_bin:
        raise RuntimeError('未找到 cloudflared，请先安装或通过 --cloudflared 指定路径')

    tunnel_token = args.tunnel_token or os.getenv('CLOUDFLARE_TUNNEL_TOKEN', '').strip()
    public_url = args.public_url or os.getenv('WEB_PUBLIC_BASE_URL', '').strip().rstrip('/')
    mode = args.mode
    if mode == 'auto':
        mode = 'named' if tunnel_token else 'quick'

    tunnel_pid, resolved_public_url = start_tunnel(mode, cloudflared_bin, tunnel_token=tunnel_token, public_url=public_url)
    print(f'[public-web] 隧道已启动，PID={tunnel_pid}')
    if resolved_public_url:
        print(f'[public-web] 公网地址 {resolved_public_url}')
    else:
        print('[public-web] 隧道已启动，但当前未拿到固定公网地址；如用 named tunnel，请设置 WEB_PUBLIC_BASE_URL')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='管理 Ski Avatar Pro 公网 Web 栈')
    subparsers = parser.add_subparsers(dest='command', required=True)

    start_parser = subparsers.add_parser('start')
    start_parser.add_argument('--mode', choices=('auto', 'quick', 'named', 'none'), default='auto')
    start_parser.add_argument('--cloudflared', type=str, default='')
    start_parser.add_argument('--tunnel-token', type=str, default='')
    start_parser.add_argument('--public-url', type=str, default='')

    subparsers.add_parser('stop')
    subparsers.add_parser('status')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == 'start':
        return start(args)
    if args.command == 'stop':
        stop_stack()
        return 0
    return status()


if __name__ == '__main__':
    raise SystemExit(main())
