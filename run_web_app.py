#!/usr/bin/env python3
from __future__ import annotations

import os
import socket
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

from webface import create_app
from webface.config import HOST, PORT, WEB_PID_FILE, get_public_base_url


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def guess_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        sock.close()


def main() -> int:
    app = create_app()
    current_pid = os.getpid()
    WEB_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEB_PID_FILE.write_text(str(current_pid))
    try:
        with make_server(HOST, PORT, app, server_class=ThreadingWSGIServer) as server:
            lan_ip = guess_lan_ip()
            print(f'[Web] Listening on http://127.0.0.1:{PORT}')
            print(f'[Web] LAN access   http://{lan_ip}:{PORT}')
            public_url = get_public_base_url()
            if public_url:
                print(f'[Web] Public URL   {public_url}')
            else:
                print('[Web] Public URL   未配置；如需公网访问可设置 WEB_PUBLIC_BASE_URL')
            print('[Web] 当前默认串行处理上传任务，避免把机器压爆。')
            print('[Web] 当前支持 JSBA / SAJ / 进阶 三模式，可供网页与小程序切换。')
            server.serve_forever()
    finally:
        try:
            if WEB_PID_FILE.exists() and WEB_PID_FILE.read_text().strip() == str(current_pid):
                WEB_PID_FILE.unlink()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
