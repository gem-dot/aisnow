from __future__ import annotations

import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
WEBAPP_ROOT = Path(__file__).resolve().parent
WEB_DATA_DIR = PROJ_ROOT / 'web_data'
UPLOAD_DIR = WEB_DATA_DIR / 'uploads'
INCOMING_DIR = WEB_DATA_DIR / 'incoming'
TASK_DIR = WEB_DATA_DIR / 'tasks'
RUNTIME_DIR = WEB_DATA_DIR / 'runtime'
PUBLIC_URL_FILE = RUNTIME_DIR / 'public_base_url.txt'
WEB_PID_FILE = RUNTIME_DIR / 'web_server.pid'
TUNNEL_PID_FILE = RUNTIME_DIR / 'tunnel.pid'
WEB_LOG_FILE = RUNTIME_DIR / 'web_server.log'
TUNNEL_LOG_FILE = RUNTIME_DIR / 'tunnel.log'
OUTPUT_DIR = PROJ_ROOT / 'output'
CACHE_DIR = OUTPUT_DIR / '.cache'
STATIC_DIR = WEBAPP_ROOT / 'static'
TEMPLATE_DIR = WEBAPP_ROOT / 'templates'
RUN_PIPELINE_SCRIPT = PROJ_ROOT / 'run_ski_pipeline.py'
QA_SCRIPT = PROJ_ROOT / 'qa_review.py'


def _resolve_python_bin() -> Path:
    override = os.getenv('WEBAPP_PYTHON_BIN', '').strip()
    candidates = [
        Path(override).expanduser() if override else None,
        Path('/Users/bill_chlt/miniconda3/envs/snowboard/bin/python'),
        Path('/Users/murphy/miniconda3/envs/ski_avatar/bin/python'),
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return Path(sys.executable)


PYTHON_BIN = _resolve_python_bin()
HOST = os.getenv('WEBAPP_HOST', '0.0.0.0').strip() or '0.0.0.0'
PORT = int(os.getenv('WEBAPP_PORT', '8765'))
PUBLIC_BASE_URL = os.getenv('WEB_PUBLIC_BASE_URL', '').strip().rstrip('/')
MAX_CONCURRENT_JOBS = 1
LOG_TAIL_LIMIT = 60
CHUNK_UPLOAD_SIZE = int(os.getenv('WEBAPP_CHUNK_UPLOAD_SIZE_MB', '4')) * 1024 * 1024
STAGE_WEIGHTS = {
    'tracking': (0.0, 32.0),
    'pose': (32.0, 92.0),
    'render': (92.0, 98.0),
    'qa': (98.0, 100.0),
}

def get_public_base_url() -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    try:
        value = PUBLIC_URL_FILE.read_text().strip().rstrip('/')
    except Exception:
        return ''
    return value


def write_public_base_url(url: str) -> None:
    url = (url or '').strip().rstrip('/')
    if url:
        PUBLIC_URL_FILE.write_text(url)
    elif PUBLIC_URL_FILE.exists():
        PUBLIC_URL_FILE.unlink()


for path in (WEB_DATA_DIR, UPLOAD_DIR, INCOMING_DIR, TASK_DIR, RUNTIME_DIR, OUTPUT_DIR, CACHE_DIR, STATIC_DIR, TEMPLATE_DIR):
    path.mkdir(parents=True, exist_ok=True)
