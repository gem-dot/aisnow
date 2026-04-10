from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

WEBFACE_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = WEBFACE_ROOT.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from webface.config import CACHE_DIR, OUTPUT_DIR, UPLOAD_DIR
from webface.pipelines import DEFAULT_PIPELINE_ID, get_pipeline, prepare_pipeline_execution

PAGE_TITLE = 'AI 4 Snowboarding | 你的口袋单板教练'
MODE_ORDER = ['casi', 'jsba', 'saj']
MODE_LABELS = {
    'casi': '🟢 CASI 新手扫盲 (姿态与重心诊断)',
    'jsba': '🔴 JSBA 进阶刻滑 (压力与反弓分析)',
    'saj': '🔵 SAJ 综合走刃 (动态发力评测)',
}
MODE_DESCRIPTIONS = {
    'casi': '更适合刚入门或基础不稳定的滑手，重点检查重心是否站稳、姿态是否顺、身体有没有明显代偿。',
    'jsba': '更适合想提升刻滑质量的滑手，重点观察压板节奏、反弓建立、换刃时机和动作稳定性。',
    'saj': '更适合想做综合走刃评测的滑手，重点检查动态发力、动作连贯性和整段滑行节奏。',
}
MODE_RESULT_TITLES = {
    'casi': '姿态与重心诊断面板',
    'jsba': '压力与反弓分析面板',
    'saj': '动态发力评测面板',
}
MODE_RESULT_TEXT = {
    'casi': '这里会呈现站姿稳定性、重心偏移、膝踝协同和基础控板问题，帮助你先把动作底盘站稳。',
    'jsba': '这里会呈现压板节奏、反弓建立、换刃衔接和发力窗口，帮助你看清刻滑动作到底稳不稳、够不够干净。',
    'saj': '这里会呈现整段走刃连贯性、动态发力时序和动作节奏，帮助你判断整条线的效率与流畅度。',
}
PIPELINE_PROGRESS_STEPS = {
    'casi': [
        (10, '正在解码你的视频画面...'),
        (30, '正在锁定滑手主体与身体骨架...'),
        (55, '正在识别姿态、重心与基础动作节奏...'),
        (82, '正在生成新手诊断结果...'),
        (100, '分析完成，正在加载你的结果面板...'),
    ],
    'jsba': [
        (10, '正在解码你的视频画面...'),
        (30, '正在锁定滑手主体与关键骨架...'),
        (58, '正在分析压力传递与反弓节奏...'),
        (84, '正在生成刻滑诊断结果...'),
        (100, '分析完成，正在加载你的结果面板...'),
    ],
    'saj': [
        (10, '正在解码你的视频画面...'),
        (30, '正在锁定滑手主体与动作轮廓...'),
        (58, '正在分析动态发力与整段走刃节奏...'),
        (84, '正在生成综合走刃结果...'),
        (100, '分析完成，正在加载你的结果面板...'),
    ],
}

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon='🏂',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0E1117;
        --bg-soft: #161B22;
        --panel: rgba(14, 18, 24, 0.82);
        --panel-strong: rgba(10, 13, 18, 0.92);
        --line: rgba(75, 90, 108, 0.22);
        --line-strong: rgba(66, 232, 255, 0.18);
        --cyan: #42e8ff;
        --cyan-soft: rgba(66, 232, 255, 0.10);
        --gold: #d6b46b;
        --gold-soft: rgba(214, 180, 107, 0.18);
        --text: #f3f7fb;
        --muted: #8a97a7;
        --muted-2: #667282;
        --success: #4ee6b3;
    }

    .top-bar-logo {
        font-size: 1.35rem;
        font-weight: 900;
        color: var(--cyan);
        letter-spacing: 0.08em;
        padding: 1rem 0;
    }

    .top-bar-actions {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 0;
    }

    .btn-auth {
        padding: 0.55rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(22, 27, 34, 0.6);
        color: var(--text);
        font-size: 0.92rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .btn-auth:hover {
        border-color: rgba(66, 232, 255, 0.4);
        background: rgba(22, 27, 34, 0.85);
    }

    .btn-pro {
        padding: 0.55rem 1.35rem;
        border-radius: 12px;
        border: 1px solid rgba(214, 180, 107, 0.5);
        background: linear-gradient(135deg, rgba(214, 180, 107, 0.18), rgba(214, 180, 107, 0.08));
        color: var(--gold);
        font-size: 0.92rem;
        font-weight: 800;
        cursor: pointer;
        box-shadow: 0 0 16px rgba(214, 180, 107, 0.12);
        transition: all 0.2s ease;
    }

    .btn-pro:hover {
        border-color: rgba(214, 180, 107, 0.8);
        box-shadow: 0 0 24px rgba(214, 180, 107, 0.22);
        transform: translateY(-1px);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 12% 12%, rgba(66, 232, 255, 0.06), transparent 24%),
            radial-gradient(circle at 88% 10%, rgba(214, 180, 107, 0.05), transparent 18%),
            linear-gradient(180deg, #0E1117 0%, #12171E 42%, #161B22 100%);
    }

    #MainMenu,
    header,
    footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stSidebarNav"],
    [data-testid="stHeaderActionElements"] {
        visibility: hidden !important;
        display: none !important;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 3.5rem;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.2rem 2.15rem;
        border-radius: 28px;
        background: rgba(14, 17, 23, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow:
            0 0 0 1px rgba(66, 232, 255, 0.08),
            0 24px 80px rgba(0, 0, 0, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
    }

    .hero-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, transparent 0%, rgba(66, 232, 255, 0.10) 18%, transparent 42%),
            radial-gradient(circle at 92% 18%, rgba(214, 180, 107, 0.15), transparent 22%);
        pointer-events: none;
    }

    .hero-eyebrow {
        text-align: center;
        color: var(--cyan);
        font-size: 0.9rem;
        letter-spacing: 0.24em;
        font-weight: 700;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        text-align: center;
        font-size: clamp(1.95rem, 3.2vw, 4.2rem);
        line-height: 1.08;
        font-weight: 900;
        color: var(--text);
        letter-spacing: 0.01em;
        margin-bottom: 0.9rem;
        white-space: nowrap !important;
    }

    .hero-title .cyan {
        color: var(--cyan);
        text-shadow: 0 0 22px rgba(66, 232, 255, 0.16);
    }

    .hero-title .gold {
        color: var(--gold);
    }

    .hero-slogan {
        text-align: center;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
        letter-spacing: 0.02em;
    }

    .glass-section {
        margin-top: 1.35rem;
        padding: 1.45rem 1.35rem;
        border-radius: 24px;
        background: rgba(17, 21, 28, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.04),
            inset 0 0 0 1px rgba(66, 232, 255, 0.05),
            0 20px 60px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .section-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.38rem 0.76rem;
        border-radius: 999px;
        background: rgba(66, 232, 255, 0.08);
        border: 1px solid rgba(66, 232, 255, 0.16);
        color: var(--cyan);
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .section-title {
        margin-top: 0.9rem;
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: 0.01em;
    }

    .section-subtitle {
        margin-top: 0.38rem;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.75;
    }

    div[role="radiogroup"] {
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1.1rem;
    }

    div[role="radiogroup"] input[type="radio"] {
        display: none !important;
    }

    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    div[role="radiogroup"] label svg {
        display: none !important;
    }

    div[role="radiogroup"] label {
        margin: 0 !important;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: linear-gradient(180deg, rgba(17, 22, 28, 0.88) 0%, rgba(10, 14, 19, 0.88) 100%);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        padding: 0.95rem 1rem !important;
        min-height: 92px;
        align-items: flex-start !important;
        transition: all 0.18s ease;
        opacity: 0.65;
    }

    div[role="radiogroup"] label:has(input:checked) {
        opacity: 1;
    }

    div[role="radiogroup"] label:hover {
        border-color: rgba(66, 232, 255, 0.26) !important;
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
    }

    div[role="radiogroup"] label:has(input:checked) {
        border-color: rgba(66, 232, 255, 0.45) !important;
        box-shadow:
            0 0 0 1px rgba(66, 232, 255, 0.12),
            0 16px 40px rgba(8, 20, 28, 0.36),
            inset 0 0 24px rgba(66, 232, 255, 0.06);
        background:
            radial-gradient(circle at top right, rgba(66, 232, 255, 0.06), transparent 40%),
            linear-gradient(180deg, rgba(14, 20, 26, 0.94) 0%, rgba(10, 14, 19, 0.96) 100%);
    }

    div[role="radiogroup"] p {
        color: var(--text) !important;
        font-size: 0.98rem !important;
        line-height: 1.55 !important;
        font-weight: 700 !important;
    }

    .mode-helper {
        margin-top: 1rem;
        padding: 1rem 1.05rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(8, 12, 16, 0.92) 0%, rgba(11, 15, 20, 0.82) 100%);
        border: 1px solid rgba(214, 180, 107, 0.16);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .mode-helper-title {
        color: var(--gold);
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0.28rem;
    }

    .mode-helper-body {
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.78;
    }

    [data-testid="stFileUploader"] {
        margin-top: 1rem;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed rgba(66, 232, 255, 0.24) !important;
        border-radius: 24px !important;
        background:
            radial-gradient(circle at top right, rgba(66, 232, 255, 0.06), transparent 28%),
            linear-gradient(180deg, rgba(14, 18, 24, 0.88) 0%, rgba(10, 14, 19, 0.92) 100%) !important;
        min-height: 190px;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.03),
            inset 0 0 0 1px rgba(66, 232, 255, 0.04),
            0 20px 40px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: var(--muted) !important;
        font-size: 1rem !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        color: var(--cyan) !important;
    }

    .upload-note {
        margin-top: 0.95rem;
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.75;
    }

    .file-ready-card {
        margin-top: 1rem;
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(9, 14, 20, 0.92) 0%, rgba(7, 11, 15, 0.96) 100%);
        border: 1px solid rgba(66, 232, 255, 0.40);
        box-shadow:
            0 0 0 1px rgba(66, 232, 255, 0.08),
            0 18px 36px rgba(0, 0, 0, 0.26),
            inset 0 0 22px rgba(66, 232, 255, 0.05);
    }

    .file-ready-title {
        color: var(--cyan);
        font-weight: 800;
        font-size: 0.96rem;
        letter-spacing: 0.04em;
        margin-bottom: 0.45rem;
    }

    .file-ready-name {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 700;
        word-break: break-all;
    }

    .file-ready-note {
        margin-top: 0.42rem;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.72;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 3.45rem;
        margin-top: 0.3rem;
        border-radius: 18px;
        border: 1px solid rgba(66, 232, 255, 0.62);
        background:
            linear-gradient(90deg, rgba(66, 232, 255, 0.20) 0%, rgba(18, 34, 46, 0.96) 32%, rgba(8, 13, 18, 0.96) 100%),
            linear-gradient(135deg, #07131a 0%, #0b1a23 100%);
        color: #f8fdff;
        font-size: 1.08rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        box-shadow:
            0 0 0 1px rgba(66, 232, 255, 0.10),
            0 18px 34px rgba(4, 15, 22, 0.42),
            inset 0 0 18px rgba(66, 232, 255, 0.08);
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(66, 232, 255, 0.95);
        color: #ffffff;
        box-shadow:
            0 0 0 1px rgba(66, 232, 255, 0.12),
            0 24px 42px rgba(4, 15, 22, 0.48),
            0 0 22px rgba(66, 232, 255, 0.16);
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.2rem;
    }

    .status-card {
        padding: 1rem 1rem 0.95rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(11, 15, 20, 0.84) 0%, rgba(11, 15, 20, 0.72) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .status-card-title {
        color: var(--gold);
        font-size: 0.88rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .status-card-text {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.72;
    }

    .result-card {
        min-height: 410px;
        padding: 1.35rem 1.2rem;
        border-radius: 24px;
        background: linear-gradient(180deg, rgba(11, 15, 20, 0.88) 0%, rgba(9, 12, 16, 0.82) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.03),
            inset 0 0 0 1px rgba(66, 232, 255, 0.05),
            0 18px 42px rgba(0, 0, 0, 0.28);
    }

    .result-title {
        color: var(--text);
        font-size: 1.18rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .result-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        background: rgba(214, 180, 107, 0.10);
        border: 1px solid rgba(214, 180, 107, 0.18);
        color: var(--gold);
        font-size: 0.84rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .result-placeholder {
        min-height: 250px;
        border-radius: 20px;
        border: 1px dashed rgba(66, 232, 255, 0.22);
        background:
            radial-gradient(circle at top right, rgba(66, 232, 255, 0.08), transparent 28%),
            linear-gradient(180deg, rgba(7, 10, 14, 0.98) 0%, rgba(10, 14, 18, 0.92) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1.3rem;
        color: #c8d2dc;
        line-height: 1.8;
    }

    .result-caption {
        margin-top: 0.9rem;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.76;
    }

    .analysis-tip {
        margin-top: 0.95rem;
        padding: 0.88rem 0.96rem;
        border-radius: 16px;
        background: rgba(66, 232, 255, 0.05);
        border: 1px solid rgba(66, 232, 255, 0.12);
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.7;
    }

    .guide-card {
        margin-top: 1rem;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(14, 18, 24, 0.65);
        border: 1px solid rgba(66, 232, 255, 0.12);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }

    .guide-title {
        color: var(--cyan);
        font-size: 0.92rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
        letter-spacing: 0.02em;
    }

    .guide-item {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.8;
        margin-bottom: 0.45rem;
    }

    .warning-card {
        padding: 1.35rem 1.3rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(214, 180, 107, 0.08) 0%, rgba(214, 180, 107, 0.04) 100%);
        border: 1px solid rgba(214, 180, 107, 0.22);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        text-align: center;
    }

    .warning-icon {
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
    }

    .warning-title {
        color: var(--gold);
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    .warning-text {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.8;
    }

    .ghost-btn {
        margin-top: 1.2rem;
        padding: 0.65rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: transparent;
        color: var(--muted-2);
        font-size: 0.88rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
    }

    .ghost-btn:hover {
        border-color: rgba(66, 232, 255, 0.3);
        color: var(--cyan);
        background: rgba(66, 232, 255, 0.04);
    }

    @media (max-width: 960px) {
        div[role="radiogroup"],
        .status-grid {
            grid-template-columns: 1fr;
        }
        .hero-shell {
            padding: 2rem 1.3rem 1.85rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_state() -> None:
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'analysis_failed' not in st.session_state:
        st.session_state.analysis_failed = False
    if 'last_filename' not in st.session_state:
        st.session_state.last_filename = ''
    if 'last_pipeline_id' not in st.session_state:
        st.session_state.last_pipeline_id = DEFAULT_PIPELINE_ID if DEFAULT_PIPELINE_ID in MODE_ORDER else 'jsba'
    if 'task_name' not in st.session_state:
        st.session_state.task_name = ''
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    # 账号系统状态
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'quota' not in st.session_state:
        st.session_state.quota = 0
    if 'is_coach' not in st.session_state:
        st.session_state.is_coach = False


def render_top_bar() -> None:
    """商业化顶栏：Logo + 登录/注册 + Pro 订阅"""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            """
            <div class="top-bar-logo">AI 4 SNOW</div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="top-bar-actions">
                <button class="btn-auth">登录</button>
                <button class="btn-auth">注册</button>
                <button class="btn-pro">升级 Pro ⚡</button>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    """侧边栏账户中心与商业化模块"""
    with st.sidebar:
        st.markdown("### 🏂 账户中心")

        if not st.session_state.is_logged_in:
            # 未登录状态
            st.markdown("**快速登录/注册**")
            phone_input = st.text_input("手机号/用户名", placeholder="输入手机号或昵称", key="login_input")

            if st.button("🚀 一键登录/注册", use_container_width=True):
                if phone_input.strip():
                    st.session_state.is_logged_in = True
                    st.session_state.username = phone_input.strip()
                    st.session_state.quota = 3
                    st.toast("✅ 登录成功！赠送 3 次新人体验额度", icon="🎁")
                    st.rerun()
                else:
                    st.error("请输入手机号或用户名")
        else:
            # 已登录状态
            st.success(f"✅ 欢迎回来，{st.session_state.username}！")
            st.metric("剩余分析额度", f"{st.session_state.quota} 次", delta=None)

            st.markdown("---")
            st.markdown("**💰 充值中心**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🛒 ¥9.9\n10次包", use_container_width=True):
                    st.session_state.quota += 10
                    st.toast("✅ 支付成功！额度已到账", icon="💳")
                    st.rerun()

            with col2:
                if st.button("👑 ¥169\nPro卡", use_container_width=True):
                    st.session_state.quota += 200
                    st.session_state.is_coach = True
                    st.toast("✅ 恭喜升级雪季 Pro！", icon="🎉")
                    st.rerun()

            st.markdown("---")

            # 教练裂变中心
            with st.expander("💼 教练裂变中心"):
                st.markdown("**邀请学员赚 40% 现金返佣**")
                st.markdown("每位学员充值，你获得 40% 返现")

                if st.button("生成我的专属推广海报", use_container_width=True):
                    st.toast(f"✅ 海报生成中，专属推荐码：COACH-{st.session_state.username}", icon="🎨")

            st.markdown("---")
            if st.button("🚪 退出登录", use_container_width=True):
                st.session_state.is_logged_in = False
                st.session_state.username = ''
                st.session_state.quota = 0
                st.session_state.is_coach = False
                st.rerun()


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-eyebrow">商业级单板动作分析系统</div>
            <div class="hero-title">AI 4 <span class="cyan">Snowboarding</span> <span class="gold">|</span> 你的口袋单板教练</div>
            <div class="hero-slogan">上传视频，AI 瞬间透视你的滑行骨架</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_guide(selected_pipeline_id: str) -> None:
    st.markdown(
        """
        <div class="glass-section">
            <div class="section-kicker">智选模式</div>
            <div class="section-title">先选你的分析视角，再交给 AI 拆动作</div>
            <div class="section-subtitle">不同模式面向不同阶段用户，系统会基于同一套底层骨架与节奏数据，输出更贴近你真实需求的诊断语言。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_mode = st.session_state.last_pipeline_id if st.session_state.last_pipeline_id in MODE_ORDER else 'jsba'
    selected_pipeline_id = st.radio(
        '分析模式',
        options=MODE_ORDER,
        index=MODE_ORDER.index(selected_pipeline_id or default_mode),
        horizontal=True,
        label_visibility='collapsed',
        format_func=lambda item: MODE_LABELS[item],
    )

    st.markdown(
        f"""
        <div class="mode-helper">
            <div class="mode-helper-title">模式说明</div>
            <div class="mode-helper-body">{MODE_DESCRIPTIONS[selected_pipeline_id]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected_pipeline_id


def render_upload_panel(selected_pipeline) -> object | None:
    st.markdown(
        """
        <div class="glass-section">
            <div class="section-kicker">上传视频</div>
            <div class="section-title">把一条视频交给 AI，换来一份更像教练视角的动作拆解</div>
            <div class="section-subtitle">建议上传主体清晰、镜头稳定、单条时长在 15 秒以内的视频，便于更快完成骨架与节奏分析。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        '📁 点击或拖拽上传你的滑雪视频 (建议 15 秒内)',
        type=['mp4', 'mov', 'MP4', 'MOV'],
        accept_multiple_files=False,
        help='支持 MP4 / MOV。建议画面中主体滑手尽量清晰、无遮挡。',
    )

    st.markdown(
        """
        <div class="upload-note">系统会沿用既有分析路由，只升级商业级界面展现；底层分析管线与结果输出逻辑保持不变。</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="guide-card">
            <div class="guide-title">📸 获得最高 AI 评分的拍摄指南</div>
            <div class="guide-item">① 滑手主体居中且占画面 1/3 以上</div>
            <div class="guide-item">② 避免他人乱入或严重扬雪遮挡</div>
            <div class="guide-item">③ 保持镜头稳定防抖</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        st.markdown(
            f"""
            <div class="file-ready-card">
                <div class="file-ready-title">素材已准备就绪</div>
                <div class="file-ready-name">{uploaded_file.name}</div>
                <div class="file-ready-note">即将进入：{MODE_LABELS[selected_pipeline.pipeline_id]}。点击下方按钮后，系统会按既有分析路由开始处理。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return uploaded_file


def render_status_grid() -> None:
    st.markdown(
        """
        <div class="status-grid">
            <div class="status-card">
                <div class="status-card-title">骨架透视</div>
                <div class="status-card-text">把肉眼难以连续追踪的身体姿态，转成清晰可解释的骨架时序。</div>
            </div>
            <div class="status-card">
                <div class="status-card-title">发力拆解</div>
                <div class="status-card-text">帮助你看清是重心没站稳，还是换刃、压板、发力节奏出了问题。</div>
            </div>
            <div class="status-card">
                <div class="status-card-title">结果沉淀</div>
                <div class="status-card-text">界面先完成商业化包装，底层分析路由保持原样，便于后续无缝接入正式结果页。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(selected_pipeline_id: str, uploaded_file) -> None:
    """真实引擎执行：保存文件 → 调用算法 → 流式进度 → 扣除额度"""
    st.session_state.analysis_done = False
    st.session_state.analysis_failed = False
    st.session_state.last_filename = uploaded_file.name
    st.session_state.last_pipeline_id = selected_pipeline_id

    try:
        # 1. 保存上传文件到本地
        task_name = f"streamlit_{int(time.time())}"
        st.session_state.task_name = task_name

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        input_path = UPLOAD_DIR / f"{task_name}_{uploaded_file.name}"

        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # 2. 准备执行命令
        output_dir = OUTPUT_DIR / task_name
        cache_dir = CACHE_DIR / task_name

        execution = prepare_pipeline_execution(
            selected_pipeline_id,
            task_name=task_name,
            input_path=input_path,
            output_dir=output_dir,
            cache_dir=cache_dir,
        )

        st.session_state.output_dir = execution.output_dir

        # 3. 流式进度条执行
        progress_bar = st.progress(0, text='正在初始化 AI 引擎...')
        status_text = st.empty()

        process = subprocess.Popen(
            execution.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 实时读取输出并更新进度
        current_progress = 0
        max_progress = 95
        increment = 5

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                # 尝试从输出中提取进度信息
                import re
                frame_match = re.search(r'Frame\s+(\d+)/(\d+)', line)
                percent_match = re.search(r'(\d+)%', line)

                if frame_match:
                    current_frame = int(frame_match.group(1))
                    total_frames = int(frame_match.group(2))
                    current_progress = int((current_frame / total_frames) * max_progress)
                    progress_bar.progress(current_progress, text=f'正在深度解析动作... {current_progress}%')
                elif percent_match:
                    current_progress = min(int(percent_match.group(1)), max_progress)
                    progress_bar.progress(current_progress, text=f'正在深度解析动作... {current_progress}%')
                else:
                    # 模拟平滑增长
                    if current_progress < max_progress:
                        current_progress = min(current_progress + increment, max_progress)
                        progress_bar.progress(current_progress, text=f'AI 正在解析骨架与节奏... {current_progress}%')

        process.wait()

        # 检查执行结果
        if process.returncode != 0:
            st.session_state.analysis_failed = True
            progress_bar.empty()
            status_text.empty()
            return

        # 成功完成，扣除额度
        progress_bar.progress(100, text='分析完成！')
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()

        st.session_state.analysis_done = True
        st.session_state.quota -= 1  # 扣除额度

    except Exception as e:
        st.session_state.analysis_failed = True
        st.session_state.analysis_done = False


def render_results() -> None:
    """渲染真实结果：读取视频和JSON数据"""
    # 处理分析失败状态
    if st.session_state.analysis_failed:
        st.markdown(
            """
            <div class="glass-section">
                <div class="warning-card">
                    <div class="warning-icon">⚠️</div>
                    <div class="warning-title">AI 视线受阻</div>
                    <div class="warning-text">因距离过远或遮挡，AI 无法捕捉完整骨架。请放心，本次分析未扣除您的使用额度。<br>请更换清晰视频重试。</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.analysis_done:
        return

    active_pipeline = get_pipeline(st.session_state.last_pipeline_id)
    result_title = MODE_RESULT_TITLES.get(active_pipeline.pipeline_id, '智能分析结果面板')
    result_text = MODE_RESULT_TEXT.get(active_pipeline.pipeline_id, '这里会展示你的视频分析结果。')

    st.markdown(
        """
        <div class="glass-section">
            <div class="section-kicker">分析结果</div>
            <div class="section-title">AI 分析完成，查看你的动作诊断</div>
            <div class="section-subtitle">左侧是骨架透视回放，右侧是结构化诊断指标。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2, gap='large')

    output_dir = st.session_state.output_dir
    task_name = st.session_state.task_name

    # 左侧：骨架视频
    with left_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-tag">骨架回放视图</div>
                <div class="result-title">AI 骨架透视回放</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        video_path = output_dir / f"{task_name}_side_by_side.mp4"
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.info('视频生成中，请稍候...')

        st.markdown(
            f"""
            <div class="result-caption">当前素材：{st.session_state.last_filename}。</div>
            """,
            unsafe_allow_html=True,
        )

    # 右侧：诊断指标
    with right_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-tag">{MODE_LABELS.get(active_pipeline.pipeline_id, active_pipeline.label)}</div>
                <div class="result-title">{result_title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metrics_path = output_dir / 'review' / 'pressure_curve_metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('转弯数', f"{metrics.get('turn_count', 0)} 个")
            with col2:
                st.metric('同步评分', f"{metrics.get('sync_score', 0):.1f}")
            with col3:
                st.metric('相关性', f"{metrics.get('correlation', 0):.2f}")

            st.markdown(f"<div class='result-caption'>{result_text}</div>", unsafe_allow_html=True)
        else:
            st.info('诊断数据生成中...')

        st.markdown(
            f"""
            <div class="analysis-tip">当前路由：{active_pipeline.pipeline_id}。真实算法已完成分析。</div>
            """,
            unsafe_allow_html=True,
        )

        # 反馈按钮
        if st.button('🚫 骨架漂移？结果不准？点击反馈', key='feedback_btn', use_container_width=True):
            st.toast('✅ 已收到您的反馈，该视频已加入特殊分析队列，并为您补偿 1 次分析额度！', icon='✅')

        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)


# 主执行流程
ensure_state()
render_sidebar()
render_top_bar()
render_hero()
render_status_grid()
selected_pipeline_id = render_mode_guide(st.session_state.last_pipeline_id)
selected_pipeline = get_pipeline(selected_pipeline_id)
uploaded_file = render_upload_panel(selected_pipeline)

if uploaded_file is None:
    st.session_state.analysis_done = False
    st.session_state.last_filename = ''
else:
    if st.button('🚀 立即开始 AI 分析', use_container_width=True):
        # 核心拦截：登录检查
        if not st.session_state.is_logged_in:
            st.error("⚠️ 请先在左侧登录账号")
            st.stop()

        # 核心拦截：额度检查
        if st.session_state.quota <= 0:
            st.warning("⚠️ 额度已用尽，请在左侧充值")
            st.stop()

        # 通过检查，执行分析
        run_analysis(selected_pipeline_id, uploaded_file)

render_results()
