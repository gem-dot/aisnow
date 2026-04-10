import streamlit as st
import sys
import subprocess
import time
import re
from pathlib import Path
from streamlit_option_menu import option_menu

WEBFACE_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = WEBFACE_ROOT.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from webface.db_manager import register_user, login_user, get_user_info, update_quota, get_referral_rate
from webface.config import UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR
from webface.pipelines import get_pipeline, prepare_pipeline_execution

st.set_page_config(
    page_title='AI 4 Snowboarding | 你的口袋单板教练',
    page_icon='🏂',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    /* ========== 1. Slate Dark 全局深空灰背景 ========== */
    :root {
        --bg-primary: #121212;
        --bg-secondary: #1E1E1E;
        --bg-card: #2A2A2A;
        --accent: #42e8ff;
        --accent-dim: rgba(66, 232, 255, 0.12);
        --text-primary: #E0E0E0;
        --text-secondary: #9E9E9E;
        --border: rgba(255, 255, 255, 0.08);
        --glow: 0 0 20px rgba(66, 232, 255, 0.15);
    }

    /* 主背景 */
    .stApp, [data-testid="stAppViewContainer"],
    .main .block-container, [data-testid="stMainBlockContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* 侧边栏背景 */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* 全局文本高对比度 */
    .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"],
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    .stTextInput label, .stSelectbox label {
        color: var(--text-primary) !important;
    }

    /* 输入框暗色化 */
    .stTextInput input, .stSelectbox select, .stTextArea textarea,
    [data-baseweb="input"] input, [data-baseweb="select"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: var(--glow) !important;
    }

    /* Tab 栏暗色化 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-secondary) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 10px !important;
        padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-card) !important;
        color: var(--accent) !important;
        box-shadow: var(--glow) !important;
    }

    /* 表单容器暗色化 */
    [data-testid="stForm"], [data-testid="stFormSubmitButton"] button {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }

    /* 按钮暗色化 */
    .stButton > button {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        box-shadow: var(--glow) !important;
        color: var(--accent) !important;
    }
    /* Primary 按钮 */
    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #42e8ff 0%, #2196F3 100%) !important;
        color: #121212 !important;
        border: none !important;
        font-weight: 700 !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 0 24px rgba(66, 232, 255, 0.4) !important;
    }

    /* Metric 卡片 */
    [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-weight: 800 !important;
    }

    /* Alert / Info / Success 暗色 */
    [data-testid="stAlert"], .stAlert {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    /* Code 块 */
    .stCodeBlock, code, [data-testid="stCode"] {
        background-color: var(--bg-card) !important;
        color: var(--accent) !important;
        border-radius: 8px !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-secondary) !important;
        border: 2px dashed rgba(66, 232, 255, 0.3) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent) !important;
    }

    /* Progress bar 发光效果 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #42e8ff, #2196F3) !important;
        border-radius: 8px !important;
    }

    /* ========== 2. 大标题不换行 ========== */
    h1 {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        color: var(--text-primary) !important;
        font-weight: 800 !important;
    }

    /* ========== 3. Radio 拟态卡片 ========== */
    div[role="radiogroup"] {
        gap: 8px !important;
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
        background-color: var(--bg-card) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        cursor: pointer !important;
        transition: all 0.25s ease !important;
        color: var(--text-primary) !important;
    }
    div[role="radiogroup"] label:hover {
        border-color: var(--accent) !important;
        background-color: var(--accent-dim) !important;
        box-shadow: var(--glow) !important;
    }
    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] label:has(input:checked) {
        border-color: var(--accent) !important;
        background-color: var(--accent-dim) !important;
        box-shadow: var(--glow) !important;
        color: var(--accent) !important;
    }

    /* option_menu 横向导航栏暗色化 */
    .nav-link {
        background-color: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        transition: all 0.2s ease !important;
    }
    .nav-link:hover {
        color: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    .nav-link.active, .nav-link--active {
        background: linear-gradient(135deg, rgba(66,232,255,0.15), rgba(33,150,243,0.15)) !important;
        color: var(--accent) !important;
        border-color: var(--accent) !important;
        box-shadow: var(--glow) !important;
    }

    /* ========== 4. 隐藏 Streamlit 原生水印 ========== */
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { visibility: hidden !important; height: 0 !important; }
    footer { visibility: hidden !important; }
    .viewerBadge_container__r5tak { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }

    /* ========== 额外：滚动条暗色 ========== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: var(--bg-card);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

PROGRESS_RE = re.compile(r'(ByteTrack|RTMPose-X|Render Overlay):\s+(\d+)%')
STAGE_WEIGHTS = {'ByteTrack': 0.3, 'RTMPose-X': 0.4, 'Render Overlay': 0.3}

def parse_progress(line):
    match = PROGRESS_RE.search(line)
    if match:
        stage, pct = match.groups()
        return stage, int(pct)
    return None, None

def run_analysis(video_path, task_name, mode):
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    output_dir = OUTPUT_DIR / task_name
    output_dir.mkdir(exist_ok=True)

    exec_plan = prepare_pipeline_execution(
        mode,
        task_name=task_name,
        input_path=video_path,
        output_dir=output_dir,
        cache_dir=CACHE_DIR,
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    process = subprocess.Popen(
        exec_plan.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    stage_progress = {}
    for line in process.stdout:
        stage, pct = parse_progress(line)
        if stage and pct is not None:
            stage_progress[stage] = pct
            total = sum(stage_progress.get(s, 0) * w for s, w in STAGE_WEIGHTS.items())
            progress_bar.progress(min(int(total), 99))
            status_text.text(f'{stage}: {pct}%')

    process.wait()
    progress_bar.progress(100)
    status_text.text('✅ 分析完成！')

    return output_dir

if not st.session_state.logged_in:
    st.title('🏂 AI 4 Snowboarding | 你的口袋单板教练')

    tab1, tab2 = st.tabs(['登录', '注册'])

    with tab1:
        with st.form('login_form'):
            username = st.text_input('用户名')
            password = st.text_input('密码', type='password')
            submit = st.form_submit_button('登录')

            if submit:
                if username and password:
                    if login_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success('✅ 登录成功！')
                        st.rerun()
                    else:
                        st.error('❌ 用户名或密码错误')
                else:
                    st.warning('⚠️ 请输入用户名和密码')

    with tab2:
        with st.form('register_form'):
            new_username = st.text_input('用户名')
            new_password = st.text_input('密码', type='password')
            referral_code = st.text_input('邀请码（可选）')
            submit_reg = st.form_submit_button('注册')

            if submit_reg:
                if new_username and new_password:
                    success, code = register_user(new_username, new_password, referral_code or None)
                    if success:
                        st.success(f'✅ 注册成功！赠送 3 次分析额度。你的邀请码：{code}')
                    else:
                        st.error('❌ 用户名已存在')
                else:
                    st.warning('⚠️ 请输入用户名和密码')

else:
    user_info = get_user_info(st.session_state.username)

    with st.sidebar:
        st.success(f'👤 {st.session_state.username}')
        st.metric('剩余分析额度', f'{user_info["quota"]} 次')

        st.markdown('---')
        st.subheader('📢 我的邀请')
        st.code(user_info['referral_code'], language=None)
        referral_rate = get_referral_rate(user_info['referral_count'])
        st.info(f'已邀请 {user_info["referral_count"]} 人 | 返佣比例 {int(referral_rate*100)}%')

        st.markdown('---')
        if st.button('💰 模拟充值 +10 次', use_container_width=True):
            update_quota(st.session_state.username, 10)
            st.rerun()

        if st.button('🚪 退出登录', use_container_width=True):
            logout()

    st.title('🏂 AI 4 Snowboarding | 你的口袋单板教练')

    mode = option_menu(
        None,
        ['CASI 新手扫盲', 'JSBA 进阶刻滑', 'SAJ 综合走刃'],
        icons=['🟢', '🔴', '🔵'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal',
    )

    mode_map = {
        'CASI 新手扫盲': 'casi',
        'JSBA 进阶刻滑': 'jsba',
        'SAJ 综合走刃': 'saj',
    }
    selected_mode = mode_map[mode]

    mode_desc = {
        'casi': '更适合刚入门或基础不稳定的滑手，重点检查重心是否站稳、姿态是否顺、身体有没有明显代偿。',
        'jsba': '更适合想提升刻滑质量的滑手，重点观察压板节奏、反弓建立、换刃时机和动作稳定性。',
        'saj': '更适合想做综合走刃评测的滑手，重点检查动态发力、动作连贯性和整段滑行节奏。',
    }

    st.info(mode_desc[selected_mode])

    uploaded_file = st.file_uploader('上传你的滑雪视频', type=['mp4', 'mov', 'avi'])

    if uploaded_file and st.button('🚀 开始分析', type='primary'):
        if user_info['quota'] <= 0:
            st.error('❌ 分析额度不足，请充值或邀请好友获取额度')
        else:
            task_name = uploaded_file.name.rsplit('.', 1)[0]
            video_path = UPLOAD_DIR / uploaded_file.name
            video_path.write_bytes(uploaded_file.read())

            update_quota(st.session_state.username, -1)

            with st.spinner('正在分析你的滑雪视频...'):
                output_dir = run_analysis(video_path, task_name, selected_mode)

            st.success('✅ 分析完成！')
            st.balloons()

            st.subheader('📊 分析结果')

            col1, col2, col3 = st.columns(3)

            overlay_video = output_dir / f'{task_name}_overlay.mp4'
            side_by_side_video = output_dir / f'{task_name}_side_by_side.mp4'
            pressure_video = output_dir / f'{task_name}_pressure_sync.mp4'

            with col1:
                if overlay_video.exists():
                    st.video(str(overlay_video))
                    with open(overlay_video, 'rb') as f:
                        st.download_button('⬇️ 下载骨骼叠加', f, file_name=overlay_video.name)

            with col2:
                if side_by_side_video.exists():
                    st.video(str(side_by_side_video))
                    with open(side_by_side_video, 'rb') as f:
                        st.download_button('⬇️ 下载对比视频', f, file_name=side_by_side_video.name)

            with col3:
                if pressure_video.exists():
                    st.video(str(pressure_video))
                    with open(pressure_video, 'rb') as f:
                        st.download_button('⬇️ 下载压力分析', f, file_name=pressure_video.name)
