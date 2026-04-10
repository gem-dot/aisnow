from __future__ import annotations

import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth

# 页面配置
st.set_page_config(
    page_title='AI 4 Snowboarding | 你的口袋单板教练',
    page_icon='🏂',
    layout='wide',
    initial_sidebar_state='expanded',
)

# 加载配置（使用绝对路径）
config_path = Path(__file__).parent / 'config.yaml'
with open(config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

# 创建认证器
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# 登录界面 - 防御性处理返回值
result = authenticator.login('main')

# 处理返回值
if result is None:
    authentication_status = None
    name = None
    username = None
else:
    name, authentication_status, username = result

if authentication_status:
    # 登录成功 - 显示主界面
    authenticator.logout('退出登录', 'sidebar')
    st.sidebar.success(f'欢迎回来，{name}！')

    # 显示额度
    quota = config['credentials']['usernames'][username].get('quota', 0)
    st.sidebar.metric('剩余分析额度', f'{quota} 次')

    # 这里导入并显示滑雪分析界面
    st.title('🏂 AI 4 Snowboarding')
    st.info('滑雪分析界面即将整合...')

elif authentication_status == False:
    st.error('用户名或密码错误')
elif authentication_status == None:
    st.warning('请输入用户名和密码登录')
    st.info('演示账号：demo / demo123')
