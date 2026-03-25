# -*- coding: utf-8 -*-
"""
Streamlit web app for the medical agent.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from agents.medical_agent import create_agent
from config import APP_CONFIG, DEFAULT_PROVIDER, EMBEDDING_PROVIDER


load_dotenv()

st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon="💊",
    layout="wide",
)


def init_session_state() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "provider" not in st.session_state:
        st.session_state.provider = DEFAULT_PROVIDER
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    if "status_level" not in st.session_state:
        st.session_state.status_level = "info"
    if "manual_api_keys" not in st.session_state:
        st.session_state.manual_api_keys = {}
    if "api_key_inputs" not in st.session_state:
        st.session_state.api_key_inputs = {}


def get_env_api_key(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    if provider == "modelscope":
        return os.getenv("MODELSCOPE_API_KEY", "")
    if provider == "minimax":
        return os.getenv("MINIMAX_API_KEY", "")
    return ""


def get_active_api_key(provider: str) -> str:
    return st.session_state.manual_api_keys.get(provider, get_env_api_key(provider))


def get_api_key_input(provider: str) -> str:
    return st.session_state.api_key_inputs.get(provider, "")


def set_api_key_input(provider: str, value: str) -> None:
    st.session_state.api_key_inputs[provider] = value


def init_agent(provider: str, api_key_input: str) -> bool:
    api_key = api_key_input.strip() or get_env_api_key(provider)
    if not api_key:
        st.session_state.status_level = "error"
        st.session_state.status_message = "请输入当前 provider 对应的 API Key，或先在 .env 中配置。"
        return False

    try:
        if api_key_input.strip():
            st.session_state.manual_api_keys[provider] = api_key_input.strip()
        st.session_state.agent = create_agent(provider, api_key)
        st.session_state.provider = provider

        retriever_error = getattr(st.session_state.agent, "retriever_error", "")
        if retriever_error:
            st.session_state.status_level = "warning"
            st.session_state.status_message = (
                "Agent 已初始化，但向量检索不可用，当前将退回为本地关键词检索。\n\n"
                f"原因：{retriever_error}"
            )
        else:
            st.session_state.status_level = "success"
            st.session_state.status_message = "Agent 初始化成功，知识库检索已启用。"
        return True
    except Exception as exc:
        st.session_state.status_level = "error"
        st.session_state.status_message = f"初始化失败：{exc}"
        return False


def display_message(role: str, content: str) -> None:
    st.chat_message(role).write(content)


def show_status(message: str, level: str) -> None:
    if not message:
        return
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def main() -> None:
    init_session_state()

    st.title(f"💊 {APP_CONFIG['title']}")
    st.markdown(f"*{APP_CONFIG['description']}*")

    with st.sidebar:
        st.header("配置")

        provider = st.selectbox(
            "选择模型",
            options=["openai", "modelscope", "minimax"],
            index=["openai", "modelscope", "minimax"].index(st.session_state.provider),
            format_func=lambda value: {
                "openai": "OpenAI",
                "modelscope": "ModelScope",
                "minimax": "MiniMax",
            }[value],
        )

        api_key_input = st.text_input(
            "API Key",
            type="password",
            value=get_api_key_input(provider),
            help="输入后仅在当前会话使用；如果留空，则回退到 .env 中的 key。",
        )
        set_api_key_input(provider, api_key_input)

        env_configured = bool(get_env_api_key(provider))
        manual_configured = provider in st.session_state.manual_api_keys

        st.caption(f"当前聊天 provider：`{provider}`")
        st.caption(f"当前 embedding provider：`{EMBEDDING_PROVIDER}`")
        st.caption(f".env key 状态：`{'已配置' if env_configured else '未配置'}`")
        st.caption(f"手动 key 状态：`{'已输入并生效' if manual_configured else '未输入'}`")

        if st.button("初始化 Agent", type="primary", use_container_width=True):
            init_agent(provider, api_key_input)

        show_status(
            st.session_state.status_message,
            st.session_state.status_level,
        )

        st.divider()

        if st.button("清空对话历史", use_container_width=True):
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.markdown(
            """
### 使用说明

1. 选择模型
2. 需要时手动输入 API Key；不输入则使用 `.env`
3. 点击“初始化 Agent”
4. 在主界面输入问题

### 当前行为

- 模型负责主回答，知识库用于补充信息
- 向量检索失败时会自动退回本地关键词检索
- 模型连接失败时会退回知识库内容摘要
- 涉及具体用药请以医生或药师意见为准
"""
        )

    if st.session_state.agent is None:
        st.info("请先在侧边栏初始化 Agent。")
        st.markdown(
            """
### 欢迎使用医药智能 Agent

你可以尝试这些问题：

- 二甲双胍的用法
- 二甲双胍忘记服药怎么办？
- 二甲双胍可以长期服用吗？
"""
        )
        return

    st.caption(
        f"当前会话 provider：`{st.session_state.provider}` | "
        f"知识库检索：`{'已启用' if st.session_state.agent.retriever else '不可用'}`"
    )

    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    prompt = st.chat_input("请输入您的问题...")
    if not prompt:
        return

    display_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("思考中..."):
        response = st.session_state.agent.chat(prompt)

    display_message("assistant", response)
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
