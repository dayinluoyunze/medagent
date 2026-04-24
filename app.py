# -*- coding: utf-8 -*-
"""
Minimal Streamlit web app for the medical agent.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agents.medical_agent import create_agent
from config import (
    APP_CONFIG,
    CHAT_METRICS_FILENAME,
    DEFAULT_PROVIDER,
    EMBEDDING_PROVIDER,
    LOG_DIR,
    get_provider_config,
)
from rag.knowledge_manager import (
    UPLOADED_KNOWLEDGE_EXTENSIONS,
    delete_uploaded_knowledge,
    list_knowledge_files,
    read_knowledge_preview,
    write_text_knowledge,
    write_url_knowledge,
    write_uploaded_knowledge,
)
from rag.ocr import get_ocr_status


load_dotenv()

st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon="💊",
    layout="wide",
)

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "modelscope": "ModelScope",
    "minimax": "MiniMax",
}

EXAMPLE_PROMPTS = [
    "二甲双胍应该饭前吃还是饭后吃？",
    "忘记服用降压药时应该怎么处理？",
    "阿莫西林常见的不良反应有哪些？",
    "哺乳期使用感冒药需要注意什么？",
]


def inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root {
    --bg: #f6f3ed;
    --bg-soft: #f1efe8;
    --panel: rgba(255, 252, 247, 0.92);
    --ink: #162228;
    --muted: #63727a;
    --line: rgba(22, 34, 40, 0.10);
    --accent: #0f6a63;
    --accent-soft: rgba(15, 106, 99, 0.10);
    --shadow: 0 16px 40px rgba(18, 34, 40, 0.06);
}

html, body, [class*="css"] {
    font-family: "Space Grotesk", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(15, 106, 99, 0.10), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-soft) 100%);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    max-width: 980px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: rgba(248, 245, 239, 0.92);
    border-right: 1px solid rgba(22, 34, 40, 0.08);
}

.minimal-shell {
    margin-bottom: 0.95rem;
}

.minimal-title {
    margin: 0 0 0.22rem;
    font-size: clamp(1.8rem, 2.6vw, 2.4rem);
    line-height: 1.06;
    letter-spacing: -0.04em;
    font-weight: 700;
}

.minimal-copy {
    margin: 0;
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.7;
}

.status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.85rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.5rem 0.74rem;
    border-radius: 999px;
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    color: var(--ink);
    font-size: 0.78rem;
    font-weight: 600;
}

.starter-card {
    padding: 1rem 1.05rem;
    border-radius: 18px;
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    color: var(--muted);
    line-height: 1.72;
    margin-bottom: 0.9rem;
}

.starter-card strong {
    color: var(--ink);
}

.section-caption {
    margin: 0 0 0.7rem;
    color: var(--muted);
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 252, 247, 0.97);
    border-radius: 20px;
    border: 1px solid var(--line);
    box-shadow: 0 12px 28px rgba(18, 34, 40, 0.05);
    padding: 0.35rem 0.55rem;
    margin-bottom: 0.8rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    font-size: 0.97rem;
    line-height: 1.8;
}

[data-testid="stChatInput"] {
    background: rgba(255, 252, 247, 0.98);
    border: 1px solid var(--line);
    border-radius: 20px;
    box-shadow: 0 16px 34px rgba(18, 34, 40, 0.08);
}

[data-testid="stChatInput"] textarea {
    font-size: 0.98rem !important;
}

.stButton button {
    border-radius: 14px;
    border: 1px solid rgba(22, 34, 40, 0.08);
    background: linear-gradient(135deg, var(--accent) 0%, #114d64 100%);
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 10px 22px rgba(15, 106, 99, 0.14);
}

.stButton button[kind="secondary"] {
    background: rgba(255, 252, 247, 0.96);
    color: var(--ink);
    box-shadow: none;
}

[data-baseweb="select"] > div,
.stTextInput input {
    border-radius: 14px !important;
}

.sidebar-title {
    margin: 0 0 0.35rem;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.sidebar-copy {
    margin: 0 0 0.85rem;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.65;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 0.8rem;
    }
}
</style>
        """,
        unsafe_allow_html=True,
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
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    if "session_editor_id" not in st.session_state:
        st.session_state.session_editor_id = ""
    if "session_editor_title" not in st.session_state:
        st.session_state.session_editor_title = ""
    if "session_delete_confirm_id" not in st.session_state:
        st.session_state.session_delete_confirm_id = ""
    if "knowledge_form_version" not in st.session_state:
        st.session_state.knowledge_form_version = 0
    if "knowledge_preview_path" not in st.session_state:
        st.session_state.knowledge_preview_path = ""
    if "knowledge_delete_confirm_path" not in st.session_state:
        st.session_state.knowledge_delete_confirm_path = ""


def get_env_api_key(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    if provider == "modelscope":
        return os.getenv("MODELSCOPE_API_KEY", "")
    if provider == "minimax":
        return os.getenv("MINIMAX_API_KEY", "")
    return ""


def get_api_key_input(provider: str) -> str:
    return st.session_state.api_key_inputs.get(provider, "")


def set_api_key_input(provider: str, value: str) -> None:
    st.session_state.api_key_inputs[provider] = value


def provider_label(provider: str | None) -> str:
    if not provider:
        return "未初始化"
    return PROVIDER_LABELS.get(provider, provider)


def current_key_source(provider: str, api_key_input: str) -> str:
    if api_key_input.strip():
        return "手动输入"
    if get_env_api_key(provider):
        return ".env"
    return "缺失"


def copy_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": item["role"], "content": item["content"]} for item in messages]


def latest_metrics() -> dict[str, Any] | None:
    metrics_path = Path(LOG_DIR) / CHAT_METRICS_FILENAME
    if not metrics_path.exists():
        return None

    try:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    return None


def reset_session_action_state() -> None:
    st.session_state.session_editor_id = ""
    st.session_state.session_editor_title = ""
    st.session_state.session_delete_confirm_id = ""


def get_runtime_snapshot() -> dict[str, Any]:
    agent = st.session_state.agent
    selected_provider = st.session_state.provider
    selected_provider_config = get_provider_config(selected_provider)

    snapshot: dict[str, Any] = {
        "selected_provider": selected_provider,
        "selected_provider_label": provider_label(selected_provider),
        "selected_model": selected_provider_config["model"],
        "selected_key_source": current_key_source(selected_provider, get_api_key_input(selected_provider)),
        "active_provider": None,
        "active_provider_label": "未初始化",
        "active_model": "-",
        "actual_retrieval_mode": "未初始化",
        "vector_ready": False,
        "retriever_error": "",
        "documents": 0,
        "message_count": 0,
        "memory_enabled": APP_CONFIG["memory_enabled"],
        "summary_present": False,
        "current_session_id": "",
        "current_session_title": "",
        "sessions_count": 0,
    }

    if agent is None:
        return snapshot

    snapshot["active_provider"] = agent.provider
    snapshot["active_provider_label"] = provider_label(agent.provider)
    snapshot["active_model"] = agent.provider_config["model"]
    snapshot["message_count"] = len(agent.conversation_history)
    snapshot["summary_present"] = bool(agent.summary_memory.strip())
    session_meta = agent.get_session_meta()
    snapshot["current_session_id"] = session_meta["session_id"]
    snapshot["current_session_title"] = session_meta["title"]
    snapshot["sessions_count"] = len(agent.list_sessions())

    retriever = agent.retriever
    if retriever is None:
        snapshot["retriever_error"] = getattr(agent, "retriever_error", "")
        return snapshot

    snapshot["documents"] = len(retriever.documents)
    snapshot["vector_ready"] = bool(retriever.vectorstore)
    snapshot["retriever_error"] = retriever.init_error or getattr(agent, "retriever_error", "")

    mode = APP_CONFIG["retrieval_mode"].lower()
    if EMBEDDING_PROVIDER == "none" or not snapshot["vector_ready"]:
        snapshot["actual_retrieval_mode"] = "keyword"
    elif mode in {"auto", "hybrid"}:
        snapshot["actual_retrieval_mode"] = "hybrid rerank"
    elif mode == "vector":
        snapshot["actual_retrieval_mode"] = "vector"
    else:
        snapshot["actual_retrieval_mode"] = "vector + fallback"

    return snapshot


def sync_messages_from_agent() -> None:
    if st.session_state.agent is None:
        st.session_state.messages = []
        return
    st.session_state.messages = copy_history(st.session_state.agent.get_history())


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


def init_agent(provider: str, api_key_input: str) -> bool:
    api_key = api_key_input.strip() or get_env_api_key(provider)
    if not api_key:
        st.session_state.status_level = "error"
        st.session_state.status_message = "缺少当前 provider 的 API Key。请手动输入，或先在 .env 里配置。"
        return False

    try:
        if api_key_input.strip():
            st.session_state.manual_api_keys[provider] = api_key_input.strip()
        st.session_state.agent = create_agent(provider, api_key)
        st.session_state.provider = provider
        sync_messages_from_agent()
        reset_session_action_state()

        retriever_error = getattr(st.session_state.agent, "retriever_error", "")
        if retriever_error:
            st.session_state.status_level = "warning"
            st.session_state.status_message = (
                "Agent 已初始化，但向量检索不可用，当前会退回本地关键词检索。\n\n"
                f"原因：{retriever_error}"
            )
        else:
            st.session_state.status_level = "success"
            st.session_state.status_message = "Agent 初始化完成，可以直接开始提问。"
        return True
    except Exception as exc:
        st.session_state.status_level = "error"
        st.session_state.status_message = f"初始化失败：{exc}"
        return False


def clear_session_and_memory() -> None:
    if st.session_state.agent:
        st.session_state.agent.start_new_session()
        sync_messages_from_agent()
    else:
        st.session_state.messages = []
    st.session_state.pending_prompt = ""
    reset_session_action_state()
    st.session_state.status_level = "success"
    st.session_state.status_message = "已创建新会话，历史会话仍保留在本地。"


def load_session_into_agent(session_id: str) -> None:
    if not st.session_state.agent:
        return
    st.session_state.agent.load_session(session_id)
    sync_messages_from_agent()
    reset_session_action_state()
    st.session_state.status_level = "success"
    st.session_state.status_message = f"已切换到历史会话：{st.session_state.agent.current_session_title}"


def rename_session_in_agent(session_id: str, title: str) -> None:
    if not st.session_state.agent:
        return
    st.session_state.agent.rename_session(session_id, title)
    sync_messages_from_agent()
    reset_session_action_state()
    st.session_state.status_level = "success"
    st.session_state.status_message = "会话标题已更新。"


def delete_session_in_agent(session_id: str) -> None:
    if not st.session_state.agent:
        return
    st.session_state.agent.delete_session(session_id)
    sync_messages_from_agent()
    reset_session_action_state()
    st.session_state.status_level = "success"
    st.session_state.status_message = "历史会话已删除。"


def format_file_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def refresh_retriever_after_knowledge_update(saved_count: int, errors: list[str]) -> None:
    error_note = ""
    if errors:
        error_note = "\n\n部分资料保存失败：" + "；".join(errors[:3])

    if not st.session_state.agent:
        st.session_state.status_level = "warning" if errors else "success"
        st.session_state.status_message = (
            f"已保存 {saved_count} 个资料。初始化 Agent 后会自动加载新资料。{error_note}"
        )
        return

    try:
        st.session_state.agent.init_retriever()
    except Exception as exc:
        st.session_state.status_level = "error"
        st.session_state.status_message = f"资料已保存，但刷新知识库失败：{exc}{error_note}"
        return

    retriever_error = getattr(st.session_state.agent, "retriever_error", "")
    if retriever_error:
        st.session_state.status_level = "warning"
        st.session_state.status_message = (
            f"已保存 {saved_count} 个资料，但向量检索刷新异常，当前会退回关键词检索。\n\n"
            f"原因：{retriever_error}{error_note}"
        )
        return

    st.session_state.status_level = "warning" if errors else "success"
    st.session_state.status_message = f"已保存 {saved_count} 个资料，并刷新知识库。{error_note}"


def rebuild_retriever_action() -> None:
    if not st.session_state.agent:
        st.session_state.status_level = "warning"
        st.session_state.status_message = "请先初始化 Agent，再重建知识索引。"
        return

    try:
        st.session_state.agent.init_retriever()
    except Exception as exc:
        st.session_state.status_level = "error"
        st.session_state.status_message = f"知识索引重建失败：{exc}"
        return

    retriever_error = getattr(st.session_state.agent, "retriever_error", "")
    if retriever_error:
        st.session_state.status_level = "warning"
        st.session_state.status_message = f"索引已重建，但向量检索不可用，当前退回关键词检索：{retriever_error}"
        return

    st.session_state.status_level = "success"
    st.session_state.status_message = "知识索引已重建。"


def delete_knowledge_action(relative_path: str) -> None:
    try:
        deleted_path = delete_uploaded_knowledge(relative_path)
    except Exception as exc:
        st.session_state.status_level = "error"
        st.session_state.status_message = f"删除资料失败：{exc}"
        return

    if st.session_state.knowledge_preview_path == relative_path:
        st.session_state.knowledge_preview_path = ""
    st.session_state.knowledge_delete_confirm_path = ""
    if st.session_state.agent:
        rebuild_retriever_action()
        if st.session_state.status_level == "success":
            st.session_state.status_message = f"已删除资料并重建索引：{deleted_path.name}"
    else:
        st.session_state.status_level = "success"
        st.session_state.status_message = f"已删除资料：{deleted_path.name}。初始化 Agent 后会加载最新知识库。"


def parse_session_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def group_sessions_by_date(sessions: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    today = datetime.now().date()
    grouped: dict[str, list[dict[str, Any]]] = {
        "今天": [],
        "昨天": [],
        "更早": [],
        "无时间": [],
    }

    for item in sessions:
        dt = parse_session_timestamp(str(item.get("updated_at", "")))
        if dt is None:
            grouped["无时间"].append(item)
            continue
        days = (today - dt.date()).days
        if days == 0:
            grouped["今天"].append(item)
        elif days == 1:
            grouped["昨天"].append(item)
        else:
            grouped["更早"].append(item)

    return [(label, items) for label, items in grouped.items() if items]


def format_session_button_label(item: dict[str, Any], *, is_current: bool) -> str:
    title = str(item.get("title", "未命名会话")).strip() or "未命名会话"
    dt = parse_session_timestamp(str(item.get("updated_at", "")))
    time_text = dt.strftime("%H:%M") if dt else "--:--"
    count = item.get("message_count", 0)
    suffix = " · 当前" if is_current else ""
    return f"{title} · {time_text} · {count}条{suffix}"


def render_session_editor(item: dict[str, Any]) -> None:
    session_id = item["session_id"]
    input_key = f"rename_input_{session_id}"
    default_value = st.session_state.session_editor_title or str(item.get("title", ""))
    title_value = st.text_input(
        "重命名会话",
        value=default_value,
        key=input_key,
        label_visibility="collapsed",
        placeholder="输入新的会话标题",
    )
    st.session_state.session_editor_title = title_value
    save_col, cancel_col = st.columns(2, gap="small")
    with save_col:
        if st.button("保存", key=f"rename_save_{session_id}", use_container_width=True):
            rename_session_in_agent(session_id, title_value)
            st.rerun()
    with cancel_col:
        if st.button("取消", key=f"rename_cancel_{session_id}", use_container_width=True):
            reset_session_action_state()
            st.rerun()


def render_session_delete_confirm(item: dict[str, Any]) -> None:
    session_id = item["session_id"]
    st.caption("删除后无法恢复。")
    confirm_col, cancel_col = st.columns(2, gap="small")
    with confirm_col:
        if st.button("确认删除", key=f"delete_confirm_{session_id}", use_container_width=True):
            delete_session_in_agent(session_id)
            st.rerun()
    with cancel_col:
        if st.button("取消", key=f"delete_cancel_{session_id}", use_container_width=True):
            reset_session_action_state()
            st.rerun()


def render_session_history(snapshot: dict[str, Any]) -> None:
    if not st.session_state.agent or not APP_CONFIG["memory_enabled"]:
        return

    sessions = st.session_state.agent.list_sessions()
    if not sessions:
        return

    current_session_id = snapshot["current_session_id"]
    with st.expander(f"历史会话 ({len(sessions)})", expanded=False):
        for group_label, group_items in group_sessions_by_date(sessions):
            st.caption(group_label)
            for item in group_items:
                session_id = item["session_id"]
                is_current = session_id == current_session_id
                row_cols = st.columns([4.8, 1.2, 1.2], gap="small")
                with row_cols[0]:
                    if st.button(
                        format_session_button_label(item, is_current=is_current),
                        key=f"session_open_{session_id}",
                        use_container_width=True,
                        type="primary" if is_current else "secondary",
                    ):
                        load_session_into_agent(session_id)
                        st.rerun()
                with row_cols[1]:
                    if st.button("改名", key=f"session_rename_{session_id}", use_container_width=True):
                        st.session_state.session_editor_id = session_id
                        st.session_state.session_editor_title = str(item.get("title", ""))
                        st.session_state.session_delete_confirm_id = ""
                        st.rerun()
                with row_cols[2]:
                    if st.button("删除", key=f"session_delete_{session_id}", use_container_width=True):
                        st.session_state.session_delete_confirm_id = session_id
                        st.session_state.session_editor_id = ""
                        st.session_state.session_editor_title = ""
                        st.rerun()

                if st.session_state.session_editor_id == session_id:
                    render_session_editor(item)
                elif st.session_state.session_delete_confirm_id == session_id:
                    render_session_delete_confirm(item)


def render_knowledge_manager() -> None:
    file_types = [suffix.lstrip(".") for suffix in UPLOADED_KNOWLEDGE_EXTENSIONS]
    version = st.session_state.knowledge_form_version

    with st.expander("知识库", expanded=False):
        st.caption("上传文件或粘贴文本，保存后写入本地知识库并刷新 RAG。")
        uploaded_files = st.file_uploader(
            "上传资料文件",
            type=file_types,
            accept_multiple_files=True,
            key=f"knowledge_files_{version}",
            help="支持 md、txt、json、jsonl、csv、docx、pdf、图片、html、xml、yaml 等可解析文件。扫描版 PDF 和图片需要本机 OCR 环境。",
        )
        text_title = st.text_input(
            "资料标题",
            key=f"knowledge_title_{version}",
            placeholder="例如：高血压用药注意事项",
        )
        url_content = st.text_area(
            "从 URL 导入",
            key=f"knowledge_urls_{version}",
            placeholder="每行一个 URL，保存后会抓取网页正文并生成本地知识快照。",
            height=82,
        )
        text_content = st.text_area(
            "粘贴文本",
            key=f"knowledge_text_{version}",
            placeholder="可以粘贴说明书片段、指南摘要、内部 FAQ 等文本资料。",
            height=110,
        )

        if st.button("保存并刷新知识库", use_container_width=True, key=f"save_knowledge_{version}"):
            saved_paths: list[Path] = []
            errors: list[str] = []

            for uploaded_file in uploaded_files or []:
                try:
                    saved_paths.append(
                        write_uploaded_knowledge(uploaded_file.name, uploaded_file.getvalue())
                    )
                except Exception as exc:
                    errors.append(f"{uploaded_file.name}: {exc}")

            if text_content.strip():
                try:
                    saved_paths.append(write_text_knowledge(text_title, text_content))
                except Exception as exc:
                    errors.append(f"文本资料: {exc}")

            if url_content.strip():
                with st.spinner("正在抓取 URL 并生成知识快照..."):
                    url_paths, url_errors = write_url_knowledge(url_content, text_title)
                saved_paths.extend(url_paths)
                errors.extend(url_errors)

            if not saved_paths and not errors:
                st.session_state.status_level = "warning"
                st.session_state.status_message = "请先上传文件、粘贴文本，或输入 URL。"
                st.rerun()

            if not saved_paths:
                st.session_state.status_level = "error"
                st.session_state.status_message = "资料保存失败：" + "；".join(errors[:3])
                st.rerun()

            refresh_retriever_after_knowledge_update(len(saved_paths), errors)
            if saved_paths:
                st.session_state.knowledge_form_version += 1
            st.rerun()

        st.divider()
        manage_cols = st.columns(2, gap="small")
        with manage_cols[0]:
            if st.button("重建索引", use_container_width=True, key="rebuild_knowledge_index"):
                rebuild_retriever_action()
                st.rerun()
        with manage_cols[1]:
            if st.button("刷新列表", use_container_width=True, key="refresh_knowledge_files"):
                st.rerun()

        knowledge_files = list_knowledge_files(limit=80)
        if not knowledge_files:
            st.caption("知识库暂无可加载资料。")
            return

        uploaded_count = sum(1 for item in knowledge_files if item["deletable"])
        st.caption(f"知识文件：{len(knowledge_files)} 个，其中网页添加 {uploaded_count} 个。")

        options = [item["relative_path"] for item in knowledge_files]
        selected_path = st.selectbox(
            "选择资料",
            options=options,
            format_func=lambda value: next(
                (
                    f"{item['name']} · {item['source']} · {format_file_size(item['size'])}"
                    for item in knowledge_files
                    if item["relative_path"] == value
                ),
                value,
            ),
            key="knowledge_file_selector",
        )
        selected_item = next(
            (item for item in knowledge_files if item["relative_path"] == selected_path),
            None,
        )
        if not selected_item:
            return

        st.caption(
            f"`{selected_item['relative_path']}` · {selected_item['modified_at']} · "
            f"{'可删除' if selected_item['deletable'] else '内置资料'}"
        )
        file_cols = st.columns(2, gap="small")
        with file_cols[0]:
            if st.button("预览", use_container_width=True, key=f"preview_{abs(hash(selected_path))}"):
                st.session_state.knowledge_preview_path = selected_path
                st.session_state.knowledge_delete_confirm_path = ""
                st.rerun()
        with file_cols[1]:
            if st.button(
                "删除",
                use_container_width=True,
                disabled=not selected_item["deletable"],
                key=f"delete_{abs(hash(selected_path))}",
            ):
                st.session_state.knowledge_delete_confirm_path = selected_path
                st.session_state.knowledge_preview_path = ""
                st.rerun()

        if st.session_state.knowledge_preview_path == selected_path:
            try:
                st.code(read_knowledge_preview(selected_path), language="markdown")
            except Exception as exc:
                st.warning(f"预览失败：{exc}")

        if st.session_state.knowledge_delete_confirm_path == selected_path:
            st.warning("只允许删除网页上传/导入的资料，删除后会刷新索引。")
            confirm_cols = st.columns(2, gap="small")
            with confirm_cols[0]:
                if st.button("确认删除", use_container_width=True, key=f"delete_confirm_{abs(hash(selected_path))}"):
                    delete_knowledge_action(selected_path)
                    st.rerun()
            with confirm_cols[1]:
                if st.button("取消", use_container_width=True, key=f"delete_cancel_{abs(hash(selected_path))}"):
                    st.session_state.knowledge_delete_confirm_path = ""
                    st.rerun()


def render_sidebar(snapshot: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Session</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-copy">所有控制项都收在这里，主屏只保留聊天。</p>',
            unsafe_allow_html=True,
        )

        selected_provider = st.selectbox(
            "聊天模型",
            options=["openai", "modelscope", "minimax"],
            index=["openai", "modelscope", "minimax"].index(st.session_state.provider),
            format_func=lambda value: PROVIDER_LABELS[value],
        )
        st.session_state.provider = selected_provider

        api_key_input = st.text_input(
            "API Key",
            type="password",
            value=get_api_key_input(selected_provider),
            help="留空则回退到 .env 中当前 provider 对应的 key。",
        )
        set_api_key_input(selected_provider, api_key_input)

        action_cols = st.columns(2, gap="small")
        with action_cols[0]:
            if st.button("初始化", type="primary", use_container_width=True):
                init_agent(selected_provider, api_key_input)
        with action_cols[1]:
            if st.button("新建会话", use_container_width=True):
                clear_session_and_memory()

        show_status(st.session_state.status_message, st.session_state.status_level)

        with st.expander("查看当前状态", expanded=False):
            ocr_status = get_ocr_status()
            st.caption(f"当前会话：`{snapshot['active_provider_label']}`")
            st.caption(f"当前模型：`{snapshot['active_model']}`")
            st.caption(f"当前标题：`{snapshot['current_session_title'] or '-'} `")
            st.caption(f"Embedding：`{EMBEDDING_PROVIDER}`")
            st.caption(f"检索模式：`{snapshot['actual_retrieval_mode']}`")
            st.caption(f"知识文档：`{snapshot['documents']}`")
            st.caption(f"OCR：`{'可用' if ocr_status['available'] else '不可用'}`")
            if not ocr_status["available"]:
                st.caption(f"OCR 原因：`{ocr_status['reason']}`")
            st.caption(f"消息数量：`{snapshot['message_count']}`")
            st.caption(f"Key 来源：`{snapshot['selected_key_source']}`")
            if snapshot["retriever_error"]:
                st.warning(snapshot["retriever_error"])

        render_knowledge_manager()

        if st.session_state.agent and APP_CONFIG["memory_enabled"]:
            render_session_history(snapshot)

        metrics = latest_metrics()
        if metrics:
            with st.expander("最近一次调用", expanded=False):
                st.caption(f"状态：`{metrics.get('status', '-')}`")
                st.caption(f"耗时：`{metrics.get('duration_ms', 0)} ms`")
                st.caption(f"命中文档：`{metrics.get('retrieved_doc_count', 0)}`")
                st.caption(f"Fallback：`{'yes' if metrics.get('fallback_used') else 'no'}`")


def render_header(snapshot: dict[str, Any]) -> None:
    pills = [
        f"会话 {snapshot['current_session_title'] or '未命名'}",
        f"会话 {snapshot['active_provider_label']}",
        f"检索 {snapshot['actual_retrieval_mode']}",
        f"Embedding {EMBEDDING_PROVIDER}",
    ]
    if snapshot["summary_present"]:
        pills.append("Memory on")
    if latest_metrics():
        pills.append(f"最近 {latest_metrics().get('status', '-')}")
    pill_html = "".join(f'<span class="status-pill">{item}</span>' for item in pills)

    st.markdown(
        f"""
<div class="minimal-shell">
  <div class="minimal-title">MedAgent RAG</div>
  <p class="minimal-copy">面向医药问答的 RAG Agent。主屏保持聊天，知识库、会话和运行状态收在左侧。</p>
  <div class="status-strip">{pill_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_example_prompts(enabled: bool) -> None:
    st.markdown('<div class="section-caption">Quick Prompts</div>', unsafe_allow_html=True)
    prompt_cols = st.columns(2, gap="small")
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        with prompt_cols[index % 2]:
            if st.button(
                prompt,
                key=f"example_prompt_{index}",
                use_container_width=True,
                disabled=not enabled,
            ):
                st.session_state.pending_prompt = prompt
                st.rerun()


def display_message(role: str, content: str) -> None:
    avatar = "🧑" if role == "user" else "💊"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def process_prompt(prompt: str) -> None:
    display_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("检索知识与生成回答中..."):
        response = st.session_state.agent.chat(prompt)

    display_message("assistant", response)
    st.session_state.messages.append({"role": "assistant", "content": response})


def render_chat_workspace(snapshot: dict[str, Any]) -> None:
    agent_ready = st.session_state.agent is not None

    if not agent_ready:
        st.markdown(
            """
<div class="starter-card">
  <strong>先初始化 Agent。</strong><br>
  左侧选择模型、确认 API Key，然后点击“初始化”。<br>
  初始化成功后，这里只保留聊天与示例问题。
</div>
            """,
            unsafe_allow_html=True,
        )
        render_example_prompts(enabled=False)
        st.chat_input("请先初始化 Agent 后再提问。", disabled=True)
        return

    if not st.session_state.messages:
        st.markdown(
            f"""
<div class="starter-card">
  <strong>当前会话已就绪。</strong><br>
  会话：{snapshot['current_session_title'] or '未命名'} · 模型：{snapshot['active_provider_label']} · 检索：{snapshot['actual_retrieval_mode']} · 知识文档：{snapshot['documents']}<br>
  你可以直接输入问题，或者先点下面的示例问题。
</div>
            """,
            unsafe_allow_html=True,
        )
        render_example_prompts(enabled=True)

    for message in st.session_state.messages:
        display_message(message["role"], message["content"])

    queued_prompt = st.session_state.pending_prompt
    prompt = st.chat_input("输入医药问题，例如用法用量、不良反应、禁忌或特殊人群注意事项...")
    if queued_prompt:
        st.session_state.pending_prompt = ""
        process_prompt(queued_prompt)
        return

    if prompt:
        process_prompt(prompt)


def main() -> None:
    inject_styles()
    init_session_state()

    snapshot = get_runtime_snapshot()
    render_sidebar(snapshot)
    snapshot = get_runtime_snapshot()
    render_header(snapshot)
    render_chat_workspace(snapshot)


if __name__ == "__main__":
    main()
