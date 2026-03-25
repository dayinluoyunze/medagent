# -*- coding: utf-8 -*-
"""
Application configuration.

Supports multiple OpenAI-compatible providers for chat and embeddings.
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv


load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "modelscope": {
        "api_base": "https://api-inference.modelscope.cn/v1",
        "model": "moonshotai/Kimi-K2.5",
        "api_key_env": "MODELSCOPE_API_KEY",
    },
    "minimax": {
        "api_base": "https://api.minimax.chat/v1",
        "model": "abab6.5s-chat",
        "api_key_env": "MINIMAX_API_KEY",
    },
}

EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model": "text-embedding-3-small",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "modelscope": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "api_base": "https://api-inference.modelscope.cn/v1",
        "api_key_env": "MODELSCOPE_API_KEY",
    },
}

EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    DEFAULT_PROVIDER if DEFAULT_PROVIDER in EMBEDDING_MODELS else "openai",
)

APP_CONFIG = {
    "title": "医药智能 Agent",
    "description": "药品知识问答助手",
    "system_prompt": """你是一个专业的医药知识助手。

回答规则：
1. 优先基于可靠医学常识和模型自身能力回答用户问题。
2. 如果提供了知识库内容，应将其作为补充证据和优先参考来源。
3. 如果知识库内容与模型常识不一致，优先提示用户存在冲突，并建议核对说明书、医生或药师意见。
4. 不提供诊断结论，不承诺疗效，不替代医生面诊。
5. 涉及具体用法用量、不良反应、禁忌证或特殊人群时，回答要谨慎、清晰，并提醒用户遵医嘱。
6. 如果回答中引用了知识库信息，请明确说明“根据知识库补充”。
7. 默认不要只给一句话结论。除非用户明确要求简短，否则尽量给出较完整的回答。
8. 回答医药问题时，优先采用以下结构：
   - 先给核心结论
   - 再补充用法用量、适用场景或注意事项
   - 如有必要，再补充不良反应、禁忌证、漏服处理或就医建议

请用专业、清晰、自然的中文作答。回答长度可以适当详细，但避免空话和重复。""",
    "max_history": 10,
    "temperature": 0.35,
    "request_timeout": 30,
    "max_retries": 1,
    "max_tokens": 3200,
    "url_fetch_timeout": 15,
    "memory_recent_messages": 6,
    "memory_summary_trigger_messages": 12,
}

KNOWLEDGE_DIR = "knowledge"
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(".cache", "vectorstore"))
VECTORSTORE_MANIFEST = "manifest.json"
MEMORY_DIR = os.getenv("MEMORY_DIR", "memory")
MEMORY_HISTORY_FILENAME = "conversation_history.md"
MEMORY_SUMMARY_FILENAME = "conversation_summary.md"
LOG_DIR = os.getenv("LOG_DIR", "logs")
CHAT_METRICS_FILENAME = "chat_metrics.jsonl"
SUPPORTED_KNOWLEDGE_EXTENSIONS = {
    ".md",
    ".txt",
    ".text",
    ".rst",
    ".log",
    ".json",
    ".jsonl",
    ".csv",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".docx",
    ".url",
    ".webloc",
    ".urls",
}


def get_provider_config(provider: str, *, for_embedding: bool = False) -> Dict[str, Any]:
    if for_embedding:
        return EMBEDDING_MODELS.get(provider, EMBEDDING_MODELS["openai"])
    return LLM_PROVIDERS.get(provider, LLM_PROVIDERS["openai"])


def get_api_key_for_provider(provider: str, *, for_embedding: bool = False) -> str:
    config = get_provider_config(provider, for_embedding=for_embedding)
    env_name = config.get("api_key_env", "")
    return os.getenv(env_name, "")
