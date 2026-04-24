# -*- coding: utf-8 -*-
"""
Application configuration.

Supports multiple OpenAI-compatible providers for chat and embeddings.
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv


load_dotenv()


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]

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
        "api_base": "https://api.minimaxi.com/v1",
        "model": "MiniMax-M2.5",
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
    DEFAULT_PROVIDER if DEFAULT_PROVIDER in EMBEDDING_MODELS else "none",
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
    "memory_enabled": _get_env_bool("MEMORY_ENABLED", True),
    "retrieval_mode": os.getenv("RETRIEVAL_MODE", "auto"),
    "citation_snippet_length": int(os.getenv("CITATION_SNIPPET_LENGTH", "120")),
    "log_question_max_chars": int(os.getenv("LOG_QUESTION_MAX_CHARS", "160")),
    "log_redact_questions": _get_env_bool("LOG_REDACT_QUESTIONS", True),
    "allow_remote_knowledge_fetch": _get_env_bool("ALLOW_REMOTE_KNOWLEDGE_FETCH", True),
    "remote_knowledge_allowlist": _get_env_list("REMOTE_KNOWLEDGE_ALLOWLIST"),
    "allow_url_knowledge_ingestion": _get_env_bool("ALLOW_URL_KNOWLEDGE_INGESTION", True),
    "url_knowledge_max_bytes": int(os.getenv("URL_KNOWLEDGE_MAX_BYTES", str(10 * 1024 * 1024))),
    "ocr_enabled": _get_env_bool("OCR_ENABLED", True),
    "ocr_lang": os.getenv("OCR_LANG", "chi_sim+eng"),
    "ocr_dpi": int(os.getenv("OCR_DPI", "200")),
    "ocr_max_pages": int(os.getenv("OCR_MAX_PAGES", "20")),
    "tesseract_cmd": os.getenv("TESSERACT_CMD", ""),
    "tessdata_prefix": os.getenv("TESSDATA_PREFIX", ""),
}

KNOWLEDGE_DIR = "knowledge"
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(".cache", "vectorstore"))
VECTORSTORE_MANIFEST = "manifest.json"
MEMORY_DIR = os.getenv("MEMORY_DIR", "memory")
MEMORY_HISTORY_FILENAME = "conversation_history.md"
MEMORY_SUMMARY_FILENAME = "conversation_summary.md"
MEMORY_CURRENT_SESSION_FILENAME = "current_session.json"
MEMORY_SESSIONS_DIRNAME = "sessions"
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
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".url",
    ".webloc",
    ".urls",
}


def get_provider_config(provider: str, *, for_embedding: bool = False) -> Dict[str, Any]:
    if for_embedding:
        if provider == "none":
            return {"model": "", "api_base": "", "api_key_env": ""}
        return EMBEDDING_MODELS.get(provider, EMBEDDING_MODELS["openai"])
    return LLM_PROVIDERS.get(provider, LLM_PROVIDERS["openai"])


def get_api_key_for_provider(provider: str, *, for_embedding: bool = False) -> str:
    config = get_provider_config(provider, for_embedding=for_embedding)
    env_name = config.get("api_key_env", "")
    return os.getenv(env_name, "")
