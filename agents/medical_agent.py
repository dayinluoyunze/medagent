# -*- coding: utf-8 -*-
"""
Core medical agent chat flow with persisted markdown memory.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, OpenAI

from config import (
    APP_CONFIG,
    DEFAULT_PROVIDER,
    EMBEDDING_PROVIDER,
    CHAT_METRICS_FILENAME,
    LOG_DIR,
    MEMORY_DIR,
    MEMORY_HISTORY_FILENAME,
    MEMORY_SUMMARY_FILENAME,
    get_api_key_for_provider,
    get_provider_config,
)
from rag.retriever import create_retriever


class MedicalAgent:
    """Medical QA agent with optional retrieval augmentation and markdown memory."""

    def __init__(self, provider: str = DEFAULT_PROVIDER, api_key: str = ""):
        self.provider = provider
        self.api_key = api_key or get_api_key_for_provider(provider)
        self.provider_config = get_provider_config(self.provider)
        self.llm = self._init_llm()
        self.raw_client = self._init_raw_client()
        self.retriever = None
        self.retriever_error = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.summary_memory = ""
        self.memory_dir = Path(MEMORY_DIR)
        self.history_file = self.memory_dir / MEMORY_HISTORY_FILENAME
        self.summary_file = self.memory_dir / MEMORY_SUMMARY_FILENAME
        self.log_dir = Path(LOG_DIR)
        self.metrics_file = self.log_dir / CHAT_METRICS_FILENAME
        self._load_memory()

    def _init_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.provider_config["model"],
            api_key=self.api_key,
            base_url=self.provider_config["api_base"],
            temperature=APP_CONFIG["temperature"],
            max_tokens=APP_CONFIG["max_tokens"],
            request_timeout=APP_CONFIG["request_timeout"],
            max_retries=APP_CONFIG["max_retries"],
        )

    def _init_raw_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.provider_config["api_base"],
            timeout=APP_CONFIG["request_timeout"],
            max_retries=APP_CONFIG["max_retries"],
        )

    def _load_memory(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_history = self._load_history_markdown()
        self.summary_memory = self._load_summary_markdown()

    def _load_history_markdown(self) -> List[Dict[str, str]]:
        if not self.history_file.exists():
            return []

        content = self.history_file.read_text(encoding="utf-8")
        pattern = re.compile(
            r"## (User|Assistant)\n(.*?)(?=\n## (?:User|Assistant)\n|\Z)",
            re.S,
        )
        messages: List[Dict[str, str]] = []
        for role, body in pattern.findall(content):
            messages.append(
                {
                    "role": "user" if role == "User" else "assistant",
                    "content": body.strip(),
                }
            )
        return messages

    def _load_summary_markdown(self) -> str:
        if not self.summary_file.exists():
            return ""

        content = self.summary_file.read_text(encoding="utf-8").strip()
        if content.startswith("# Summary Memory"):
            content = content[len("# Summary Memory") :].strip()
        return content

    def _save_memory(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        history_lines = ["# Conversation History", ""]
        for msg in self.conversation_history:
            heading = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"## {heading}")
            history_lines.append(msg["content"].strip())
            history_lines.append("")
        self.history_file.write_text("\n".join(history_lines).strip() + "\n", encoding="utf-8")

        summary_lines = ["# Summary Memory", "", self.summary_memory.strip()]
        self.summary_file.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")

    def _generate_summary_text(self, messages: List[Dict[str, str]]) -> str:
        transcript_lines = []
        for msg in messages:
            speaker = "用户" if msg["role"] == "user" else "助手"
            transcript_lines.append(f"{speaker}: {msg['content']}")
        transcript = "\n".join(transcript_lines)

        prompt = (
            "请把以下医药问答对话总结为长期记忆，使用中文 markdown 输出。\n"
            "要求：\n"
            "1. 保留用户的关键背景、已问过的重要问题、药品名称、结论和待跟进事项。\n"
            "2. 删除寒暄、重复内容和低价值细节。\n"
            "3. 输出尽量简洁，但信息完整。\n"
            "4. 如果已有历史摘要，请先融合，再输出更新后的完整摘要。\n\n"
            f"已有历史摘要：\n{self.summary_memory or '无'}\n\n"
            f"新增对话：\n{transcript}"
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return str(response.content).strip()
        except Exception:
            bullets = []
            if self.summary_memory:
                bullets.append(self.summary_memory.strip())
            for msg in messages[-6:]:
                speaker = "用户" if msg["role"] == "user" else "助手"
                bullets.append(f"- {speaker}：{msg['content'].strip()}")
            return "\n".join(bullets).strip()

    def _maybe_rollup_memory(self) -> None:
        trigger = APP_CONFIG["memory_summary_trigger_messages"]
        keep_recent = APP_CONFIG["memory_recent_messages"]

        if len(self.conversation_history) <= trigger:
            return

        old_messages = self.conversation_history[:-keep_recent]
        recent_messages = self.conversation_history[-keep_recent:]

        self.summary_memory = self._generate_summary_text(old_messages)
        self.conversation_history = recent_messages
        self._save_memory()

    def init_retriever(self) -> None:
        embedding_api_key = self.api_key if self.provider == EMBEDDING_PROVIDER else ""
        retriever = create_retriever(
            embedding_provider=EMBEDDING_PROVIDER,
            embedding_api_key=embedding_api_key,
        )
        self.retriever = retriever
        self.retriever_error = retriever.init_error

    def _retrieve_context(self, user_input: str, k: int = 4) -> tuple[str, List[Document]]:
        if not self.retriever:
            return "", []

        docs = self.retriever.similarity_search(user_input, k=k)
        if not docs:
            return "", []

        context = "\n\n".join(doc.page_content for doc in docs)
        return context, docs

    def _format_source_label(self, doc: Document) -> str:
        source = str(doc.metadata.get("source", "")).strip()
        source_file = str(doc.metadata.get("source_file", "")).strip()

        if source.startswith(("http://", "https://")):
            if source_file:
                return f"{source} (from {Path(source_file).name})"
            return source

        if source:
            return Path(source).name

        return "unknown source"

    def _append_sources(self, answer: str, docs: List[Document]) -> str:
        if not docs:
            return answer

        source_lines: List[str] = []
        seen = set()
        for doc in docs:
            label = self._format_source_label(doc)
            if label in seen:
                continue
            seen.add(label)
            source_lines.append(f"- {label}")

        if not source_lines:
            return answer

        return f"{str(answer).strip()}\n\n参考来源：\n" + "\n".join(source_lines)

    def _log_chat_event(
        self,
        *,
        user_input: str,
        answer: str,
        docs: List[Document],
        duration_ms: float,
        status: str,
        fallback_used: bool,
        error_type: str = "",
    ) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider": self.provider,
            "embedding_provider": EMBEDDING_PROVIDER,
            "question": user_input,
            "answer_length": len(str(answer)),
            "retrieved_doc_count": len(docs),
            "source_labels": [self._format_source_label(doc) for doc in docs],
            "knowledge_hit": bool(docs),
            "fallback_used": fallback_used,
            "status": status,
            "error_type": error_type,
            "duration_ms": round(duration_ms, 2),
        }
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _build_prompt(self, user_input: str, retrieved_context: str = "") -> List[Any]:
        messages: List[Any] = [SystemMessage(content=APP_CONFIG["system_prompt"])]

        if self.summary_memory:
            messages.append(
                SystemMessage(
                    content=(
                        "以下是本次会话的长期记忆摘要，请作为上下文参考：\n\n"
                        f"{self.summary_memory}"
                    )
                )
            )

        for msg in self.conversation_history[-APP_CONFIG["max_history"] :]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        if retrieved_context:
            context_prompt = (
                "以下是与用户问题相关的知识库补充内容。"
                "你应先直接回答用户问题，再吸收这些信息进行补充。"
                "如果使用了这些内容，请在回答中明确写出“根据知识库补充”。"
                "回答时不要过度简写，尽量给出完整且有条理的说明。\n\n"
                f"知识库补充内容：\n{retrieved_context}"
            )
            messages.append(SystemMessage(content=context_prompt))

        messages.append(HumanMessage(content=user_input))
        return messages

    def _to_openai_messages(
        self,
        user_input: str,
        retrieved_context: str = "",
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": APP_CONFIG["system_prompt"]}]

        if self.summary_memory:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是本次会话的长期记忆摘要，请作为上下文参考：\n\n"
                        f"{self.summary_memory}"
                    ),
                }
            )

        for msg in self.conversation_history[-APP_CONFIG["max_history"] :]:
            messages.append(
                {
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"],
                }
            )

        if retrieved_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是与用户问题相关的知识库补充内容。"
                        "你应先直接回答用户问题，再吸收这些信息进行补充。"
                        "如果使用了这些内容，请在回答中明确写出“根据知识库补充”。"
                        "回答时不要过度简写，尽量给出完整且有条理的说明。\n\n"
                        f"知识库补充内容：\n{retrieved_context}"
                    ),
                }
            )

        messages.append({"role": "user", "content": user_input})
        return messages

    def _invoke_with_raw_client(self, user_input: str, retrieved_context: str = "") -> str:
        response = self.raw_client.chat.completions.create(
            model=self.provider_config["model"],
            messages=self._to_openai_messages(user_input, retrieved_context),
            temperature=APP_CONFIG["temperature"],
            max_tokens=APP_CONFIG["max_tokens"],
        )
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else None
        if not content:
            raise RuntimeError("模型返回了空内容。")
        return str(content).strip()

    def _record_turn(self, user_input: str, answer: str) -> None:
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": answer})
        self._maybe_rollup_memory()
        self._save_memory()

    def _build_local_fallback_answer(self, user_input: str, docs: List[Document]) -> str:
        if not self.retriever:
            return (
                "抱歉，当前模型服务不可用，且本地知识库也未准备好，"
                "暂时无法回答这个问题。"
            )

        if not docs:
            return "抱歉，当前模型服务不可用，且本地知识库中没有找到明确相关内容。"

        context = "\n\n".join(doc.page_content for doc in docs)
        answer = (
            "当前模型服务暂不可用，以下内容根据本地知识库整理：\n\n"
            f"{context}\n\n"
            "以上为知识库补充信息，仅供参考；如涉及具体用药方案，请以医生或药师意见为准。"
        )
        return self._append_sources(answer, docs)

    def chat(self, user_input: str) -> str:
        started_at = time.perf_counter()
        retrieved_context, docs = self._retrieve_context(user_input)
        try:
            messages = self._build_prompt(user_input, retrieved_context)
            response = self.llm.invoke(messages)
            answer = self._append_sources(str(response.content).strip(), docs)
            self._record_turn(user_input, answer)
            self._log_chat_event(
                user_input=user_input,
                answer=answer,
                docs=docs,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                status="success",
                fallback_used=False,
            )
            return answer
        except (APIConnectionError, APITimeoutError):
            answer = self._build_local_fallback_answer(user_input, docs)
            self._record_turn(user_input, answer)
            self._log_chat_event(
                user_input=user_input,
                answer=answer,
                docs=docs,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                status="fallback",
                fallback_used=True,
                error_type="api_connection_or_timeout",
            )
            return answer
        except Exception as primary_exc:
            try:
                raw_answer = self._invoke_with_raw_client(user_input, retrieved_context)
                answer = self._append_sources(raw_answer, docs)
                self._record_turn(user_input, answer)
                self._log_chat_event(
                    user_input=user_input,
                    answer=answer,
                    docs=docs,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                    status="success_after_retry",
                    fallback_used=False,
                    error_type=type(primary_exc).__name__,
                )
                return answer
            except (APIConnectionError, APITimeoutError):
                answer = self._build_local_fallback_answer(user_input, docs)
                self._record_turn(user_input, answer)
                self._log_chat_event(
                    user_input=user_input,
                    answer=answer,
                    docs=docs,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                    status="fallback",
                    fallback_used=True,
                    error_type="api_connection_or_timeout",
                )
                return answer
            except Exception as exc:
                answer = f"抱歉，处理您的请求时发生错误：{exc}"
                self._log_chat_event(
                    user_input=user_input,
                    answer=answer,
                    docs=docs,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                    status="error",
                    fallback_used=False,
                    error_type=type(exc).__name__,
                )
                return answer

    def clear_history(self) -> None:
        self.conversation_history = []
        self.summary_memory = ""
        for path in (self.history_file, self.summary_file):
            if path.exists():
                path.unlink()

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history


def create_agent(provider: str = DEFAULT_PROVIDER, api_key: str = "") -> MedicalAgent:
    agent = MedicalAgent(provider, api_key)
    agent.init_retriever()
    return agent
