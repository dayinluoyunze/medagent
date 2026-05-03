# -*- coding: utf-8 -*-
"""
Core medical agent chat flow with persisted markdown memory.
"""

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, OpenAI

from config import (
    APP_CONFIG,
    DEFAULT_PROVIDER,
    EMBEDDING_PROVIDER,
    CHAT_METRICS_FILENAME,
    LOG_DIR,
    MEDICAL_KNOWLEDGE_DIR,
    MEMORY_DIR,
    MEMORY_CURRENT_SESSION_FILENAME,
    MEMORY_HISTORY_FILENAME,
    MEMORY_SESSIONS_DIRNAME,
    MEMORY_SUMMARY_FILENAME,
    PERSONAL_KNOWLEDGE_DIR,
    get_api_key_for_provider,
    get_provider_config,
)
from rag.retriever import create_retriever


THINK_BLOCK_PATTERN = re.compile(r"(?is)<think\b[^>]*>.*?</think\s*>")
THINK_OPEN_PATTERN = re.compile(r"(?is)<think\b[^>]*>.*")
HTML_THINK_BLOCK_PATTERN = re.compile(r"(?is)&lt;think\b.*?&lt;/think\s*&gt;")
HTML_THINK_OPEN_PATTERN = re.compile(r"(?is)&lt;think\b.*")


class MedicalAgent:
    """Medical QA agent with optional retrieval augmentation and markdown memory."""

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        api_key: str = "",
        embedding_provider: str = EMBEDDING_PROVIDER,
        embedding_api_key: str = "",
    ):
        self.provider = provider
        self.api_key = api_key or get_api_key_for_provider(provider)
        self.embedding_provider = embedding_provider
        self.embedding_api_key = (
            embedding_api_key
            or get_api_key_for_provider(embedding_provider, for_embedding=True)
        )
        self.provider_config = get_provider_config(self.provider)
        self.llm = self._init_llm()
        self.raw_client = self._init_raw_client()
        self.retriever = None
        self.retriever_error = ""
        self.medical_retriever = None
        self.personal_retriever = None
        self.medical_retriever_error = ""
        self.personal_retriever_error = ""
        self.medical_knowledge_enabled = APP_CONFIG["medical_knowledge_enabled"]
        self.personal_knowledge_enabled = APP_CONFIG["personal_knowledge_enabled"]
        self.conversation_history: List[Dict[str, str]] = []
        self.summary_memory = ""
        self.memory_dir = Path(MEMORY_DIR)
        self.history_file = self.memory_dir / MEMORY_HISTORY_FILENAME
        self.summary_file = self.memory_dir / MEMORY_SUMMARY_FILENAME
        self.current_session_file = self.memory_dir / MEMORY_CURRENT_SESSION_FILENAME
        self.sessions_dir = self.memory_dir / MEMORY_SESSIONS_DIRNAME
        self.log_dir = Path(LOG_DIR)
        self.metrics_file = self.log_dir / CHAT_METRICS_FILENAME
        self.current_session_id = ""
        self.current_session_title = ""
        self._load_memory()

    @property
    def _uses_openai_compatible_raw_only(self) -> bool:
        return self.provider == "minimax"

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
        if not APP_CONFIG["memory_enabled"]:
            self.conversation_history = []
            self.summary_memory = ""
            self.current_session_id = ""
            self.current_session_title = ""
            return
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._load_current_session_meta()
        self.conversation_history = self._load_history_markdown()
        self.summary_memory = self._load_summary_markdown()
        self._ensure_session_initialized()

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _read_session_payload(self, session_id: str) -> Dict[str, Any]:
        return json.loads(self._session_path(session_id).read_text(encoding="utf-8"))

    def _new_session_id(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]

    def _default_session_title(self) -> str:
        return datetime.now().strftime("会话 %m-%d %H:%M")

    def _derive_session_title(self, messages: List[Dict[str, str]]) -> str:
        for message in messages:
            if message["role"] != "user":
                continue
            content = re.sub(r"\s+", " ", message["content"]).strip()
            if not content:
                continue
            return content[:28] + ("..." if len(content) > 28 else "")
        return self._default_session_title()

    def _load_current_session_meta(self) -> None:
        if not self.current_session_file.exists():
            self.current_session_id = ""
            self.current_session_title = ""
            return

        try:
            payload = json.loads(self.current_session_file.read_text(encoding="utf-8"))
        except Exception:
            self.current_session_id = ""
            self.current_session_title = ""
            return

        self.current_session_id = str(payload.get("session_id", "")).strip()
        self.current_session_title = str(payload.get("title", "")).strip()

    def _write_current_session_meta(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.current_session_id,
            "title": self.current_session_title,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.current_session_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_session_payload(self) -> Dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        title = self._derive_session_title(self.conversation_history)
        if not self.current_session_title:
            self.current_session_title = title
        elif self.current_session_title == self._default_session_title() and self.conversation_history:
            self.current_session_title = title

        return {
            "session_id": self.current_session_id,
            "title": self.current_session_title or title,
            "provider": self.provider,
            "updated_at": now,
            "message_count": len(self.conversation_history),
            "summary_memory": self.summary_memory,
            "conversation_history": self.conversation_history,
        }

    def _write_session_snapshot(self) -> None:
        if not APP_CONFIG["memory_enabled"] or not self.current_session_id:
            return
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        payload = self._build_session_payload()
        self._session_path(self.current_session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_current_session_meta()

    def _ensure_session_initialized(self) -> None:
        if self.current_session_id:
            session_path = self._session_path(self.current_session_id)
            if session_path.exists():
                if not self.current_session_title:
                    try:
                        payload = json.loads(session_path.read_text(encoding="utf-8"))
                        self.current_session_title = str(payload.get("title", "")).strip()
                    except Exception:
                        self.current_session_title = ""
                return

        if self.conversation_history or self.summary_memory:
            self.current_session_id = self._new_session_id()
            self.current_session_title = self._derive_session_title(self.conversation_history)
            self._write_session_snapshot()
            return

        self.current_session_id = self._new_session_id()
        self.current_session_title = self._default_session_title()
        self._write_current_session_meta()

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
        if not APP_CONFIG["memory_enabled"]:
            return
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        history_lines = ["# Conversation History", ""]
        for msg in self.conversation_history:
            heading = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"## {heading}")
            history_lines.append(msg["content"].strip())
            history_lines.append("")
        self.history_file.write_text("\n".join(history_lines).strip() + "\n", encoding="utf-8")

        summary_lines = ["# Summary Memory", "", self.summary_memory.strip()]
        self.summary_file.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
        self._write_session_snapshot()

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
            return self._invoke_langchain_messages([HumanMessage(content=prompt)])
        except Exception:
            bullets = []
            if self.summary_memory:
                bullets.append(self.summary_memory.strip())
            for msg in messages[-6:]:
                speaker = "用户" if msg["role"] == "user" else "助手"
                bullets.append(f"- {speaker}：{msg['content'].strip()}")
            return "\n".join(bullets).strip()

    def _maybe_rollup_memory(self) -> None:
        if not APP_CONFIG["memory_enabled"]:
            return
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
        self.medical_retriever = create_retriever(
            embedding_provider=self.embedding_provider,
            embedding_api_key=self.embedding_api_key,
            knowledge_dir=MEDICAL_KNOWLEDGE_DIR,
            knowledge_base="medical",
            index_namespace="medical",
        )
        self.personal_retriever = create_retriever(
            embedding_provider=self.embedding_provider,
            embedding_api_key=self.embedding_api_key,
            knowledge_dir=PERSONAL_KNOWLEDGE_DIR,
            knowledge_base="personal",
            index_namespace="personal",
        )
        self.retriever = self.medical_retriever
        self.medical_retriever_error = self.medical_retriever.init_error
        self.personal_retriever_error = self.personal_retriever.init_error
        self.retriever_error = self._active_retriever_error()

    def set_knowledge_enabled(self, medical_enabled: bool, personal_enabled: bool) -> None:
        self.medical_knowledge_enabled = bool(medical_enabled)
        self.personal_knowledge_enabled = bool(personal_enabled)
        self.retriever_error = self._active_retriever_error()

    def _active_retriever_error(self) -> str:
        errors = []
        if getattr(self, "medical_knowledge_enabled", True) and self.medical_retriever_error:
            errors.append(f"医疗知识库：{self.medical_retriever_error}")
        if getattr(self, "personal_knowledge_enabled", False) and self.personal_retriever_error:
            errors.append(f"个人信息库：{self.personal_retriever_error}")
        return "；".join(errors)

    def _retrieve_context(self, user_input: str, k: int = 4) -> tuple[str, List[Document]]:
        sections: List[str] = []
        all_docs: List[Document] = []
        stores = [
            ("医疗知识库", self.medical_retriever, self.medical_knowledge_enabled),
            ("个人信息库", self.personal_retriever, self.personal_knowledge_enabled),
        ]

        for label, retriever, enabled in stores:
            if not enabled or not retriever:
                continue
            docs = retriever.similarity_search(user_input, k=k)
            if not docs:
                continue
            chunk_lines = []
            for doc in docs:
                source_label = self._format_source_label(doc)
                relevance_score = doc.metadata.get("relevance_score", "")
                matched_terms = ", ".join(str(term) for term in doc.metadata.get("matched_terms", [])[:5])
                evidence_header = f"[来源：{source_label}"
                if relevance_score != "":
                    evidence_header += f"｜相关性：{relevance_score}"
                if matched_terms:
                    evidence_header += f"｜命中：{matched_terms}"
                evidence_header += "]"
                chunk_lines.append(f"{evidence_header}\n{doc.page_content}")
            sections.append(
                f"【{label}】\n" + "\n\n".join(chunk_lines)
            )
            all_docs.extend(docs)

        return "\n\n".join(sections), all_docs

    def _assess_medical_risk(self, user_input: str) -> Dict[str, Any]:
        text = user_input.strip()
        flags: List[str] = []
        level = "normal"

        personal_markers = ["我", "本人", "家里人", "患者", "老人", "孩子", "宝宝", "孕妇", "我爸", "我妈"]
        urgent_markers = ["现在", "突然", "立刻", "紧急", "严重", "怎么办", "能不能马上", "要不要去医院"]
        emergency_terms = [
            "胸痛",
            "呼吸困难",
            "昏迷",
            "抽搐",
            "大出血",
            "休克",
            "严重过敏",
            "自杀",
            "中毒",
            "意识不清",
            "血氧低",
        ]
        diagnosis_terms = ["是不是", "确诊", "诊断", "什么病", "判断我", "能否判断", "属于什么病"]
        medication_adjustment_terms = [
            "停药",
            "加量",
            "减量",
            "换药",
            "联合用药",
            "一起吃",
            "怎么调整",
            "处方",
            "剂量怎么改",
        ]
        special_population_terms = ["孕妇", "备孕", "哺乳", "儿童", "小孩", "老人", "肝肾功能不全"]

        has_personal_context = any(token in text for token in personal_markers)
        has_urgent_context = any(token in text for token in urgent_markers)

        if any(token in text for token in special_population_terms):
            flags.append("special_population")

        if any(token in text for token in emergency_terms) and (has_personal_context or has_urgent_context):
            level = "emergency"
            flags.append("emergency")
        elif any(token in text for token in diagnosis_terms) and has_personal_context:
            level = "diagnosis"
            flags.append("diagnosis")
        elif any(token in text for token in medication_adjustment_terms) and has_personal_context:
            level = "personalized_medication"
            flags.append("personalized_medication")

        return {"level": level, "flags": flags}

    def _build_emergency_response(self) -> str:
        return (
            "这类情况存在潜在紧急风险，我不能替代医生做远程诊断或处置。\n\n"
            "建议你立即联系急救或尽快前往线下医疗机构；如果出现胸痛加重、呼吸困难、意识改变、抽搐、"
            "大出血或严重过敏等情况，请优先拨打急救电话或直接就医。\n\n"
            "如果你愿意，我可以继续帮你整理就医前需要准备的信息，例如当前症状、已用药物、既往病史和过敏史。"
        )

    def _build_guardrail_prompts(self, risk_assessment: Dict[str, Any]) -> List[str]:
        level = risk_assessment["level"]
        prompts: List[str] = []
        if level == "diagnosis":
            prompts.append(
                "用户正在请求个体化诊断判断。你可以提供一般性医学信息和就医建议，"
                "但不要给出确诊结论，也不要假装完成诊断。"
            )
        elif level == "personalized_medication":
            prompts.append(
                "用户正在请求个体化用药调整。你可以解释一般原则、风险点和应咨询的专业角色，"
                "但不要直接给出针对个人的处方调整方案。"
            )

        if "special_population" in risk_assessment["flags"]:
            prompts.append(
                "该问题涉及特殊人群用药，请在回答中明确提示更高风险和线下专业咨询建议。"
            )
        return prompts

    def _append_risk_notice(self, answer: str, risk_assessment: Dict[str, Any]) -> str:
        level = risk_assessment["level"]
        if level == "diagnosis":
            return (
                f"{answer.strip()}\n\n"
                "安全提示：以上仅为一般信息，不能替代线下诊断；如果你在描述的是本人或家属的实际症状，"
                "建议尽快由医生结合病史、查体和检查结果判断。"
            )
        if level == "personalized_medication":
            return (
                f"{answer.strip()}\n\n"
                "安全提示：涉及个人停药、加量、减量或换药时，不能仅凭线上信息决定，"
                "请结合原始处方、肝肾功能、合并用药和既往病史咨询医生或药师。"
            )
        return answer

    def _sanitize_model_output(self, text: str) -> str:
        sanitized = str(text or "")
        sanitized = HTML_THINK_BLOCK_PATTERN.sub("", sanitized)
        sanitized = HTML_THINK_OPEN_PATTERN.sub("", sanitized)
        sanitized = THINK_BLOCK_PATTERN.sub("", sanitized)
        sanitized = THINK_OPEN_PATTERN.sub("", sanitized)
        sanitized = re.sub(r"(?is)</think\s*>", "", sanitized)
        sanitized = re.sub(r"(?is)&lt;/think\s*&gt;", "", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
        return sanitized or "抱歉，模型没有返回可展示的答案。"

    def _sanitize_for_logging(self, text: str) -> str:
        if not APP_CONFIG["log_redact_questions"]:
            return text[: APP_CONFIG["log_question_max_chars"]]

        sanitized = text
        sanitized = re.sub(r"[\w.\-]+@[\w.\-]+\.\w+", "[EMAIL]", sanitized)
        sanitized = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE]", sanitized)
        sanitized = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[ID]", sanitized)
        sanitized = re.sub(r"\b\d{1,3}\s*岁\b", "[AGE]", sanitized)
        sanitized = re.sub(r"(?<!\d)\d{6,}(?!\d)", "[NUMBER]", sanitized)
        return sanitized[: APP_CONFIG["log_question_max_chars"]]

    def _format_source_label(self, doc: Document) -> str:
        source = str(doc.metadata.get("source", "")).strip()
        source_file = str(doc.metadata.get("source_file", "")).strip()
        knowledge_base = str(doc.metadata.get("knowledge_base", "")).strip()
        prefix = "个人信息库/" if knowledge_base == "personal" else ""

        if source.startswith(("http://", "https://")):
            if source_file:
                return f"{prefix}{source} (from {Path(source_file).name})"
            return f"{prefix}{source}"

        if source:
            return f"{prefix}{Path(source).name}"

        return f"{prefix}unknown source"

    def _append_sources(self, answer: str, docs: List[Document]) -> str:
        if not docs:
            return answer

        evidence_lines: List[str] = []
        source_lines: List[str] = []
        seen = set()
        seen_evidence = set()
        for doc in docs:
            label = self._format_source_label(doc)
            matched_terms = doc.metadata.get("matched_terms", [])
            relevance_score = doc.metadata.get("relevance_score", "")
            if (relevance_score != "" or matched_terms) and label not in seen_evidence:
                evidence_line = f"- {label}"
                if relevance_score != "":
                    evidence_line += f"：相关性 {relevance_score}"
                if matched_terms:
                    evidence_line += "，命中 " + "、".join(str(term) for term in matched_terms[:5])
                evidence_lines.append(evidence_line)
                seen_evidence.add(label)

            if label in seen:
                continue
            seen.add(label)
            excerpt = str(doc.metadata.get("excerpt", "")).strip()
            line = f"- {label}"
            if excerpt:
                line += f" | 片段：{excerpt}"
            source_lines.append(line)

        if not source_lines:
            return answer

        sections = [str(answer).strip()]
        if evidence_lines:
            sections.append("回答依据：\n" + "\n".join(evidence_lines))
        sections.append("参考来源：\n" + "\n".join(source_lines))
        return "\n\n".join(sections)

    def _base_system_prompt(self) -> str:
        if self.provider == "minimax":
            return (
                "你是专业医药知识助手。\n"
                "回答要求：\n"
                "1. 只提供一般性医药信息，不做诊断，不给个体化处方调整。\n"
                "2. 如果提供了知识库补充，优先参考，并在回答中明确写出“根据知识库补充”。\n"
                "3. 先给核心结论，再补充注意事项或就医建议。\n"
                "4. 涉及特殊人群、禁忌、不良反应或漏服处理时，回答要谨慎。\n"
                "5. 使用清晰、自然、专业的中文，避免空话和重复。"
            )
        return APP_CONFIG["system_prompt"].strip()

    def _build_system_prompt_text(
        self,
        retrieved_context: str = "",
        extra_system_prompts: List[str] | None = None,
    ) -> str:
        sections = [self._base_system_prompt()]
        sections.append(
            "输出限制：不要输出 <think>...</think>、chain-of-thought、隐藏思考过程或逐步推理；"
            "只输出用户可见的最终答案和必要依据。"
        )

        for prompt in extra_system_prompts or []:
            if prompt and prompt.strip():
                sections.append(prompt.strip())

        if self.summary_memory.strip():
            sections.append(
                "以下是本次会话的长期记忆摘要，请作为上下文参考：\n\n"
                f"{self.summary_memory.strip()}"
            )

        if retrieved_context:
            sections.append(
                "以下是与用户问题相关的知识库补充内容。"
                "你应先直接回答用户问题，再吸收这些信息进行补充。"
                "如果使用了这些内容，请在回答中明确写出“根据知识库补充”。"
                "只允许使用与用户问题主题、药品或疾病一致的片段；"
                "如果片段中混有其他药品、其他疾病或明显无关内容，必须忽略，不能据此引用。"
                "如果知识库证据不足，应明确说明未检索到足够相关证据，不要强行引用。"
                "不要输出逐步推理或隐藏思考过程，只输出面向用户的简洁依据。"
                "回答时不要过度简写，尽量给出完整且有条理的说明。\n\n"
                f"知识库补充内容：\n{retrieved_context}"
            )

        return "\n\n".join(sections).strip()

    def _langchain_messages_to_openai(
        self,
        messages: List[BaseMessage],
    ) -> List[Dict[str, str]]:
        converted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            converted.append({"role": role, "content": str(msg.content)})
        return converted

    def _build_completion_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.provider_config["model"],
            "messages": messages,
            "temperature": APP_CONFIG["temperature"],
        }
        if self.provider == "minimax":
            kwargs["max_completion_tokens"] = min(APP_CONFIG["max_tokens"], 1024)
        else:
            kwargs["max_tokens"] = APP_CONFIG["max_tokens"]
        return kwargs

    def _invoke_openai_compatible_messages(self, messages: List[Dict[str, str]]) -> str:
        response = self.raw_client.chat.completions.create(
            **self._build_completion_kwargs(messages)
        )
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else None
        if not content:
            raise RuntimeError("模型返回了空内容。")
        return str(content).strip()

    def _invoke_langchain_messages(self, messages: List[BaseMessage]) -> str:
        if self._uses_openai_compatible_raw_only:
            return self._invoke_openai_compatible_messages(
                self._langchain_messages_to_openai(messages)
            )

        response = self.llm.invoke(messages)
        return str(response.content).strip()

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
        risk_level: str = "normal",
        risk_flags: List[str] | None = None,
    ) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "provider": self.provider,
            "embedding_provider": getattr(self, "embedding_provider", EMBEDDING_PROVIDER),
            "question_redacted": self._sanitize_for_logging(user_input),
            "answer_length": len(str(answer)),
            "retrieved_doc_count": len(docs),
            "source_labels": [self._format_source_label(doc) for doc in docs],
            "knowledge_bases_hit": sorted(
                {
                    str(doc.metadata.get("knowledge_base", "medical"))
                    for doc in docs
                    if doc.metadata.get("knowledge_base", "medical")
                }
            ),
            "medical_knowledge_enabled": getattr(self, "medical_knowledge_enabled", True),
            "personal_knowledge_enabled": getattr(self, "personal_knowledge_enabled", False),
            "knowledge_hit": bool(docs),
            "fallback_used": fallback_used,
            "status": status,
            "error_type": error_type,
            "risk_level": risk_level,
            "risk_flags": risk_flags or [],
            "duration_ms": round(duration_ms, 2),
        }
        with self.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _build_prompt(
        self,
        user_input: str,
        retrieved_context: str = "",
        extra_system_prompts: List[str] | None = None,
    ) -> List[Any]:
        messages: List[Any] = [
            SystemMessage(
                content=self._build_system_prompt_text(
                    retrieved_context,
                    extra_system_prompts,
                )
            )
        ]

        for msg in self.conversation_history[-APP_CONFIG["max_history"] :]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_input))
        return messages

    def _to_openai_messages(
        self,
        user_input: str,
        retrieved_context: str = "",
        extra_system_prompts: List[str] | None = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self._build_system_prompt_text(
                    retrieved_context,
                    extra_system_prompts,
                ),
            }
        ]

        for msg in self.conversation_history[-APP_CONFIG["max_history"] :]:
            messages.append(
                {
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"],
                }
            )

        messages.append({"role": "user", "content": user_input})
        return messages

    def _invoke_with_raw_client(
        self,
        user_input: str,
        retrieved_context: str = "",
        extra_system_prompts: List[str] | None = None,
    ) -> str:
        return self._invoke_openai_compatible_messages(
            self._to_openai_messages(user_input, retrieved_context, extra_system_prompts)
        )

    def _record_turn(self, user_input: str, answer: str) -> None:
        answer = self._sanitize_model_output(answer)
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": answer})
        self._maybe_rollup_memory()
        self._save_memory()

    def _build_local_fallback_answer(self, user_input: str, docs: List[Document]) -> str:
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
        risk_assessment = self._assess_medical_risk(user_input)
        if risk_assessment["level"] == "emergency":
            answer = self._build_emergency_response()
            self._log_chat_event(
                user_input=user_input,
                answer=answer,
                docs=[],
                duration_ms=(time.perf_counter() - started_at) * 1000,
                status="guardrail_block",
                fallback_used=False,
                error_type="emergency_risk",
                risk_level=risk_assessment["level"],
                risk_flags=risk_assessment["flags"],
            )
            return answer

        extra_system_prompts = self._build_guardrail_prompts(risk_assessment)
        retrieved_context, docs = self._retrieve_context(user_input)
        try:
            messages = self._build_prompt(user_input, retrieved_context, extra_system_prompts)
            answer = self._append_risk_notice(
                self._sanitize_model_output(self._invoke_langchain_messages(messages)),
                risk_assessment,
            )
            answer = self._append_sources(answer, docs)
            self._record_turn(user_input, answer)
            self._log_chat_event(
                user_input=user_input,
                answer=answer,
                docs=docs,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                status="success",
                fallback_used=False,
                risk_level=risk_assessment["level"],
                risk_flags=risk_assessment["flags"],
            )
            return answer
        except (APIConnectionError, APITimeoutError):
            answer = self._build_local_fallback_answer(user_input, docs)
            answer = self._append_risk_notice(answer, risk_assessment)
            self._record_turn(user_input, answer)
            self._log_chat_event(
                user_input=user_input,
                answer=answer,
                docs=docs,
                duration_ms=(time.perf_counter() - started_at) * 1000,
                status="fallback",
                fallback_used=True,
                error_type="api_connection_or_timeout",
                risk_level=risk_assessment["level"],
                risk_flags=risk_assessment["flags"],
            )
            return answer
        except Exception as primary_exc:
            try:
                raw_answer = self._invoke_with_raw_client(
                    user_input,
                    retrieved_context,
                    extra_system_prompts,
                )
                answer = self._append_risk_notice(
                    self._sanitize_model_output(raw_answer),
                    risk_assessment,
                )
                answer = self._append_sources(answer, docs)
                self._record_turn(user_input, answer)
                self._log_chat_event(
                    user_input=user_input,
                    answer=answer,
                    docs=docs,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                    status="success_after_retry",
                    fallback_used=False,
                    error_type=type(primary_exc).__name__,
                    risk_level=risk_assessment["level"],
                    risk_flags=risk_assessment["flags"],
                )
                return answer
            except Exception as exc:
                if self._uses_openai_compatible_raw_only:
                    answer = self._build_local_fallback_answer(user_input, docs)
                    answer = self._append_risk_notice(answer, risk_assessment)
                    self._record_turn(user_input, answer)
                    self._log_chat_event(
                        user_input=user_input,
                        answer=answer,
                        docs=docs,
                        duration_ms=(time.perf_counter() - started_at) * 1000,
                        status="fallback",
                        fallback_used=True,
                        error_type=type(exc).__name__,
                        risk_level=risk_assessment["level"],
                        risk_flags=risk_assessment["flags"],
                    )
                    return answer
                raise
            except (APIConnectionError, APITimeoutError):
                answer = self._build_local_fallback_answer(user_input, docs)
                answer = self._append_risk_notice(answer, risk_assessment)
                self._record_turn(user_input, answer)
                self._log_chat_event(
                    user_input=user_input,
                    answer=answer,
                    docs=docs,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                    status="fallback",
                    fallback_used=True,
                    error_type="api_connection_or_timeout",
                    risk_level=risk_assessment["level"],
                    risk_flags=risk_assessment["flags"],
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
                    risk_level=risk_assessment["level"],
                    risk_flags=risk_assessment["flags"],
                )
                return answer

    def clear_history(self) -> None:
        self.conversation_history = []
        self.summary_memory = ""
        for path in (self.history_file, self.summary_file):
            if path.exists():
                path.unlink()
        self.current_session_id = self._new_session_id()
        self.current_session_title = self._default_session_title()
        if APP_CONFIG["memory_enabled"]:
            self._write_current_session_meta()

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def get_session_meta(self) -> Dict[str, Any]:
        return {
            "session_id": self.current_session_id,
            "title": self.current_session_title or self._default_session_title(),
            "provider": self.provider,
            "message_count": len(self.conversation_history),
            "summary_present": bool(self.summary_memory.strip()),
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        if not APP_CONFIG["memory_enabled"]:
            return []
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        sessions: List[Dict[str, Any]] = []
        for path in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            sessions.append(
                {
                    "session_id": str(payload.get("session_id", path.stem)),
                    "title": str(payload.get("title", path.stem)),
                    "updated_at": str(payload.get("updated_at", "")),
                    "message_count": int(payload.get("message_count", 0)),
                    "provider": str(payload.get("provider", self.provider)),
                }
            )

        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return sessions

    def load_session(self, session_id: str) -> None:
        payload = self._read_session_payload(session_id)
        self.current_session_id = str(payload.get("session_id", session_id))
        self.current_session_title = str(payload.get("title", "")).strip() or self._default_session_title()
        self.conversation_history = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in payload.get("conversation_history", [])
            if isinstance(item, dict) and "role" in item and "content" in item
        ]
        self.summary_memory = str(payload.get("summary_memory", "")).strip()
        self._save_memory()

    def start_new_session(self) -> None:
        if APP_CONFIG["memory_enabled"] and self.conversation_history:
            self._write_session_snapshot()
        self.clear_history()

    def rename_session(self, session_id: str, title: str) -> None:
        normalized_title = re.sub(r"\s+", " ", title).strip()
        if not normalized_title:
            raise ValueError("会话标题不能为空。")

        if session_id == self.current_session_id:
            self.current_session_title = normalized_title
            self._save_memory()
            return

        payload = self._read_session_payload(session_id)
        payload["title"] = normalized_title
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._session_path(session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete_session(self, session_id: str) -> None:
        session_path = self._session_path(session_id)
        if session_path.exists():
            session_path.unlink()

        if session_id != self.current_session_id:
            return

        self.conversation_history = []
        self.summary_memory = ""
        for path in (self.history_file, self.summary_file):
            if path.exists():
                path.unlink()
        self.current_session_id = self._new_session_id()
        self.current_session_title = self._default_session_title()
        if APP_CONFIG["memory_enabled"]:
            self._write_current_session_meta()


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    api_key: str = "",
    embedding_provider: str = EMBEDDING_PROVIDER,
    embedding_api_key: str = "",
) -> MedicalAgent:
    agent = MedicalAgent(provider, api_key, embedding_provider, embedding_api_key)
    agent.init_retriever()
    return agent
