import json
import unittest
from pathlib import Path
import shutil
import uuid

from langchain_core.documents import Document

from agents.medical_agent import MedicalAgent


class MedicalAgentTests(unittest.TestCase):
    def test_assess_medical_risk_detects_emergency(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)

        risk = agent._assess_medical_risk("我现在胸痛还呼吸困难，要不要立刻去医院？")

        self.assertEqual(risk["level"], "emergency")
        self.assertIn("emergency", risk["flags"])

    def test_sanitize_for_logging_redacts_common_sensitive_patterns(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)

        sanitized = agent._sanitize_for_logging(
            "我叫张三，28岁，手机号13812345678，邮箱test@example.com，身份证110105199901011234"
        )

        self.assertIn("[PHONE]", sanitized)
        self.assertIn("[EMAIL]", sanitized)
        self.assertIn("[ID]", sanitized)
        self.assertIn("[AGE]", sanitized)

    def test_append_sources_deduplicates_and_formats_labels(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)
        docs = [
            Document(
                page_content="a",
                metadata={"source": "knowledge/products.md", "excerpt": "二甲双胍 随餐服用"},
            ),
            Document(
                page_content="b",
                metadata={
                    "source": "https://example.com/a",
                    "source_file": "knowledge/sample.urls",
                    "excerpt": "网页知识片段",
                },
            ),
            Document(page_content="c", metadata={"source": "knowledge/products.md"}),
        ]

        answer = agent._append_sources("测试回答", docs)

        self.assertIn("参考来源：", answer)
        self.assertIn("- products.md | 片段：二甲双胍 随餐服用", answer)
        self.assertIn("- https://example.com/a (from sample.urls) | 片段：网页知识片段", answer)
        self.assertEqual(answer.count("products.md"), 1)

    def test_build_local_fallback_answer_includes_sources(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)
        agent.retriever = object()
        docs = [Document(page_content="知识片段", metadata={"source": "knowledge/drugs.md"})]

        answer = agent._build_local_fallback_answer("二甲双胍怎么吃", docs)

        self.assertIn("根据本地知识库整理", answer)
        self.assertIn("知识片段", answer)
        self.assertIn("- drugs.md", answer)

    def test_retrieve_context_respects_independent_knowledge_switches(self) -> None:
        class FakeRetriever:
            def __init__(self, docs: list[Document]):
                self.docs = docs
                self.calls = 0

            def similarity_search(self, query: str, k: int = 4) -> list[Document]:
                self.calls += 1
                return self.docs

        agent = MedicalAgent.__new__(MedicalAgent)
        medical = FakeRetriever(
            [Document(page_content="医疗知识片段", metadata={"knowledge_base": "medical"})]
        )
        personal = FakeRetriever(
            [Document(page_content="个人信息片段", metadata={"knowledge_base": "personal"})]
        )
        agent.medical_retriever = medical
        agent.personal_retriever = personal
        agent.medical_knowledge_enabled = True
        agent.personal_knowledge_enabled = False

        context, docs = agent._retrieve_context("问题")

        self.assertIn("【医疗知识库】", context)
        self.assertIn("医疗知识片段", context)
        self.assertNotIn("个人信息片段", context)
        self.assertEqual(len(docs), 1)
        self.assertEqual(medical.calls, 1)
        self.assertEqual(personal.calls, 0)

        agent.personal_knowledge_enabled = True
        context, docs = agent._retrieve_context("问题")

        self.assertIn("【个人信息库】", context)
        self.assertIn("个人信息片段", context)
        self.assertEqual(len(docs), 2)

    def test_personal_source_label_is_explicitly_prefixed(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)
        doc = Document(
            page_content="个人信息",
            metadata={"source": "personal_knowledge/uploads/profile.md", "knowledge_base": "personal"},
        )

        self.assertEqual(agent._format_source_label(doc), "个人信息库/profile.md")

    def test_to_openai_messages_merges_system_context_for_provider_compatibility(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)
        agent.provider = "minimax"
        agent.summary_memory = "- 用户长期服用二甲双胍"
        agent.conversation_history = [
            {"role": "user", "content": "之前问过二甲双胍"},
            {"role": "assistant", "content": "建议随餐服用"},
        ]

        messages = agent._to_openai_messages(
            "二甲双胍应该饭前吃还是饭后吃？",
            "二甲双胍建议随餐服用。",
            ["涉及特殊人群时请更谨慎。"],
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(sum(msg["role"] == "system" for msg in messages), 1)
        self.assertIn("涉及特殊人群时请更谨慎。", messages[0]["content"])
        self.assertIn("用户长期服用二甲双胍", messages[0]["content"])
        self.assertIn("知识库补充内容", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[-1]["role"], "user")

    def test_save_and_load_markdown_memory(self) -> None:
        tmpdir = Path("tests") / f".tmp_memory_{uuid.uuid4().hex}"
        try:
            agent = MedicalAgent.__new__(MedicalAgent)
            agent.memory_dir = tmpdir
            agent.history_file = agent.memory_dir / "conversation_history.md"
            agent.summary_file = agent.memory_dir / "conversation_summary.md"
            agent.sessions_dir = agent.memory_dir / "sessions"
            agent.current_session_file = agent.memory_dir / "current_session.json"
            agent.current_session_id = "session-1"
            agent.current_session_title = "问题A"
            agent.provider = "minimax"
            agent.conversation_history = [
                {"role": "user", "content": "问题A"},
                {"role": "assistant", "content": "回答A"},
            ]
            agent.summary_memory = "- 历史摘要"

            agent._save_memory()

            loaded_history = agent._load_history_markdown()
            loaded_summary = agent._load_summary_markdown()
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

        self.assertEqual(loaded_history, agent.conversation_history)
        self.assertEqual(loaded_summary, "- 历史摘要")

    def test_session_snapshot_can_be_listed_and_loaded(self) -> None:
        tmpdir = Path("tests") / f".tmp_sessions_{uuid.uuid4().hex}"
        try:
            agent = MedicalAgent.__new__(MedicalAgent)
            agent.memory_dir = tmpdir
            agent.history_file = agent.memory_dir / "conversation_history.md"
            agent.summary_file = agent.memory_dir / "conversation_summary.md"
            agent.sessions_dir = agent.memory_dir / "sessions"
            agent.current_session_file = agent.memory_dir / "current_session.json"
            agent.current_session_id = "session-1"
            agent.current_session_title = "二甲双胍怎么吃"
            agent.provider = "minimax"
            agent.conversation_history = [
                {"role": "user", "content": "二甲双胍怎么吃"},
                {"role": "assistant", "content": "建议随餐服用"},
            ]
            agent.summary_memory = "- 关注用法用量"

            agent._save_memory()
            sessions = agent.list_sessions()

            agent.conversation_history = []
            agent.summary_memory = ""
            agent.current_session_id = ""
            agent.current_session_title = ""
            agent.load_session("session-1")
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "session-1")
        self.assertEqual(agent.current_session_title, "二甲双胍怎么吃")
        self.assertEqual(agent.conversation_history[0]["content"], "二甲双胍怎么吃")
        self.assertEqual(agent.summary_memory, "- 关注用法用量")

    def test_session_can_be_renamed_and_deleted(self) -> None:
        tmpdir = Path("tests") / f".tmp_sessions_{uuid.uuid4().hex}"
        try:
            agent = MedicalAgent.__new__(MedicalAgent)
            agent.memory_dir = tmpdir
            agent.history_file = agent.memory_dir / "conversation_history.md"
            agent.summary_file = agent.memory_dir / "conversation_summary.md"
            agent.sessions_dir = agent.memory_dir / "sessions"
            agent.current_session_file = agent.memory_dir / "current_session.json"
            agent.current_session_id = "session-1"
            agent.current_session_title = "初始标题"
            agent.provider = "minimax"
            agent.conversation_history = [
                {"role": "user", "content": "初始问题"},
                {"role": "assistant", "content": "初始回答"},
            ]
            agent.summary_memory = ""

            agent._save_memory()
            agent.rename_session("session-1", "重命名后的会话")
            renamed = agent.list_sessions()
            agent.delete_session("session-1")
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

        self.assertEqual(renamed[0]["title"], "重命名后的会话")
        self.assertEqual(agent.current_session_title.startswith("会话 "), True)
        self.assertEqual(agent.conversation_history, [])

    def test_log_chat_event_writes_metrics_jsonl(self) -> None:
        tmpdir = Path("tests") / f".tmp_logs_{uuid.uuid4().hex}"
        try:
            agent = MedicalAgent.__new__(MedicalAgent)
            agent.provider = "openai"
            agent.log_dir = tmpdir
            agent.metrics_file = tmpdir / "chat_metrics.jsonl"
            docs = [Document(page_content="片段", metadata={"source": "knowledge/products.md"})]

            agent._log_chat_event(
                user_input="二甲双胍怎么吃",
                answer="测试回答",
                docs=docs,
                duration_ms=123.456,
                status="success",
                fallback_used=False,
            )

            lines = agent.metrics_file.read_text(encoding="utf-8").splitlines()
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["provider"], "openai")
        self.assertTrue(payload["knowledge_hit"])
        self.assertEqual(payload["retrieved_doc_count"], 1)
        self.assertEqual(payload["source_labels"], ["products.md"])
        self.assertEqual(payload["knowledge_bases_hit"], ["medical"])
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["question_redacted"], "二甲双胍怎么吃")


if __name__ == "__main__":
    unittest.main()
