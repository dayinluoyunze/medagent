import json
import unittest
from pathlib import Path
import shutil
import uuid

from langchain_core.documents import Document

from agents.medical_agent import MedicalAgent


class MedicalAgentTests(unittest.TestCase):
    def test_append_sources_deduplicates_and_formats_labels(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)
        docs = [
            Document(page_content="a", metadata={"source": "knowledge/products.md"}),
            Document(
                page_content="b",
                metadata={
                    "source": "https://example.com/a",
                    "source_file": "knowledge/sample.urls",
                },
            ),
            Document(page_content="c", metadata={"source": "knowledge/products.md"}),
        ]

        answer = agent._append_sources("测试回答", docs)

        self.assertIn("参考来源：", answer)
        self.assertIn("- products.md", answer)
        self.assertIn("- https://example.com/a (from sample.urls)", answer)
        self.assertEqual(answer.count("products.md"), 1)

    def test_build_local_fallback_answer_includes_sources(self) -> None:
        agent = MedicalAgent.__new__(MedicalAgent)
        agent.retriever = object()
        docs = [Document(page_content="知识片段", metadata={"source": "knowledge/drugs.md"})]

        answer = agent._build_local_fallback_answer("二甲双胍怎么吃", docs)

        self.assertIn("根据本地知识库整理", answer)
        self.assertIn("知识片段", answer)
        self.assertIn("- drugs.md", answer)

    def test_save_and_load_markdown_memory(self) -> None:
        tmpdir = Path("tests") / f".tmp_memory_{uuid.uuid4().hex}"
        try:
            agent = MedicalAgent.__new__(MedicalAgent)
            agent.memory_dir = tmpdir
            agent.history_file = agent.memory_dir / "conversation_history.md"
            agent.summary_file = agent.memory_dir / "conversation_summary.md"
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
        self.assertEqual(payload["status"], "success")


if __name__ == "__main__":
    unittest.main()
