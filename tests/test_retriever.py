import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import rag.retriever as retriever_module
from langchain_core.documents import Document

from config import APP_CONFIG
from rag.retriever import KnowledgeRetriever


class RetrieverTests(unittest.TestCase):
    def test_load_pdf_file_extracts_page_text(self) -> None:
        class FakePage:
            def __init__(self, text: str):
                self.text = text

            def extract_text(self) -> str:
                return self.text

        class FakePdfReader:
            def __init__(self, _: str):
                self.pages = [FakePage("二甲双胍 随餐服用"), FakePage("")]

        original_reader = retriever_module.PdfReader
        retriever_module.PdfReader = FakePdfReader
        try:
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "guide.pdf"
                path.write_bytes(b"%PDF-1.4")

                retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
                docs = retriever._load_pdf_file(path)

                self.assertEqual(docs[0].metadata["file_type"], ".pdf")
                self.assertIn("[page 1]", docs[0].page_content)
                self.assertIn("二甲双胍", docs[0].page_content)
        finally:
            retriever_module.PdfReader = original_reader

    def test_load_pdf_file_falls_back_to_ocr_when_text_layer_is_empty(self) -> None:
        class FakePage:
            def extract_text(self) -> str:
                return ""

        class FakePdfReader:
            def __init__(self, _: str):
                self.pages = [FakePage()]

        original_reader = retriever_module.PdfReader
        original_ocr = retriever_module.pdf_file_to_ocr_text
        retriever_module.PdfReader = FakePdfReader
        retriever_module.pdf_file_to_ocr_text = lambda _: "[ocr page 1]\n二甲双胍 OCR 文本"
        try:
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "scan.pdf"
                path.write_bytes(b"%PDF-1.4")

                retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
                docs = retriever._load_pdf_file(path)

                self.assertTrue(docs[0].metadata["ocr"])
                self.assertIn("OCR 文本", docs[0].page_content)
        finally:
            retriever_module.PdfReader = original_reader
            retriever_module.pdf_file_to_ocr_text = original_ocr

    def test_load_image_file_uses_ocr_text(self) -> None:
        original_ocr = retriever_module.image_file_to_text
        retriever_module.image_file_to_text = lambda _: "图片 OCR 说明书内容"
        try:
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "scan.png"
                path.write_bytes(b"image")

                retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
                docs = retriever._load_image_file(path)

                self.assertTrue(docs[0].metadata["ocr"])
                self.assertEqual(docs[0].metadata["file_type"], ".png")
                self.assertIn("说明书", docs[0].page_content)
        finally:
            retriever_module.image_file_to_text = original_ocr

    def test_load_text_file_uses_url_frontmatter_as_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "url-snapshot.md"
            path.write_text(
                "---\n"
                "medagent_source_type: url\n"
                "medagent_source_url: https://example.com/guide\n"
                "medagent_source_title: 指南\n"
                "---\n\n"
                "# 指南\n\n二甲双胍随餐服用。\n",
                encoding="utf-8",
            )

            retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
            docs = retriever._load_text_file(path)

            self.assertEqual(docs[0].metadata["source"], "https://example.com/guide")
            self.assertEqual(docs[0].metadata["source_file"], str(path))
            self.assertNotIn("medagent_source_url", docs[0].page_content)

    def test_custom_knowledge_dir_and_base_are_recorded_in_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.md"
            path.write_text("过敏史：青霉素。", encoding="utf-8")

            retriever = KnowledgeRetriever(
                embedding_provider="none",
                knowledge_dir=tmpdir,
                knowledge_base="personal",
                index_namespace="test-personal",
            )

            self.assertEqual(len(retriever.documents), 1)
            self.assertEqual(retriever.documents[0].metadata["knowledge_base"], "personal")
            self.assertEqual(retriever.documents[0].metadata["knowledge_dir"], str(Path(tmpdir)))
            self.assertIn("过敏史", retriever.documents[0].page_content)

    def test_keyword_search_returns_matching_chunk(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.chunks = [
            Document(page_content="二甲双胍 随餐服用 可以减少胃肠道反应", metadata={}),
            Document(page_content="阿托伐他汀 夜间服用效果更好", metadata={}),
        ]

        docs = retriever._keyword_search("二甲双胍 随餐服用", k=2)

        self.assertTrue(docs)
        self.assertIn("二甲双胍", docs[0].page_content)

    def test_markdown_sections_keep_faq_items_separate(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        doc = Document(
            page_content=(
                "# FAQ\n\n"
                "## 产品相关\n\n"
                "### Q1: 厄贝沙坦片和缬沙坦有什么区别？\n"
                "A: 两者都属于ARB类降压药。\n\n"
                "### Q2: 二甲双胍片可以长期服用吗？\n"
                "A: 二甲双胍是2型糖尿病的一线用药。"
            ),
            metadata={"source": "knowledge/qa.md", "file_type": ".md"},
        )

        sections = retriever._split_document_by_markdown_sections(doc)

        self.assertTrue(any("Q1" in section.page_content for section in sections))
        q1_section = next(section for section in sections if "Q1" in section.page_content)
        self.assertNotIn("二甲双胍", q1_section.page_content)

    def test_similarity_search_filters_lower_overlap_sibling_section(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.vectorstore = None
        retriever.init_error = ""
        retriever.chunks = [
            Document(
                page_content="## 药品A（降压药）\n### 不良反应\n常见：头晕、头痛。",
                metadata={"source": "knowledge/products.md", "file_type": ".md"},
            ),
            Document(
                page_content="## 药品B（降糖药）\n### 不良反应\n常见：胃肠道反应。",
                metadata={"source": "knowledge/products.md", "file_type": ".md"},
            ),
        ]

        docs = retriever.similarity_search("降压药常见副作用有哪些？", k=4)

        self.assertEqual(len(docs), 1)
        self.assertIn("降压药", docs[0].page_content)
        self.assertNotIn("降糖药", docs[0].page_content)

    def test_similarity_search_supports_new_knowledge_without_hardcoded_entities(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.vectorstore = None
        retriever.init_error = ""
        retriever.chunks = [
            Document(
                page_content="## 新药X（抗眩晕药）\n### 不良反应\n常见：嗜睡、口干。",
                metadata={"source": "knowledge/new.md", "file_type": ".md"},
            ),
            Document(
                page_content="## 新药Y（抗过敏药）\n### 不良反应\n常见：皮疹。",
                metadata={"source": "knowledge/new.md", "file_type": ".md"},
            ),
        ]

        docs = retriever.similarity_search("新药X常见副作用有哪些？", k=4)

        self.assertEqual(len(docs), 1)
        self.assertIn("新药X", docs[0].page_content)
        self.assertNotIn("新药Y", docs[0].page_content)

    def test_similarity_search_falls_back_when_vector_query_fails(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.vectorstore = Mock()
        retriever.vectorstore.similarity_search.side_effect = RuntimeError("rate limited")
        retriever.init_error = ""
        retriever.chunks = [Document(page_content="华法林 阿司匹林 出血风险", metadata={})]

        original_mode = APP_CONFIG["retrieval_mode"]
        APP_CONFIG["retrieval_mode"] = "auto"
        try:
            docs = retriever.similarity_search("华法林 阿司匹林", k=1)
        finally:
            APP_CONFIG["retrieval_mode"] = original_mode

        self.assertEqual(len(docs), 1)
        self.assertIn("出血风险", docs[0].page_content)
        self.assertIn("向量检索查询失败", retriever.init_error)

    def test_hybrid_search_reranks_keyword_exact_match_above_vector_rank(self) -> None:
        irrelevant_doc = Document(page_content="阿托伐他汀 夜间服用效果更好", metadata={"source": "a.md"})
        relevant_doc = Document(page_content="二甲双胍 随餐服用 可以减少胃肠道反应", metadata={"source": "b.md"})

        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.vectorstore = Mock()
        retriever.vectorstore.similarity_search.return_value = [irrelevant_doc, relevant_doc]
        retriever.init_error = ""
        retriever.chunks = [relevant_doc]

        original_mode = APP_CONFIG["retrieval_mode"]
        APP_CONFIG["retrieval_mode"] = "hybrid"
        try:
            docs = retriever.similarity_search("二甲双胍 随餐服用", k=1)
        finally:
            APP_CONFIG["retrieval_mode"] = original_mode

        self.assertEqual(docs[0].metadata["source"], "b.md")
        self.assertEqual(docs[0].metadata["retrieval_mode"], "hybrid_rerank")
        self.assertIn("rerank_score", docs[0].metadata)

    def test_similarity_search_adds_excerpt_metadata_in_keyword_mode(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.vectorstore = None
        retriever.init_error = ""
        retriever.chunks = [
            Document(page_content="二甲双胍 随餐服用 可以减少胃肠道反应", metadata={"source": "knowledge/products.md"})
        ]

        original_mode = APP_CONFIG["retrieval_mode"]
        APP_CONFIG["retrieval_mode"] = "keyword"
        try:
            docs = retriever.similarity_search("二甲双胍 随餐服用", k=1)
        finally:
            APP_CONFIG["retrieval_mode"] = original_mode

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["retrieval_mode"], "keyword")
        self.assertIn("二甲双胍", docs[0].metadata["excerpt"])


if __name__ == "__main__":
    unittest.main()
