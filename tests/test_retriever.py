import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from rag.retriever import KnowledgeRetriever


class RetrieverTests(unittest.TestCase):
    def test_keyword_search_returns_matching_chunk(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.chunks = [
            Document(page_content="二甲双胍 随餐服用 可以减少胃肠道反应", metadata={}),
            Document(page_content="阿托伐他汀 夜间服用效果更好", metadata={}),
        ]

        docs = retriever._keyword_search("二甲双胍 随餐服用", k=2)

        self.assertTrue(docs)
        self.assertIn("二甲双胍", docs[0].page_content)

    def test_similarity_search_falls_back_when_vector_query_fails(self) -> None:
        retriever = KnowledgeRetriever.__new__(KnowledgeRetriever)
        retriever.vectorstore = Mock()
        retriever.vectorstore.similarity_search.side_effect = RuntimeError("rate limited")
        retriever.init_error = ""
        retriever.chunks = [Document(page_content="华法林 阿司匹林 出血风险", metadata={})]

        docs = retriever.similarity_search("华法林 阿司匹林", k=1)

        self.assertEqual(len(docs), 1)
        self.assertIn("出血风险", docs[0].page_content)
        self.assertIn("向量检索查询失败", retriever.init_error)


if __name__ == "__main__":
    unittest.main()
