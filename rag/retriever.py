# -*- coding: utf-8 -*-
"""
RAG retriever backed by local knowledge files and a persisted FAISS index.
"""

import csv
import hashlib
import json
import os
import plistlib
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from config import (
    APP_CONFIG,
    EMBEDDING_PROVIDER,
    KNOWLEDGE_DIR,
    SUPPORTED_KNOWLEDGE_EXTENSIONS,
    VECTORSTORE_DIR,
    VECTORSTORE_MANIFEST,
    get_api_key_for_provider,
    get_provider_config,
)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover
    DocxDocument = None


class KnowledgeRetriever:
    """Loads local knowledge files and provides vector or keyword retrieval."""

    def __init__(
        self,
        embedding_provider: str | None = None,
        embedding_api_key: str = "",
    ):
        self.embedding_provider = embedding_provider or EMBEDDING_PROVIDER
        self.embedding_api_key = (
            embedding_api_key
            or get_api_key_for_provider(self.embedding_provider, for_embedding=True)
        )
        self.vectorstore = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.init_error = ""
        self.index_dir = os.path.join(VECTORSTORE_DIR, self.embedding_provider)
        self.manifest_path = os.path.join(self.index_dir, VECTORSTORE_MANIFEST)
        self._init_vectorstore()

    def _get_embedding_model(self) -> OpenAIEmbeddings:
        config = get_provider_config(self.embedding_provider, for_embedding=True)
        return OpenAIEmbeddings(
            model=config["model"],
            api_key=self.embedding_api_key,
            base_url=config["api_base"],
            request_timeout=APP_CONFIG["request_timeout"],
            max_retries=APP_CONFIG["max_retries"],
        )

    def _knowledge_files(self) -> List[Path]:
        root = Path(KNOWLEDGE_DIR)
        if not root.exists():
            return []

        files: List[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_EXTENSIONS:
                files.append(path)
        files.sort()
        return files

    def _make_document(self, content: str, filepath: Path, file_type: str) -> Document:
        return Document(
            page_content=content,
            metadata={"source": str(filepath), "file_type": file_type},
        )

    def _load_text_file(self, filepath: Path) -> List[Document]:
        loader = TextLoader(str(filepath), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(filepath)
            doc.metadata["file_type"] = filepath.suffix.lower()
        return docs

    def _load_json_file(self, filepath: Path) -> List[Document]:
        content = filepath.read_text(encoding="utf-8")
        if filepath.suffix.lower() == ".jsonl":
            rows = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            normalized = json.dumps(rows, ensure_ascii=False, indent=2)
        else:
            normalized = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
        return [self._make_document(normalized, filepath, filepath.suffix.lower())]

    def _load_csv_file(self, filepath: Path) -> List[Document]:
        lines: List[str] = []
        with filepath.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                lines.append(", ".join(cell.strip() for cell in row))
        return [self._make_document("\n".join(lines), filepath, filepath.suffix.lower())]

    def _load_docx_file(self, filepath: Path) -> List[Document]:
        if DocxDocument is None:
            raise RuntimeError("python-docx is not installed")

        doc = DocxDocument(str(filepath))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))

        return [self._make_document("\n".join(paragraphs), filepath, filepath.suffix.lower())]

    def _extract_urls_from_file(self, filepath: Path) -> List[str]:
        suffix = filepath.suffix.lower()
        content = filepath.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".webloc":
            try:
                data = plistlib.loads(filepath.read_bytes())
                url = data.get("URL")
                return [url] if isinstance(url, str) and url else []
            except Exception:
                pass

        if suffix == ".url":
            urls = re.findall(r"^URL=(.+)$", content, flags=re.MULTILINE)
            if urls:
                return [url.strip() for url in urls if url.strip()]

        urls = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^https?://", line, flags=re.IGNORECASE):
                urls.append(line)
        return urls

    def _html_to_text(self, html: str) -> str:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)

        html = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
        html = re.sub(r"(?s)<[^>]+>", " ", html)
        html = re.sub(r"\s+", " ", html)
        return html.strip()

    def _fetch_url_content(self, url: str) -> str:
        request = Request(
            url,
            headers={"User-Agent": "medagent-rag-loader/1.0"},
        )
        with urlopen(request, timeout=APP_CONFIG["url_fetch_timeout"]) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="ignore")
        return self._html_to_text(html)

    def _load_url_file(self, filepath: Path) -> List[Document]:
        documents: List[Document] = []
        for url in self._extract_urls_from_file(filepath):
            try:
                text = self._fetch_url_content(url)
                if not text:
                    continue
                hostname = urlparse(url).netloc or url
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "source_file": str(filepath),
                            "file_type": filepath.suffix.lower(),
                            "host": hostname,
                        },
                    )
                )
            except Exception as exc:
                print(f"Failed to fetch URL {url}: {exc}")
        return documents

    def _load_single_file(self, filepath: Path) -> List[Document]:
        suffix = filepath.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return self._load_json_file(filepath)
        if suffix == ".csv":
            return self._load_csv_file(filepath)
        if suffix == ".docx":
            return self._load_docx_file(filepath)
        if suffix in {".url", ".webloc", ".urls"}:
            return self._load_url_file(filepath)
        return self._load_text_file(filepath)

    def _load_documents(self) -> List[Document]:
        documents: List[Document] = []
        for filepath in self._knowledge_files():
            try:
                documents.extend(self._load_single_file(filepath))
            except Exception as exc:
                print(f"Failed to load knowledge file {filepath.name}: {exc}")
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n",
        )
        return text_splitter.split_documents(documents)

    def _hash_files(self, files: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        for filepath in files:
            digest.update(str(filepath.relative_to(Path(KNOWLEDGE_DIR))).encode("utf-8"))
            digest.update(filepath.read_bytes())
        return digest.hexdigest()

    def _current_manifest(self) -> dict:
        files = self._knowledge_files()
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": get_provider_config(
                self.embedding_provider, for_embedding=True
            )["model"],
            "knowledge_signature": self._hash_files(files),
            "file_count": len(files),
            "loader_version": 2,
        }

    def _read_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        try:
            return json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_manifest(self, manifest: dict) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        Path(self.manifest_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_cached_vectorstore(self, embedding: OpenAIEmbeddings) -> FAISS | None:
        index_file = os.path.join(self.index_dir, "index.faiss")
        store_file = os.path.join(self.index_dir, "index.pkl")
        if not (os.path.exists(index_file) and os.path.exists(store_file)):
            return None
        try:
            return FAISS.load_local(
                self.index_dir,
                embedding,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            print(f"Failed to load cached vectorstore: {exc}")
            return None

    def _init_vectorstore(self) -> None:
        self.documents = self._load_documents()
        if not self.documents:
            self.init_error = "未找到可加载的知识库文件。"
            return

        self.chunks = self._split_documents(self.documents)

        if not self.embedding_api_key:
            self.init_error = (
                f"缺少 embedding provider '{self.embedding_provider}' 的 API Key，"
                "已退回本地关键词检索。"
            )
            return

        try:
            embedding = self._get_embedding_model()
            current_manifest = self._current_manifest()
            cached_manifest = self._read_manifest()

            if cached_manifest == current_manifest:
                self.vectorstore = self._load_cached_vectorstore(embedding)
                if self.vectorstore is not None:
                    return

            self.vectorstore = FAISS.from_documents(self.chunks, embedding)
            os.makedirs(self.index_dir, exist_ok=True)
            self.vectorstore.save_local(self.index_dir)
            self._write_manifest(current_manifest)
        except Exception as exc:
            self.init_error = f"初始化向量检索失败：{exc}。已退回本地关键词检索。"
            print(self.init_error)

    def _tokenize_query(self, query: str) -> List[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]{1,8}|[A-Za-z0-9_-]+", query.lower())
        return [token for token in tokens if token.strip()]

    def _keyword_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.chunks:
            return []

        tokens = self._tokenize_query(query)
        if not tokens:
            return self.chunks[:k]

        scored_chunks = []
        for doc in self.chunks:
            content = doc.page_content.lower()
            score = sum(content.count(token) for token in tokens)

            if score == 0:
                char_counter = Counter(content)
                score = sum(char_counter.get(token, 0) for token in tokens if len(token) == 1)

            if score > 0:
                scored_chunks.append((score, doc))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_chunks[:k]]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vectorstore is not None:
            try:
                return self.vectorstore.similarity_search(query, k=k)
            except Exception as exc:
                # Query-time embedding calls can still fail because of rate limits or
                # provider availability, so keep keyword fallback available.
                self.init_error = f"向量检索查询失败：{exc}。已退回本地关键词检索。"
                print(self.init_error)
                return self._keyword_search(query, k=k)
        return self._keyword_search(query, k=k)

    def get_context(self, query: str, k: int = 4) -> str:
        docs = self.similarity_search(query, k)
        if not docs:
            return "未在知识库中找到相关信息。"
        return "\n\n".join(doc.page_content for doc in docs)


def create_retriever(
    embedding_provider: str | None = None,
    embedding_api_key: str = "",
) -> KnowledgeRetriever:
    return KnowledgeRetriever(
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
    )
