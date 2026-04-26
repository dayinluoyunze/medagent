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

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
from rag.ocr import IMAGE_KNOWLEDGE_EXTENSIONS, image_file_to_text, pdf_file_to_ocr_text

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover
    DocxDocument = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None


class KnowledgeRetriever:
    """Loads local knowledge files and provides vector or keyword retrieval."""

    def __init__(
        self,
        embedding_provider: str | None = None,
        embedding_api_key: str = "",
        knowledge_dir: str | Path = KNOWLEDGE_DIR,
        knowledge_base: str = "medical",
        index_namespace: str | None = None,
    ):
        self.embedding_provider = embedding_provider or EMBEDDING_PROVIDER
        self.embedding_api_key = (
            embedding_api_key
            or get_api_key_for_provider(self.embedding_provider, for_embedding=True)
        )
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_base = knowledge_base
        self.index_namespace = index_namespace or self._safe_index_namespace(
            str(self.knowledge_dir)
        )
        self.vectorstore = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.init_error = ""
        self.index_dir = os.path.join(
            VECTORSTORE_DIR,
            self.embedding_provider,
            self.index_namespace,
        )
        self.manifest_path = os.path.join(self.index_dir, VECTORSTORE_MANIFEST)
        self._init_vectorstore()

    def _safe_index_namespace(self, value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
        return normalized or "knowledge"

    def _metadata_with_knowledge_base(self, metadata: dict) -> dict:
        enriched = dict(metadata)
        enriched.setdefault("knowledge_base", getattr(self, "knowledge_base", "medical"))
        enriched.setdefault(
            "knowledge_dir",
            str(getattr(self, "knowledge_dir", Path(KNOWLEDGE_DIR))),
        )
        return enriched

    def _allow_remote_url(self, url: str) -> bool:
        if not APP_CONFIG["allow_remote_knowledge_fetch"]:
            return False

        allowlist = APP_CONFIG["remote_knowledge_allowlist"]
        if not allowlist:
            return True

        hostname = (urlparse(url).netloc or "").lower()
        return any(hostname == host.lower() or hostname.endswith(f".{host.lower()}") for host in allowlist)

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
        root = Path(getattr(self, "knowledge_dir", KNOWLEDGE_DIR))
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
            metadata=self._metadata_with_knowledge_base(
                {"source": str(filepath), "file_type": file_type}
            ),
        )

    def _split_frontmatter(self, content: str) -> tuple[dict, str]:
        if not content.startswith("---\n"):
            return {}, content

        end_index = content.find("\n---", 4)
        if end_index < 0:
            return {}, content

        metadata: dict[str, str] = {}
        frontmatter = content[4:end_index]
        for line in frontmatter.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")

        body = content[end_index + 4 :].lstrip()
        return metadata, body

    def _load_text_file(self, filepath: Path) -> List[Document]:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        frontmatter, body = self._split_frontmatter(content)
        metadata = self._metadata_with_knowledge_base(
            {"source": str(filepath), "file_type": filepath.suffix.lower()}
        )

        source_url = str(frontmatter.get("medagent_source_url", "")).strip()
        if source_url.startswith(("http://", "https://")):
            metadata["source"] = source_url
            metadata["source_file"] = str(filepath)
            metadata["source_title"] = str(frontmatter.get("medagent_source_title", "")).strip()
            metadata["source_type"] = str(frontmatter.get("medagent_source_type", "")).strip()

        return [Document(page_content=body, metadata=metadata)]

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

    def _load_pdf_file(self, filepath: Path) -> List[Document]:
        if PdfReader is None:
            raise RuntimeError("pypdf is not installed")

        reader = PdfReader(str(filepath))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append(f"[page {index}]\n{text}")

        content = "\n\n".join(pages).strip()
        if not content:
            content = pdf_file_to_ocr_text(filepath)
            doc = self._make_document(content, filepath, filepath.suffix.lower())
            doc.metadata["ocr"] = True
            return [doc]

        return [self._make_document(content, filepath, filepath.suffix.lower())]

    def _load_image_file(self, filepath: Path) -> List[Document]:
        content = image_file_to_text(filepath)
        doc = self._make_document(content, filepath, filepath.suffix.lower())
        doc.metadata["ocr"] = True
        return [doc]

    def _load_html_file(self, filepath: Path) -> List[Document]:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        return [self._make_document(self._html_to_text(content), filepath, filepath.suffix.lower())]

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
        if not self._allow_remote_url(url):
            raise RuntimeError(f"remote knowledge host is not allowed: {url}")

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
            if not self._allow_remote_url(url):
                continue
            try:
                text = self._fetch_url_content(url)
                if not text:
                    continue
                hostname = urlparse(url).netloc or url
                documents.append(
                    Document(
                        page_content=text,
                        metadata=self._metadata_with_knowledge_base(
                            {
                                "source": url,
                                "source_file": str(filepath),
                                "file_type": filepath.suffix.lower(),
                                "host": hostname,
                            }
                        ),
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
        if suffix == ".pdf":
            return self._load_pdf_file(filepath)
        if suffix in IMAGE_KNOWLEDGE_EXTENSIONS:
            return self._load_image_file(filepath)
        if suffix in {".html", ".htm"}:
            return self._load_html_file(filepath)
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
        )
        return text_splitter.split_documents(documents)

    def _hash_files(self, files: Iterable[Path]) -> str:
        digest = hashlib.sha256()
        root = Path(getattr(self, "knowledge_dir", KNOWLEDGE_DIR))
        for filepath in files:
            try:
                relative_path = filepath.relative_to(root)
            except ValueError:
                relative_path = filepath
            digest.update(str(relative_path).encode("utf-8"))
            digest.update(filepath.read_bytes())
        return digest.hexdigest()

    def _current_manifest(self) -> dict:
        files = self._knowledge_files()
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": get_provider_config(
                self.embedding_provider, for_embedding=True
            )["model"],
            "knowledge_base": self.knowledge_base,
            "knowledge_dir": str(self.knowledge_dir),
            "index_namespace": self.index_namespace,
            "retrieval_mode": APP_CONFIG["retrieval_mode"],
            "allow_remote_knowledge_fetch": APP_CONFIG["allow_remote_knowledge_fetch"],
            "remote_knowledge_allowlist": APP_CONFIG["remote_knowledge_allowlist"],
            "ocr_enabled": APP_CONFIG["ocr_enabled"],
            "ocr_lang": APP_CONFIG["ocr_lang"],
            "ocr_dpi": APP_CONFIG["ocr_dpi"],
            "ocr_max_pages": APP_CONFIG["ocr_max_pages"],
            "knowledge_signature": self._hash_files(files),
            "file_count": len(files),
            "loader_version": 7,
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

        if self.embedding_provider == "none":
            self.init_error = ""
            return

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
        normalized = query.strip().lower()
        if not normalized:
            return []

        tokens: List[str] = []
        if jieba is not None:
            tokens.extend(
                token.strip().lower()
                for token in jieba.lcut(normalized)
                if token.strip()
            )

        regex_tokens = re.findall(r"[\u4e00-\u9fff]{1,8}|[A-Za-z0-9_-]+", normalized)
        tokens.extend(regex_tokens)

        deduped: List[str] = []
        seen = set()
        for token in tokens:
            if len(token) <= 1 and not re.match(r"[A-Za-z0-9]", token):
                continue
            if token not in seen:
                seen.add(token)
                deduped.append(token)
        return deduped

    def _keyword_score(self, query: str, content: str, tokens: List[str]) -> int:
        score = 0
        if query.lower() in content:
            score += 10

        for token in tokens:
            occurrences = content.count(token)
            if not occurrences:
                continue
            token_weight = 1
            if len(token) >= 4:
                token_weight = 3
            elif len(token) >= 2:
                token_weight = 2
            score += occurrences * token_weight

        if score == 0:
            char_counter = Counter(content)
            score = sum(char_counter.get(token, 0) for token in tokens if len(token) == 1)

        return score

    def _build_excerpt(self, content: str, tokens: List[str]) -> str:
        flat_content = re.sub(r"\s+", " ", content).strip()
        if not flat_content:
            return ""

        for token in sorted(tokens, key=len, reverse=True):
            index = flat_content.lower().find(token.lower())
            if index >= 0:
                half_window = max(20, APP_CONFIG["citation_snippet_length"] // 2)
                start = max(0, index - half_window)
                end = min(len(flat_content), index + len(token) + half_window)
                snippet = flat_content[start:end].strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(flat_content):
                    snippet += "..."
                return snippet

        return flat_content[: APP_CONFIG["citation_snippet_length"]].strip()

    def _decorate_docs(
        self,
        query: str,
        docs: List[Document],
        *,
        retrieval_mode: str,
    ) -> List[Document]:
        tokens = self._tokenize_query(query)
        decorated: List[Document] = []
        for doc in docs:
            metadata = dict(doc.metadata)
            metadata["retrieval_mode"] = retrieval_mode
            metadata["excerpt"] = self._build_excerpt(doc.page_content, tokens)
            decorated.append(Document(page_content=doc.page_content, metadata=metadata))
        return decorated

    def _merge_docs(
        self,
        primary_docs: List[Document],
        secondary_docs: List[Document],
        *,
        k: int,
    ) -> List[Document]:
        merged: List[Document] = []
        seen = set()
        for doc in primary_docs + secondary_docs:
            key = self._doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= k:
                break
        return merged

    def _doc_key(self, doc: Document) -> tuple[str, str, str]:
        return (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("source_file", "")),
            doc.page_content[:200],
        )

    def _rank_hybrid_docs(
        self,
        query: str,
        vector_docs: List[Document],
        keyword_docs: List[Document],
        *,
        k: int,
    ) -> List[Document]:
        tokens = self._tokenize_query(query)
        scores: dict[tuple[str, str, str], float] = {}
        docs: dict[tuple[str, str, str], Document] = {}

        for rank, doc in enumerate(vector_docs, start=1):
            key = self._doc_key(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 2.5 / rank

        for rank, doc in enumerate(keyword_docs, start=1):
            key = self._doc_key(doc)
            docs.setdefault(key, doc)
            content = doc.page_content.lower()
            keyword_score = self._keyword_score(query, content, tokens)
            scores[key] = scores.get(key, 0.0) + 1.8 / rank + min(keyword_score / 6.0, 8.0)

        ranked_keys = sorted(scores, key=lambda item: scores[item], reverse=True)
        ranked_docs: List[Document] = []
        for key in ranked_keys[:k]:
            metadata = dict(docs[key].metadata)
            metadata["rerank_score"] = round(scores[key], 4)
            ranked_docs.append(Document(page_content=docs[key].page_content, metadata=metadata))
        return ranked_docs

    def _keyword_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.chunks:
            return []

        tokens = self._tokenize_query(query)
        if not tokens:
            return self.chunks[:k]

        scored_chunks = []
        for doc in self.chunks:
            content = doc.page_content.lower()
            score = self._keyword_score(query, content, tokens)

            if score > 0:
                scored_chunks.append((score, doc))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_chunks[:k]]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        retrieval_mode = APP_CONFIG["retrieval_mode"]
        candidate_k = max(k * 3, k)
        keyword_docs = self._keyword_search(query, k=candidate_k)

        if self.vectorstore is not None and retrieval_mode in {"auto", "vector", "hybrid"}:
            try:
                vector_docs = self.vectorstore.similarity_search(query, k=candidate_k)
                if retrieval_mode == "vector":
                    return self._decorate_docs(query, vector_docs[:k], retrieval_mode="vector")

                merged_docs = self._rank_hybrid_docs(query, vector_docs, keyword_docs, k=k)
                mode = "hybrid_rerank" if keyword_docs else "vector"
                return self._decorate_docs(query, merged_docs, retrieval_mode=mode)
            except Exception as exc:
                # Query-time embedding calls can still fail because of rate limits or
                # provider availability, so keep keyword fallback available.
                self.init_error = f"向量检索查询失败：{exc}。已退回本地关键词检索。"
                print(self.init_error)
                return self._decorate_docs(query, keyword_docs[:k], retrieval_mode="keyword")

        return self._decorate_docs(query, keyword_docs[:k], retrieval_mode="keyword")

    def get_context(self, query: str, k: int = 4) -> str:
        docs = self.similarity_search(query, k)
        if not docs:
            return "未在知识库中找到相关信息。"
        return "\n\n".join(doc.page_content for doc in docs)


def create_retriever(
    embedding_provider: str | None = None,
    embedding_api_key: str = "",
    knowledge_dir: str | Path = KNOWLEDGE_DIR,
    knowledge_base: str = "medical",
    index_namespace: str | None = None,
) -> KnowledgeRetriever:
    return KnowledgeRetriever(
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
        knowledge_dir=knowledge_dir,
        knowledge_base=knowledge_base,
        index_namespace=index_namespace,
    )
