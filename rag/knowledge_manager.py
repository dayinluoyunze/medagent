# -*- coding: utf-8 -*-
"""
Utilities for adding knowledge files from the web UI.
"""

from __future__ import annotations

import re
import socket
import uuid
from io import BytesIO
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from config import APP_CONFIG, KNOWLEDGE_DIR, SUPPORTED_KNOWLEDGE_EXTENSIONS
from rag.ocr import IMAGE_KNOWLEDGE_EXTENSIONS, image_bytes_to_text, pdf_bytes_to_ocr_text

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


UPLOAD_SUBDIR = "uploads"
UPLOADED_KNOWLEDGE_EXTENSIONS = tuple(
    sorted(SUPPORTED_KNOWLEDGE_EXTENSIONS - {".url", ".urls", ".webloc"})
)


def _uploads_root(knowledge_dir: str | Path = KNOWLEDGE_DIR) -> Path:
    root = Path(knowledge_dir).expanduser().resolve()
    uploads = (root / UPLOAD_SUBDIR).resolve()
    uploads.relative_to(root)
    uploads.mkdir(parents=True, exist_ok=True)
    return uploads


def _ensure_inside(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("保存路径超出知识库目录") from exc


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._ -]+", "_", value).strip(" ._-")
    stem = re.sub(r"\s+", "-", stem)
    return stem[:80] or "knowledge"


def _host_allowed(hostname: str) -> bool:
    allowlist = APP_CONFIG["remote_knowledge_allowlist"]
    if not allowlist:
        return True
    normalized = hostname.lower()
    return any(
        normalized == host.lower() or normalized.endswith(f".{host.lower()}")
        for host in allowlist
    )


def _host_resolves_to_public_address(hostname: str) -> bool:
    infos = socket.getaddrinfo(hostname, None)
    if not infos:
        return False

    for item in infos:
        address = item[4][0]
        parsed = ip_address(address)
        if (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_multicast
            or parsed.is_reserved
            or parsed.is_unspecified
        ):
            return False
    return True


def validate_url_for_ingestion(url: str, *, check_network: bool = True) -> str:
    if not APP_CONFIG["allow_url_knowledge_ingestion"]:
        raise ValueError("当前配置禁止通过 URL 添加知识。")

    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("只支持 http 或 https URL")

    if not _host_allowed(parsed.hostname):
        raise ValueError(f"URL host 不在允许列表中：{parsed.hostname}")

    if check_network and not _host_resolves_to_public_address(parsed.hostname):
        raise ValueError(f"URL host 不是公网地址：{parsed.hostname}")

    return normalized


def parse_urls(value: str) -> list[str]:
    urls = re.findall(r"https?://[^\s,，]+", value.strip(), flags=re.IGNORECASE)
    deduped: list[str] = []
    seen = set()
    for url in urls:
        normalized = url.strip().rstrip("。；;)")
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _html_to_text(html: str) -> tuple[str, str]:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    else:
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def _pdf_bytes_to_text(data: bytes) -> str:
    if PdfReader is None:
        raise ValueError("pypdf is not installed")

    reader = PdfReader(BytesIO(data))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"[page {index}]\n{text}")

    content = "\n\n".join(pages).strip()
    if not content:
        return pdf_bytes_to_ocr_text(data)
    return content


def fetch_url_snapshot(url: str) -> dict[str, str]:
    validated_url = validate_url_for_ingestion(url)
    max_bytes = APP_CONFIG["url_knowledge_max_bytes"]
    parsed = urlparse(validated_url)
    request = Request(
        validated_url,
        headers={"User-Agent": "medagent-knowledge-ingest/1.0"},
    )

    with urlopen(request, timeout=APP_CONFIG["url_fetch_timeout"]) as response:
        content_type = response.headers.get("Content-Type", "")
        is_pdf = bool(re.search(r"pdf", content_type, re.IGNORECASE)) or parsed.path.lower().endswith(".pdf")
        is_image = bool(re.search(r"image/", content_type, re.IGNORECASE)) or Path(parsed.path).suffix.lower() in IMAGE_KNOWLEDGE_EXTENSIONS
        if content_type and not is_pdf and not is_image and not re.search(r"(text|html|json|xml)", content_type, re.IGNORECASE):
            raise ValueError(f"暂不支持该 URL 的内容类型：{content_type}")

        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f"URL 内容超过限制：{max_bytes} bytes")

        if is_pdf:
            return {
                "url": validated_url,
                "title": Path(parsed.path).stem or parsed.netloc,
                "content": _pdf_bytes_to_text(raw),
            }

        if is_image:
            return {
                "url": validated_url,
                "title": Path(parsed.path).stem or parsed.netloc,
                "content": image_bytes_to_text(raw),
            }

        charset = response.headers.get_content_charset() or "utf-8"
        decoded = raw.decode(charset, errors="ignore")

    if re.search(r"html", content_type, re.IGNORECASE) or "<html" in decoded[:500].lower():
        title, text = _html_to_text(decoded)
    else:
        title = parsed.netloc
        text = decoded.strip()

    if not text:
        raise ValueError("未能从 URL 中提取到有效文本")

    return {
        "url": validated_url,
        "title": title or parsed.netloc,
        "content": text,
    }


def sanitize_filename(filename: str) -> str:
    source_name = str(filename).replace("\\", "/").split("/")[-1]
    suffix = Path(source_name).suffix.lower()
    if suffix not in UPLOADED_KNOWLEDGE_EXTENSIONS:
        supported = ", ".join(UPLOADED_KNOWLEDGE_EXTENSIONS)
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}。支持：{supported}")

    stem = _safe_stem(Path(source_name).stem)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:8]
    return f"{timestamp}-{token}-{stem}{suffix}"


def write_uploaded_knowledge(
    filename: str,
    data: bytes,
    *,
    knowledge_dir: str | Path = KNOWLEDGE_DIR,
) -> Path:
    if not data:
        raise ValueError("文件内容为空")

    uploads = _uploads_root(knowledge_dir)
    target = (uploads / sanitize_filename(filename)).resolve()
    _ensure_inside(target, uploads)
    target.write_bytes(data)
    return target


def write_text_knowledge(
    title: str,
    content: str,
    *,
    knowledge_dir: str | Path = KNOWLEDGE_DIR,
) -> Path:
    body = content.strip()
    if not body:
        raise ValueError("文本内容为空")

    clean_title = " ".join(title.split()).strip() or "网页添加资料"
    stem = _safe_stem(clean_title)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:8]
    filename = f"{timestamp}-{token}-{stem}.md"

    uploads = _uploads_root(knowledge_dir)
    target = (uploads / filename).resolve()
    _ensure_inside(target, uploads)
    added_at = datetime.now().isoformat(timespec="seconds")
    target.write_text(
        f"# {clean_title}\n\n来源：MedAgent 网页添加\n添加时间：{added_at}\n\n{body}\n",
        encoding="utf-8",
    )
    return target


def write_url_knowledge(
    urls_text: str,
    title: str = "",
    *,
    knowledge_dir: str | Path = KNOWLEDGE_DIR,
    fetcher: Any | None = None,
) -> tuple[list[Path], list[str]]:
    urls = parse_urls(urls_text)
    if not urls:
        return [], ["没有识别到 http(s) URL"]

    uploads = _uploads_root(knowledge_dir)
    saved_paths: list[Path] = []
    errors: list[str] = []

    for url in urls:
        try:
            if fetcher is None:
                snapshot = fetch_url_snapshot(url)
            else:
                validate_url_for_ingestion(url, check_network=False)
                snapshot = fetcher(url)

            source_url = str(snapshot["url"])
            source_title = " ".join(str(snapshot.get("title", "")).split()).strip()
            content = str(snapshot["content"]).strip()
            if not content:
                raise ValueError("URL 正文为空")

            display_title = " ".join(title.split()).strip() or source_title or urlparse(source_url).netloc
            stem = _safe_stem(display_title)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            token = uuid.uuid4().hex[:8]
            target = (uploads / f"{timestamp}-{token}-{stem}.md").resolve()
            _ensure_inside(target, uploads)
            ingested_at = datetime.now().isoformat(timespec="seconds")
            target.write_text(
                "\n".join(
                    [
                        "---",
                        f"medagent_source_type: url",
                        f"medagent_source_url: {source_url}",
                        f"medagent_source_title: {source_title}",
                        f"medagent_ingested_at: {ingested_at}",
                        "---",
                        "",
                        f"# {display_title}",
                        "",
                        f"原始 URL：{source_url}",
                        f"抓取时间：{ingested_at}",
                        "",
                        content,
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            saved_paths.append(target)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return saved_paths, errors


def list_uploaded_knowledge(
    *,
    knowledge_dir: str | Path = KNOWLEDGE_DIR,
    limit: int = 6,
) -> list[dict[str, Any]]:
    uploads = _uploads_root(knowledge_dir)
    files = [path for path in uploads.iterdir() if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    items: list[dict[str, Any]] = []
    for path in files[:limit]:
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "path": str(path),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return items
