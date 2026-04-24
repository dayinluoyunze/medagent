# -*- coding: utf-8 -*-
"""
Optional OCR helpers for scanned PDFs and image knowledge files.

The Python dependencies are lightweight wrappers. OCR itself requires a local
Tesseract binary when these functions are used.
"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

from config import APP_CONFIG

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover
    pdfium = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None


IMAGE_KNOWLEDGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class OcrUnavailableError(RuntimeError):
    """Raised when OCR is requested but the local runtime is unavailable."""


def _ensure_ocr_runtime() -> None:
    if not APP_CONFIG["ocr_enabled"]:
        raise OcrUnavailableError("OCR_ENABLED=false，未启用 OCR。")
    if pytesseract is None:
        raise OcrUnavailableError("缺少 pytesseract 依赖。")
    if Image is None or ImageOps is None:
        raise OcrUnavailableError("缺少 Pillow 依赖。")

    tesseract_cmd = APP_CONFIG["tesseract_cmd"].strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _image_to_text(image: "Image.Image") -> str:
    _ensure_ocr_runtime()
    normalized_image = ImageOps.exif_transpose(image).convert("RGB")
    try:
        text = pytesseract.image_to_string(normalized_image, lang=APP_CONFIG["ocr_lang"])
    except Exception as exc:
        raise OcrUnavailableError(f"OCR 执行失败：{exc}") from exc

    normalized = _normalize_text(text)
    if not normalized:
        raise OcrUnavailableError("OCR 未识别到有效文本。")
    return normalized


def image_bytes_to_text(data: bytes) -> str:
    _ensure_ocr_runtime()
    try:
        with Image.open(BytesIO(data)) as image:
            return _image_to_text(image)
    except OcrUnavailableError:
        raise
    except Exception as exc:
        raise OcrUnavailableError(f"图片 OCR 失败：{exc}") from exc


def image_file_to_text(path: str | Path) -> str:
    _ensure_ocr_runtime()
    try:
        with Image.open(path) as image:
            return _image_to_text(image)
    except OcrUnavailableError:
        raise
    except Exception as exc:
        raise OcrUnavailableError(f"图片 OCR 失败：{exc}") from exc


def pdf_file_to_ocr_text(path: str | Path) -> str:
    _ensure_ocr_runtime()
    if pdfium is None:
        raise OcrUnavailableError("缺少 pypdfium2 依赖，无法渲染 PDF 页面做 OCR。")

    try:
        document = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise OcrUnavailableError(f"PDF 渲染初始化失败：{exc}") from exc

    page_count = min(len(document), APP_CONFIG["ocr_max_pages"])
    if page_count <= 0:
        raise OcrUnavailableError("PDF 没有可渲染页面。")

    scale = APP_CONFIG["ocr_dpi"] / 72
    pages = []
    try:
        for index in range(page_count):
            page = document[index]
            try:
                image = page.render(scale=scale).to_pil()
                text = _image_to_text(image)
                if text:
                    pages.append(f"[ocr page {index + 1}]\n{text}")
            except OcrUnavailableError:
                raise
            except Exception as exc:
                raise OcrUnavailableError(f"PDF 第 {index + 1} 页 OCR 失败：{exc}") from exc
            finally:
                page.close()
    finally:
        document.close()

    content = "\n\n".join(pages).strip()
    if not content:
        raise OcrUnavailableError("PDF OCR 未识别到有效文本。")
    return content


def pdf_bytes_to_ocr_text(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(data)
        temp_path = Path(handle.name)

    try:
        return pdf_file_to_ocr_text(temp_path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
