# -*- coding: utf-8 -*-
"""
Optional OCR helpers for scanned PDFs and image knowledge files.

The Python dependencies are lightweight wrappers. OCR itself requires a local
Tesseract binary when these functions are used.
"""

from __future__ import annotations

import tempfile
import shutil
import os
from io import BytesIO
from pathlib import Path
from typing import Any

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


def _common_tesseract_paths() -> list[str]:
    return [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]


def _resolve_tesseract_cmd() -> str:
    configured_cmd = APP_CONFIG["tesseract_cmd"].strip()
    if configured_cmd and Path(configured_cmd).exists():
        return configured_cmd

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return path_cmd

    for candidate in _common_tesseract_paths():
        if Path(candidate).exists():
            return candidate
    return ""


def _resolve_tessdata_prefix() -> str:
    configured_prefix = APP_CONFIG["tessdata_prefix"].strip()
    if configured_prefix and Path(configured_prefix).exists():
        return configured_prefix

    local_tessdata = Path(__file__).resolve().parents[1] / ".cache" / "tessdata"
    if local_tessdata.exists():
        return str(local_tessdata)

    return os.getenv("TESSDATA_PREFIX", "")


def _ensure_ocr_runtime() -> None:
    if not APP_CONFIG["ocr_enabled"]:
        raise OcrUnavailableError("OCR_ENABLED=false，未启用 OCR。")
    if pytesseract is None:
        raise OcrUnavailableError("缺少 pytesseract 依赖。")
    if Image is None or ImageOps is None:
        raise OcrUnavailableError("缺少 Pillow 依赖。")

    tesseract_cmd = _resolve_tesseract_cmd()
    if not tesseract_cmd:
        raise OcrUnavailableError("未找到 Tesseract 程序。")

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    tessdata_prefix = _resolve_tessdata_prefix()
    if tessdata_prefix:
        os.environ["TESSDATA_PREFIX"] = tessdata_prefix


def get_ocr_status() -> dict[str, Any]:
    if not APP_CONFIG["ocr_enabled"]:
        return {
            "available": False,
            "reason": "OCR_ENABLED=false",
            "tesseract_cmd": "",
            "lang": APP_CONFIG["ocr_lang"],
        }
    if pytesseract is None:
        return {
            "available": False,
            "reason": "缺少 pytesseract 依赖",
            "tesseract_cmd": "",
            "lang": APP_CONFIG["ocr_lang"],
        }
    if Image is None or ImageOps is None:
        return {
            "available": False,
            "reason": "缺少 Pillow 依赖",
            "tesseract_cmd": "",
            "lang": APP_CONFIG["ocr_lang"],
        }

    resolved_cmd = _resolve_tesseract_cmd()
    if not resolved_cmd:
        return {
            "available": False,
            "reason": "未找到 Tesseract 程序",
            "tesseract_cmd": "",
            "tessdata_prefix": _resolve_tessdata_prefix(),
            "lang": APP_CONFIG["ocr_lang"],
        }

    return {
        "available": True,
        "reason": "",
        "tesseract_cmd": resolved_cmd,
        "tessdata_prefix": _resolve_tessdata_prefix(),
        "lang": APP_CONFIG["ocr_lang"],
    }


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
