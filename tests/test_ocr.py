import unittest
from unittest.mock import Mock

import rag.ocr as ocr_module
from config import APP_CONFIG
from rag.ocr import OcrUnavailableError, _normalize_text, get_ocr_status


class OcrTests(unittest.TestCase):
    def test_normalize_text_removes_empty_lines(self) -> None:
        self.assertEqual(_normalize_text(" A \n\n B "), "A\nB")

    def test_ocr_runtime_can_be_disabled(self) -> None:
        original_enabled = APP_CONFIG["ocr_enabled"]
        APP_CONFIG["ocr_enabled"] = False
        try:
            with self.assertRaises(OcrUnavailableError):
                ocr_module._ensure_ocr_runtime()
            self.assertFalse(get_ocr_status()["available"])
        finally:
            APP_CONFIG["ocr_enabled"] = original_enabled

    def test_get_ocr_status_reports_missing_tesseract(self) -> None:
        original_which = ocr_module.shutil.which
        original_common_paths = ocr_module._common_tesseract_paths
        original_cmd = APP_CONFIG["tesseract_cmd"]
        original_enabled = APP_CONFIG["ocr_enabled"]
        APP_CONFIG["tesseract_cmd"] = ""
        APP_CONFIG["ocr_enabled"] = True
        ocr_module.shutil.which = lambda _: None
        ocr_module._common_tesseract_paths = lambda: []
        try:
            status = get_ocr_status()
            self.assertFalse(status["available"])
            self.assertIn("Tesseract", status["reason"])
        finally:
            APP_CONFIG["tesseract_cmd"] = original_cmd
            APP_CONFIG["ocr_enabled"] = original_enabled
            ocr_module.shutil.which = original_which
            ocr_module._common_tesseract_paths = original_common_paths

    def test_image_bytes_to_text_uses_tesseract(self) -> None:
        class FakeImage:
            def __enter__(self):
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def convert(self, _: str) -> "FakeImage":
                return self

        original_image = ocr_module.Image
        original_image_ops = ocr_module.ImageOps
        original_tesseract = ocr_module.pytesseract
        original_enabled = APP_CONFIG["ocr_enabled"]

        fake_image_module = Mock()
        fake_image_module.open.return_value = FakeImage()
        fake_ops = Mock()
        fake_ops.exif_transpose.side_effect = lambda image: image
        fake_tesseract = Mock()
        fake_tesseract.image_to_string.return_value = " 二甲双胍 OCR "
        fake_tesseract.pytesseract = Mock()

        ocr_module.Image = fake_image_module
        ocr_module.ImageOps = fake_ops
        ocr_module.pytesseract = fake_tesseract
        APP_CONFIG["ocr_enabled"] = True
        try:
            self.assertEqual(ocr_module.image_bytes_to_text(b"image"), "二甲双胍 OCR")
        finally:
            ocr_module.Image = original_image
            ocr_module.ImageOps = original_image_ops
            ocr_module.pytesseract = original_tesseract
            APP_CONFIG["ocr_enabled"] = original_enabled


if __name__ == "__main__":
    unittest.main()
