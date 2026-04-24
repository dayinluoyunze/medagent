import tempfile
import unittest
from pathlib import Path

from rag.knowledge_manager import (
    UPLOADED_KNOWLEDGE_EXTENSIONS,
    list_uploaded_knowledge,
    parse_urls,
    sanitize_filename,
    write_text_knowledge,
    write_url_knowledge,
    write_uploaded_knowledge,
)


class KnowledgeManagerTests(unittest.TestCase):
    def test_pdf_upload_extension_is_supported(self) -> None:
        self.assertIn(".pdf", UPLOADED_KNOWLEDGE_EXTENSIONS)

    def test_image_upload_extension_is_supported(self) -> None:
        self.assertIn(".png", UPLOADED_KNOWLEDGE_EXTENSIONS)
        self.assertIn(".jpg", UPLOADED_KNOWLEDGE_EXTENSIONS)

    def test_sanitize_filename_drops_path_parts(self) -> None:
        filename = sanitize_filename(r"..\..\secret.md")

        self.assertTrue(filename.endswith("-secret.md"))
        self.assertNotIn("..", filename)
        self.assertNotIn("/", filename)
        self.assertNotIn("\\", filename)

    def test_write_uploaded_knowledge_stays_under_upload_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_uploaded_knowledge("../guide.md", "二甲双胍 随餐服用".encode("utf-8"), knowledge_dir=tmpdir)

            self.assertEqual(path.parent, Path(tmpdir).resolve() / "uploads")
            self.assertTrue(path.exists())
            self.assertIn("二甲双胍", path.read_text(encoding="utf-8"))

    def test_write_uploaded_knowledge_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_uploaded_knowledge("payload.exe", b"bad", knowledge_dir=tmpdir)

    def test_write_text_knowledge_creates_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_text_knowledge("高血压 用药", "避免自行停药。", knowledge_dir=tmpdir)

            self.assertEqual(path.suffix, ".md")
            self.assertIn("高血压 用药", path.read_text(encoding="utf-8"))
            self.assertIn("避免自行停药", path.read_text(encoding="utf-8"))

    def test_parse_urls_deduplicates_http_urls(self) -> None:
        urls = parse_urls("https://example.com/a\nhttps://example.com/a，https://example.com/b")

        self.assertEqual(urls, ["https://example.com/a", "https://example.com/b"])

    def test_write_url_knowledge_creates_markdown_snapshot(self) -> None:
        def fake_fetcher(url: str) -> dict[str, str]:
            return {
                "url": url,
                "title": "网页标题",
                "content": "二甲双胍随餐服用可以减少胃肠道反应。",
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths, errors = write_url_knowledge(
                "https://example.com/guide",
                "",
                knowledge_dir=tmpdir,
                fetcher=fake_fetcher,
            )

            self.assertEqual(errors, [])
            self.assertEqual(len(paths), 1)
            content = paths[0].read_text(encoding="utf-8")
            self.assertIn("medagent_source_url: https://example.com/guide", content)
            self.assertIn("二甲双胍随餐服用", content)

    def test_list_uploaded_knowledge_returns_recent_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            write_uploaded_knowledge("a.md", b"A", knowledge_dir=tmpdir)
            write_uploaded_knowledge("b.md", b"B", knowledge_dir=tmpdir)

            files = list_uploaded_knowledge(knowledge_dir=tmpdir, limit=1)

            self.assertEqual(len(files), 1)
            self.assertIn("name", files[0])
            self.assertIn("size", files[0])


if __name__ == "__main__":
    unittest.main()
