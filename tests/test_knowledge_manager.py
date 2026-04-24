import tempfile
import unittest
from pathlib import Path

from rag.knowledge_manager import (
    UPLOADED_KNOWLEDGE_EXTENSIONS,
    delete_uploaded_knowledge,
    list_knowledge_files,
    list_uploaded_knowledge,
    parse_urls,
    read_knowledge_preview,
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

    def test_list_knowledge_files_marks_uploaded_files_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "built-in.md").write_text("内置资料", encoding="utf-8")
            uploaded = write_text_knowledge("用户资料", "网页添加内容", knowledge_dir=tmpdir)

            files = list_knowledge_files(knowledge_dir=tmpdir)
            by_relative_path = {item["relative_path"]: item for item in files}

            self.assertFalse(by_relative_path["built-in.md"]["deletable"])
            self.assertTrue(by_relative_path[uploaded.relative_to(root).as_posix()]["deletable"])

    def test_read_preview_and_delete_uploaded_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_text_knowledge("用户资料", "可以删除的资料", knowledge_dir=tmpdir)
            relative_path = path.relative_to(Path(tmpdir)).as_posix()

            self.assertIn("可以删除", read_knowledge_preview(relative_path, knowledge_dir=tmpdir))
            deleted_path = delete_uploaded_knowledge(relative_path, knowledge_dir=tmpdir)

            self.assertEqual(deleted_path, path)
            self.assertFalse(path.exists())

    def test_delete_uploaded_knowledge_rejects_builtin_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "built-in.md"
            path.write_text("内置资料", encoding="utf-8")

            with self.assertRaises(ValueError):
                delete_uploaded_knowledge("built-in.md", knowledge_dir=tmpdir)


if __name__ == "__main__":
    unittest.main()
