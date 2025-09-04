"""
Unit tests for Origin-RAG DocumentLoader module.
"""

import unittest
import os
import tempfile
from origin_rag.document_loader import DocumentLoader


class TestDocumentLoader(unittest.TestCase):

    def test_load_sample_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write("# Title\nLine 2 content\nLine 3 content")
            tmp_path = tmp.name

        try:
            doc = DocumentLoader.load_file(tmp_path)
            self.assertEqual(doc.total_lines, 3)
            self.assertEqual(len(doc.lines), 3)
            self.assertEqual(doc.lines[0].line_number, 1)
            self.assertEqual(doc.lines[0].content, "# Title")
            self.assertEqual(doc.lines[2].line_number, 3)
            self.assertEqual(doc.lines[2].content, "Line 3 content")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            DocumentLoader.load_file("non_existent_file.md")


if __name__ == "__main__":
    unittest.main()
