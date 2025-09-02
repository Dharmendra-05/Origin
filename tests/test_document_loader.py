"""
Unit tests for Origin-RAG DocumentLoader module.
"""

import os
import pytest
from origin_rag.document_loader import DocumentLoader


def test_load_sample_file(tmp_path):
    sample_file = tmp_path / "sample.md"
    sample_file.write_text("# Title\nLine 2 content\nLine 3 content", encoding="utf-8")

    doc = DocumentLoader.load_file(str(sample_file))

    assert doc.file_name == "sample.md"
    assert doc.total_lines == 3
    assert len(doc.lines) == 3
    assert doc.lines[0].line_number == 1
    assert doc.lines[0].content == "# Title"
    assert doc.lines[2].line_number == 3
    assert doc.lines[2].content == "Line 3 content"


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load_file("non_existent_file.md")
