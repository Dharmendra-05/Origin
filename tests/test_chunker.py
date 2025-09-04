"""
Unit tests for Origin-RAG TextChunker module.
"""

import unittest
from origin_rag.document_loader import Document, DocumentLine
from origin_rag.chunker import TextChunker


class TestTextChunker(unittest.TestCase):

    def test_chunker_line_bounds(self):
        lines = [
            DocumentLine(line_number=1, content="Header Line 1"),
            DocumentLine(line_number=2, content="Detailed explanation text line 2."),
            DocumentLine(line_number=3, content="Detailed explanation text line 3."),
            DocumentLine(line_number=4, content="Conclusion line 4.")
        ]
        
        doc = Document(
            doc_id="test_doc",
            file_path="/path/test.md",
            file_name="test.md",
            file_type=".md",
            total_lines=4,
            content="\n".join([l.content for l in lines]),
            lines=lines,
            content_hash="123456"
        )

        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)

        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]
        self.assertEqual(first_chunk.start_line, 1)
        self.assertGreaterEqual(first_chunk.end_line, 1)
        self.assertEqual(first_chunk.file_name, "test.md")


if __name__ == "__main__":
    unittest.main()
