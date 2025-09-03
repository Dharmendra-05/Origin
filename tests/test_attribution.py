"""
Unit tests for AttributionVerifier module.
"""

import unittest
from origin_rag.chunker import TextChunk
from origin_rag.attribution import AttributionVerifier


class TestAttributionVerifier(unittest.TestCase):

    def test_attribution_verification(self):
        chunk = TextChunk(
            chunk_id="chk_1",
            doc_id="doc_1",
            file_name="spec.md",
            file_path="/path/spec.md",
            start_line=10,
            end_line=15,
            content="Attribution coverage is calculated using n-gram overlap.",
            char_count=50,
            word_count=7,
            chunk_hash="abc"
        )

        verifier = AttributionVerifier()
        answer = "Attribution coverage is calculated using n-gram overlap."
        report = verifier.verify(answer, [chunk])

        self.assertGreaterEqual(report.attribution_coverage, 0.8)
        self.assertLessEqual(report.hallucination_score, 0.2)
        self.assertFalse(report.is_hallucination_risk)
        self.assertEqual(len(report.citations), 1)
        self.assertEqual(report.citations[0].source_file, "spec.md")
        self.assertEqual(report.citations[0].start_line, 10)


if __name__ == "__main__":
    unittest.main()
