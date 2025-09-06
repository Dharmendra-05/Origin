"""
Unit tests for AttributionVisualizer and TelemetryLogger modules.
"""

import unittest
import os
import tempfile
from origin_rag.chunker import TextChunk
from origin_rag.attribution import AttributionVerifier
from origin_rag.visualizer import AttributionVisualizer
from origin_rag.telemetry import TelemetryLogger
from origin_rag.pipeline import RAGQueryResult


class TestVisualizerAndTelemetry(unittest.TestCase):

    def setUp(self):
        self.chunk = TextChunk(
            chunk_id="chk_1",
            doc_id="doc_1",
            file_name="system.md",
            file_path="/system.md",
            start_line=1,
            end_line=2,
            content="Line 1 header\nLine 2 content",
            char_count=25,
            word_count=4,
            chunk_hash="xyz"
        )
        self.verifier = AttributionVerifier()
        self.report = self.verifier.verify("Line 2 content", [self.chunk])

    def test_ascii_visualization(self):
        ascii_out = AttributionVisualizer.render_ascii_heatmap(self.report, self.chunk)
        self.assertIn("LINE ATTRIBUTION HEATMAP", ascii_out)
        self.assertIn("system.md", ascii_out)

    def test_html_visualization(self):
        html_out = AttributionVisualizer.render_html_heatmap(self.report, self.chunk)
        self.assertIn('<div class="heatmap-box">', html_out)

    def test_telemetry_logging(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            logger = TelemetryLogger(log_file=tmp_path)
            res = RAGQueryResult(
                query="test",
                answer="Line 2 content",
                retrieved_chunks=[],
                attribution_report=self.report
            )
            rec = logger.log_query_event(res, 12.5)
            self.assertEqual(rec.latency_ms, 12.5)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
