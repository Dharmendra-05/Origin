"""
Integration tests for OriginRAGPipeline module.
"""

import unittest
import os
import shutil
import tempfile
from origin_rag.pipeline import OriginRAGPipeline


class TestOriginRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.doc_file = os.path.join(self.test_dir, "system.md")
        with open(self.doc_file, "w", encoding="utf-8") as f:
            f.write("# System Specs\nOrigin-RAG uses hybrid retrieval merging vector similarity and BM25 search.")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_pipeline_end_to_end(self):
        pipeline = OriginRAGPipeline(llm_provider="mock")
        count = pipeline.ingest_directory(self.test_dir)
        self.assertGreater(count, 0)

        res = pipeline.query("What does Origin-RAG use for retrieval?")
        self.assertEqual(res.query, "What does Origin-RAG use for retrieval?")
        self.assertGreater(len(res.retrieved_chunks), 0)
        self.assertIsNotNone(res.attribution_report)


if __name__ == "__main__":
    unittest.main()
