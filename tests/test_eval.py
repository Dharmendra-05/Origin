"""
Unit tests for RAGEvaluator benchmark module.
"""

import unittest
import os
import shutil
import tempfile
from origin_rag.pipeline import OriginRAGPipeline
from origin_rag.eval import RAGEvaluator, EvaluationSample


class TestRAGEvaluator(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.doc_file = os.path.join(self.test_dir, "arch.md")
        with open(self.doc_file, "w", encoding="utf-8") as f:
            f.write("# Architecture\nAttribution coverage formula computes n-gram overlap ratios.")

        self.pipeline = OriginRAGPipeline(llm_provider="mock")
        self.pipeline.ingest_directory(self.test_dir)
        self.evaluator = RAGEvaluator(self.pipeline)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_evaluate_sample(self):
        sample = EvaluationSample(
            query="What is attribution coverage formula?",
            expected_doc_ids=[self.pipeline.documents[0].doc_id],
            ground_truth_answer="Computes n-gram overlap ratios."
        )

        res = self.evaluator.evaluate_sample(sample)
        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("mrr", res)
        self.assertGreater(res["mrr"], 0.0)

    def test_evaluate_dataset(self):
        samples = [
            EvaluationSample(
                query="What is attribution coverage formula?",
                expected_doc_ids=[self.pipeline.documents[0].doc_id],
                ground_truth_answer="Computes n-gram overlap ratios."
            )
        ]
        result = self.evaluator.evaluate_dataset(samples)
        self.assertEqual(result.total_queries, 1)
        self.assertGreater(result.mrr_score, 0.0)


if __name__ == "__main__":
    unittest.main()
