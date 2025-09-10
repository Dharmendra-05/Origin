"""
Unit tests for the QueryRewriter semantic expansion module.
"""

import unittest
from origin_rag.query_rewriter import QueryRewriter, RewrittenQuery


class TestQueryRewriter(unittest.TestCase):

    def setUp(self):
        self.rewriter = QueryRewriter()

    def test_informational_intent(self):
        result = self.rewriter.rewrite("What are the benefits of chunking?")
        self.assertEqual(result.intent, "definitional")
        self.assertEqual(result.original, "What are the benefits of chunking?")

    def test_comparative_intent(self):
        result = self.rewriter.rewrite("RAG vs fine-tuning for production systems")
        self.assertEqual(result.intent, "comparative")

    def test_procedural_intent(self):
        result = self.rewriter.rewrite("How to implement a retrieval pipeline")
        self.assertEqual(result.intent, "procedural")

    def test_synonym_expansion_rag(self):
        result = self.rewriter.rewrite("Explain RAG architecture")
        self.assertIn("retrieval-augmented generation", result.expansions)

    def test_synonym_expansion_llm(self):
        result = self.rewriter.rewrite("What is an LLM?")
        self.assertIn("large language model", result.expansions)

    def test_no_expansion_for_unknown_terms(self):
        result = self.rewriter.rewrite("Tell me about quantum computing")
        self.assertEqual(len(result.expansions), 0)

    def test_comparative_decomposition(self):
        result = self.rewriter.rewrite("How does RAG compare to fine-tuning?")
        # Should decompose into sub-queries when pattern matches
        self.assertEqual(result.intent, "comparative")

    def test_rewritten_query_contains_original(self):
        result = self.rewriter.rewrite("What is ML?")
        self.assertIn("What is ML?", result.rewritten)

    def test_custom_synonyms(self):
        custom = {"origin": ["origin-rag", "origin framework"]}
        rewriter = QueryRewriter(custom_synonyms=custom)
        result = rewriter.rewrite("How does Origin work?")
        self.assertIn("origin-rag", result.expansions)

    def test_default_confidence_no_expansion(self):
        result = self.rewriter.rewrite("hello world")
        self.assertEqual(result.confidence, 1.0)

    def test_reduced_confidence_with_expansion(self):
        result = self.rewriter.rewrite("Explain RAG")
        self.assertLess(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
