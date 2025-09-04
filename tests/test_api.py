"""
Unit tests for FastAPI REST Endpoints in Origin-RAG.
"""

import unittest
from fastapi.testclient import TestClient
from origin_rag.api.main import app


class TestFastAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("indexed_documents", data)

    def test_verify_endpoint(self):
        payload = {
            "answer_text": "Origin-RAG computes attribution metrics.",
            "context_text": "Origin-RAG computes attribution metrics using n-gram overlap."
        }
        response = self.client.post("/verify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["attribution_coverage"], 0.8)
        self.assertFalse(data["is_hallucination_risk"])


if __name__ == "__main__":
    unittest.main()
