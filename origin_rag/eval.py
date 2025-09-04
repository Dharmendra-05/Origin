"""
Evaluation & Benchmarking Engine for Origin-RAG.
Calculates Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Attribution Score.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
from origin_rag.pipeline import OriginRAGPipeline


class EvaluationSample(BaseModel):
    """Single benchmark question with ground truth answers and expected document IDs."""
    query: str
    expected_doc_ids: List[str]
    ground_truth_answer: str


class EvaluationResult(BaseModel):
    """Aggregated metrics across a benchmark evaluation dataset."""
    total_queries: int
    precision_at_k: float
    recall_at_k: float
    mrr_score: float
    avg_attribution_coverage: float
    avg_hallucination_score: float


class RAGEvaluator:
    """Evaluates RAG pipeline retrieval and attribution quality against ground truth metrics."""

    def __init__(self, pipeline: OriginRAGPipeline):
        self.pipeline = pipeline

    def evaluate_sample(self, sample: EvaluationSample, top_k: int = 4) -> Dict[str, Any]:
        """Evaluates a single benchmark sample."""
        result = self.pipeline.query(sample.query, top_k=top_k)
        retrieved_doc_ids = [sr.chunk.doc_id for sr in result.retrieved_chunks]

        # Calculate Precision & Recall @ K
        hits = [doc_id for doc_id in retrieved_doc_ids if doc_id in sample.expected_doc_ids]
        precision = len(hits) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
        recall = len(hits) / len(sample.expected_doc_ids) if sample.expected_doc_ids else 0.0

        # Calculate MRR (Mean Reciprocal Rank)
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in sample.expected_doc_ids:
                reciprocal_rank = 1.0 / rank
                break

        report = result.attribution_report

        return {
            "precision": precision,
            "recall": recall,
            "mrr": reciprocal_rank,
            "attribution_coverage": report.attribution_coverage,
            "hallucination_score": report.hallucination_score
        }

    def evaluate_dataset(self, samples: List[EvaluationSample], top_k: int = 4) -> EvaluationResult:
        """Executes benchmark evaluation across a dataset."""
        if not samples:
            return EvaluationResult(
                total_queries=0, precision_at_k=0.0, recall_at_k=0.0,
                mrr_score=0.0, avg_attribution_coverage=0.0, avg_hallucination_score=0.0
            )

        precisions, recalls, mrrs, coverages, hallucinations = [], [], [], [], []

        for sample in samples:
            res = self.evaluate_sample(sample, top_k=top_k)
            precisions.append(res["precision"])
            recalls.append(res["recall"])
            mrrs.append(res["mrr"])
            coverages.append(res["attribution_coverage"])
            hallucinations.append(res["hallucination_score"])

        n = len(samples)
        return EvaluationResult(
            total_queries=n,
            precision_at_k=round(sum(precisions) / n, 4),
            recall_at_k=round(sum(recalls) / n, 4),
            mrr_score=round(sum(mrrs) / n, 4),
            avg_attribution_coverage=round(sum(coverages) / n, 4),
            avg_hallucination_score=round(sum(hallucinations) / n, 4)
        )
