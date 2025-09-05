"""
Telemetry and Audit Logging module for Origin-RAG pipeline monitoring.
"""

import json
import time
from typing import Dict, Any
from pydantic import BaseModel
from origin_rag.pipeline import RAGQueryResult


class TelemetryRecord(BaseModel):
    """Telemetry log record for a RAG execution event."""
    timestamp: float
    query: str
    latency_ms: float
    retrieved_chunk_count: int
    attribution_coverage: float
    hallucination_score: float
    is_risk: bool


class TelemetryLogger:
    """Logs query telemetry and quality metrics for MLOps audit trails."""

    def __init__(self, log_file: str = "telemetry.jsonl"):
        self.log_file = log_file

    def log_query_event(self, result: RAGQueryResult, latency_ms: float) -> TelemetryRecord:
        """Logs a query execution record to a JSONL audit file."""
        report = result.attribution_report
        record = TelemetryRecord(
            timestamp=time.time(),
            query=result.query,
            latency_ms=round(latency_ms, 2),
            retrieved_chunk_count=len(result.retrieved_chunks),
            attribution_coverage=report.attribution_coverage,
            hallucination_score=report.hallucination_score,
            is_risk=report.is_hallucination_risk
        )

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.model_dump()) + "\n")
        except Exception:
            pass  # Silent fallback for restricted environments

        return record
