"""
FastAPI Pydantic Schemas for requests, responses, citation metadata, and attribution evaluation reports.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request payload to ingest local directory or file."""
    path: str = Field(..., description="Absolute or relative file/directory path to ingest")


class IngestResponse(BaseModel):
    """Response payload for document ingestion."""
    status: str
    documents_loaded: int
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    """Request payload for RAG question answering."""
    query: str = Field(..., description="User query or instruction")
    top_k: int = Field(default=4, description="Number of context chunks to retrieve")
    llm_provider: Optional[str] = Field(default="mock", description="LLM backend: mock, openai, or ollama")


class CitationItem(BaseModel):
    """Citation metadata for an attributed source line range."""
    source_file: str
    start_line: int
    end_line: int
    matched_text_snippet: str
    confidence_score: float
    citation_tag: str


class AttributionDetails(BaseModel):
    """Attribution report metrics."""
    attribution_coverage: float
    hallucination_score: float
    is_hallucination_risk: bool
    summary_notes: str


class QueryResponse(BaseModel):
    """Response payload containing answer, citations, and attribution scores."""
    query: str
    answer: str
    citations: List[CitationItem]
    attribution: AttributionDetails
    retrieved_chunk_count: int


class VerifyRequest(BaseModel):
    """Request payload to verify external text against raw document context."""
    answer_text: str
    context_text: str


class VerifyResponse(BaseModel):
    """Response payload for standalone text verification."""
    attribution_coverage: float
    hallucination_score: float
    is_hallucination_risk: bool
    notes: str
