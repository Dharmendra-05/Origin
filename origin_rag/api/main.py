"""
FastAPI Microservice Application for Origin-RAG.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from origin_rag.config import settings
from origin_rag.pipeline import OriginRAGPipeline
from origin_rag.attribution import AttributionVerifier
from origin_rag.chunker import TextChunk
from origin_rag.api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, CitationItem, AttributionDetails,
    VerifyRequest, VerifyResponse
)

app = FastAPI(
    title="Origin-RAG API Engine",
    description="High-Precision Source-Attribution & Line-Level Citation RAG Microservice",
    version="0.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = OriginRAGPipeline()
verifier = AttributionVerifier()


@app.get("/health", tags=["System"])
def health_check():
    """Returns system status and index telemetry."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "indexed_documents": len(pipeline.documents),
        "indexed_chunks": len(pipeline.chunks)
    }


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
def ingest_documents(req: IngestRequest):
    """Ingests local directory or file into vector store."""
    try:
        count = pipeline.ingest_directory(req.path)
        return IngestResponse(
            status="success",
            documents_loaded=len(pipeline.documents),
            chunks_created=count,
            message=f"Successfully indexed {count} chunks from {req.path}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(req: QueryRequest):
    """Executes RAG query and returns answer with line citations and hallucination score."""
    if not pipeline.chunks:
        raise HTTPException(status_code=400, detail="Vector store is empty. Call /ingest first.")

    res = pipeline.query(req.query, top_k=req.top_k)
    report = res.attribution_report

    citation_items = [
        CitationItem(
            source_file=c.source_file,
            start_line=c.start_line,
            end_line=c.end_line,
            matched_text_snippet=c.matched_text_snippet,
            confidence_score=c.confidence_score,
            citation_tag=c.citation_tag
        )
        for c in report.citations
    ]

    return QueryResponse(
        query=res.query,
        answer=res.answer,
        citations=citation_items,
        attribution=AttributionDetails(
            attribution_coverage=report.attribution_coverage,
            hallucination_score=report.hallucination_score,
            is_hallucination_risk=report.is_hallucination_risk,
            summary_notes=report.summary_notes
        ),
        retrieved_chunk_count=len(res.retrieved_chunks)
    )


@app.post("/verify", response_model=VerifyResponse, tags=["Verification"])
def verify_attribution(req: VerifyRequest):
    """Verifies arbitrary answer text against a context string."""
    dummy_chunk = TextChunk(
        chunk_id="chk_ext",
        doc_id="doc_ext",
        file_name="external_context.txt",
        file_path="/external_context.txt",
        start_line=1,
        end_line=max(1, len(req.context_text.splitlines())),
        content=req.context_text,
        char_count=len(req.context_text),
        word_count=len(req.context_text.split()),
        chunk_hash="ext"
    )

    report = verifier.verify(req.answer_text, [dummy_chunk])
    return VerifyResponse(
        attribution_coverage=report.attribution_coverage,
        hallucination_score=report.hallucination_score,
        is_hallucination_risk=report.is_hallucination_risk,
        notes=report.summary_notes
    )
