"""
OriginRAGPipeline coordinating ingestion, hybrid retrieval, synthesis, and verification.
"""

from typing import List, Optional
from pydantic import BaseModel
from origin_rag.document_loader import DocumentLoader, Document
from origin_rag.chunker import TextChunker, TextChunk
from origin_rag.vector_store import HybridVectorStore, SearchResult
from origin_rag.retriever import HybridRetriever
from origin_rag.attribution import AttributionVerifier, AttributionReport
from origin_rag.generator import LLMGenerator


class RAGQueryResult(BaseModel):
    """Encapsulates answer output, retrieved context, and attribution report."""
    query: str
    answer: str
    retrieved_chunks: List[SearchResult]
    attribution_report: AttributionReport


class OriginRAGPipeline:
    """Main orchestrator engine for Origin-RAG operations."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 60, llm_provider: str = "mock"):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = HybridVectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.verifier = AttributionVerifier()
        self.generator = LLMGenerator(provider=llm_provider)
        
        self.documents: List[Document] = []
        self.chunks: List[TextChunk] = []

    def ingest_directory(self, dir_path: str) -> int:
        """Ingests a folder of documents and indexes them for retrieval."""
        docs = self.loader.load_directory(dir_path)
        new_chunks = []
        for doc in docs:
            self.documents.append(doc)
            c_list = self.chunker.chunk_document(doc)
            new_chunks.extend(c_list)
            
        self.chunks.extend(new_chunks)
        self.vector_store.add_chunks(new_chunks)
        return len(new_chunks)

    def query(self, query_text: str, top_k: int = 4) -> RAGQueryResult:
        """Executes full RAG workflow with attribution verification."""
        search_results = self.retriever.retrieve(query_text, top_k=top_k)
        retrieved_chunks = [sr.chunk for sr in search_results]

        answer = self.generator.generate(query_text, retrieved_chunks)
        attribution_report = self.verifier.verify(answer, retrieved_chunks)

        return RAGQueryResult(
            query=query_text,
            answer=answer,
            retrieved_chunks=search_results,
            attribution_report=attribution_report
        )
