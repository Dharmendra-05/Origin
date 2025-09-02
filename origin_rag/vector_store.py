"""
In-memory hybrid vector store supporting sparse BM25 term weighting and dense vector similarity.
"""

import math
import re
from typing import List, Tuple, Dict
from pydantic import BaseModel
from origin_rag.chunker import TextChunk


class SearchResult(BaseModel):
    """Container for chunk retrieval results with similarity metrics."""
    chunk: TextChunk
    sparse_score: float = 0.0
    dense_score: float = 0.0
    combined_score: float = 0.0


class HybridVectorStore:
    """Stores text chunks and indexes them for sparse (BM25) and dense similarity queries."""

    def __init__(self):
        self.chunks: List[TextChunk] = []
        self.doc_freqs: Dict[str, int] = {}
        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Adds text chunks to the store and updates sparse term frequencies."""
        for chunk in chunks:
            self.chunks.append(chunk)
            words = set(self._tokenize(chunk.content))
            for w in words:
                self.doc_freqs[w] = self.doc_freqs.get(w, 0) + 1

        self.num_docs = len(self.chunks)
        if self.num_docs > 0:
            total_len = sum(len(self._tokenize(c.content)) for c in self.chunks)
            self.avg_doc_len = total_len / self.num_docs

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase tokenizer for BM25 matching."""
        return re.findall(r"\w+", text.lower())

    def bm25_search(self, query: str, top_k: int = 4, k1: float = 1.5, b: float = 0.75) -> List[Tuple[TextChunk, float]]:
        """Executes BM25 sparse search over stored text chunks."""
        if not self.chunks:
            return []

        query_terms = self._tokenize(query)
        scores = []

        for chunk in self.chunks:
            doc_terms = self._tokenize(chunk.content)
            doc_len = len(doc_terms)
            score = 0.0
            
            term_counts = {}
            for t in doc_terms:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term in query_terms:
                if term in term_counts:
                    freq = term_counts[term]
                    df = self.doc_freqs.get(term, 0)
                    idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                    
                    numerator = freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * (doc_len / (self.avg_doc_len + 1e-5)))
                    score += idf * (numerator / denominator)

            scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def dummy_dense_search(self, query: str, top_k: int = 4) -> List[Tuple[TextChunk, float]]:
        """Simulates dense embedding vector similarity using token overlap Jaccard score."""
        if not self.chunks:
            return []
            
        q_terms = set(self._tokenize(query))
        scores = []
        
        for chunk in self.chunks:
            c_terms = set(self._tokenize(chunk.content))
            intersection = q_terms.intersection(c_terms)
            union = q_terms.union(c_terms)
            sim = len(intersection) / len(union) if union else 0.0
            scores.append((chunk, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
