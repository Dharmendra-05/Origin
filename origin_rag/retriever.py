"""
Hybrid Retriever combining BM25 and Dense search using Reciprocal Rank Fusion (RRF).
"""

from typing import List
from origin_rag.vector_store import HybridVectorStore, SearchResult


class HybridRetriever:
    """Combines multiple search result rankings via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, vector_store: HybridVectorStore, rrf_k: int = 60):
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Retrieves and merges sparse and dense rankings using Reciprocal Rank Fusion."""
        bm25_results = self.vector_store.bm25_search(query, top_k=top_k * 2)
        dense_results = self.vector_store.dummy_dense_search(query, top_k=top_k * 2)

        rrf_scores = {}
        chunk_map = {}

        # Process BM25 ranking
        for rank, (chunk, score) in enumerate(bm25_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            
            if cid not in rrf_scores:
                rrf_scores[cid] = {"sparse": score, "dense": 0.0, "rrf": 0.0}
            rrf_scores[cid]["sparse"] = score
            rrf_scores[cid]["rrf"] += rrf_score

        # Process Dense ranking
        for rank, (chunk, score) in enumerate(dense_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            
            if cid not in rrf_scores:
                rrf_scores[cid] = {"sparse": 0.0, "dense": score, "rrf": 0.0}
            rrf_scores[cid]["dense"] = score
            rrf_scores[cid]["rrf"] += rrf_score

        # Sort by final RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid]["rrf"], reverse=True)
        
        final_results = []
        for cid in sorted_cids[:top_k]:
            chunk = chunk_map[cid]
            metrics = rrf_scores[cid]
            final_results.append(
                SearchResult(
                    chunk=chunk,
                    sparse_score=round(metrics["sparse"], 4),
                    dense_score=round(metrics["dense"], 4),
                    combined_score=round(metrics["rrf"], 6)
                )
            )

        return final_results
