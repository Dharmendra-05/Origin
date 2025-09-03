"""
Attribution Verifier module computing n-gram semantic overlap, line-level citations, and Hallucination Risk Scores.
"""

import re
from typing import List, Dict, Any, Set
from pydantic import BaseModel, Field
from origin_rag.chunker import TextChunk


class Citation(BaseModel):
    """Represents a validated citation linking answer text to document line ranges."""
    source_file: str
    start_line: int
    end_line: int
    matched_text_snippet: str
    confidence_score: float
    citation_tag: str


class AttributionReport(BaseModel):
    """Evaluation summary detailing attribution metrics and hallucination risks."""
    answer_text: str
    citations: List[Citation]
    attribution_coverage: float = Field(..., description="Percentage of answer grounded in source context (0.0 - 1.0)")
    hallucination_score: float = Field(..., description="Hallucination risk score (0.0 - 1.0)")
    is_hallucination_risk: bool
    summary_notes: str


class AttributionVerifier:
    """Verifies generated LLM responses against retrieved text chunks to ensure factual grounding."""

    def __init__(self, min_ngram: int = 2, max_ngram: int = 4, threshold: float = 0.70):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.threshold = threshold

    def _extract_ngrams(self, text: str, n: int) -> Set[str]:
        """Extracts normalized word n-grams from text."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < n:
            return set()
        return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

    def verify(self, answer: str, retrieved_chunks: List[TextChunk]) -> AttributionReport:
        """Analyzes answer text against retrieved chunks and calculates attribution metrics."""
        if not answer.strip() or not retrieved_chunks:
            return AttributionReport(
                answer_text=answer,
                citations=[],
                attribution_coverage=0.0,
                hallucination_score=1.0,
                is_hallucination_risk=True,
                summary_notes="No retrieved context available or empty answer."
            )

        context_text = " ".join([c.content for c in retrieved_chunks])
        answer_ngrams = self._extract_ngrams(answer, self.min_ngram)
        context_ngrams = self._extract_ngrams(context_text, self.min_ngram)

        if not answer_ngrams:
            coverage = 1.0
        else:
            matched_ngrams = answer_ngrams.intersection(context_ngrams)
            coverage = len(matched_ngrams) / len(answer_ngrams)

        hallucination_score = max(0.0, min(1.0, round(1.0 - coverage, 4)))
        is_risk = coverage < self.threshold

        # Extract citations per chunk
        citations = []
        for chunk in retrieved_chunks:
            chunk_ngrams = self._extract_ngrams(chunk.content, self.min_ngram)
            if not answer_ngrams:
                chunk_overlap = 0.0
            else:
                overlap = answer_ngrams.intersection(chunk_ngrams)
                chunk_overlap = len(overlap) / len(answer_ngrams)

            if chunk_overlap > 0.15:
                tag = f"[Source: {chunk.file_name}#L{chunk.start_line}-L{chunk.end_line}]"
                citations.append(
                    Citation(
                        source_file=chunk.file_name,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        matched_text_snippet=chunk.content[:120] + "...",
                        confidence_score=round(chunk_overlap, 4),
                        citation_tag=tag
                    )
                )

        notes = (
            "High verification confidence. Grounded in context."
            if not is_risk
            else f"Warning: Low attribution coverage ({round(coverage * 100, 1)}%). Potential hallucination."
        )

        return AttributionReport(
            answer_text=answer,
            citations=citations,
            attribution_coverage=round(coverage, 4),
            hallucination_score=hallucination_score,
            is_hallucination_risk=is_risk,
            summary_notes=notes
        )
