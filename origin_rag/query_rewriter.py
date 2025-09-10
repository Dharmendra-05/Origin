"""
Query Rewriter for semantic expansion and reformulation.
Transforms raw user queries into optimized retrieval queries using
synonym expansion, intent classification, and sub-query decomposition.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class RewrittenQuery:
    """Container for a rewritten and expanded query."""
    original: str
    rewritten: str
    sub_queries: List[str] = field(default_factory=list)
    expansions: List[str] = field(default_factory=list)
    intent: str = "informational"
    confidence: float = 1.0


class QueryRewriter:
    """
    Rewrites user queries for better retrieval performance.

    Applies three transformations:
      1. Intent classification (informational / comparative / procedural)
      2. Synonym and acronym expansion
      3. Sub-query decomposition for complex questions

    Usage:
        rewriter = QueryRewriter()
        result = rewriter.rewrite("How does RAG compare to fine-tuning?")
        # result.intent == "comparative"
        # result.sub_queries == ["How does RAG work?", "How does fine-tuning work?"]
    """

    # Built-in domain synonyms for ML/AI terminology
    SYNONYM_MAP: Dict[str, List[str]] = {
        "rag": ["retrieval-augmented generation", "retrieval augmented generation"],
        "llm": ["large language model", "language model"],
        "ml": ["machine learning"],
        "ai": ["artificial intelligence"],
        "nlp": ["natural language processing"],
        "dl": ["deep learning"],
        "bert": ["bidirectional encoder representations from transformers"],
        "gpt": ["generative pre-trained transformer"],
        "fine-tuning": ["fine tuning", "finetuning", "model adaptation"],
        "embedding": ["vector representation", "dense representation"],
        "chunking": ["text segmentation", "document splitting"],
        "hallucination": ["confabulation", "factual inconsistency"],
        "grounding": ["fact anchoring", "source attribution"],
    }

    # Intent classification patterns
    INTENT_PATTERNS: Dict[str, List[str]] = {
        "comparative": [
            r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b",
            r"\bdifference\b", r"\bbetter\b", r"\bworse\b"
        ],
        "procedural": [
            r"\bhow\s+to\b", r"\bsteps?\b", r"\bguide\b",
            r"\btutorial\b", r"\bimplement\b", r"\bsetup\b"
        ],
        "definitional": [
            r"\bwhat\s+is\b", r"\bdefine\b", r"\bmeaning\b",
            r"\bexplain\b"
        ],
    }

    def __init__(self, custom_synonyms: Optional[Dict[str, List[str]]] = None):
        if custom_synonyms:
            self.SYNONYM_MAP.update(custom_synonyms)

    def rewrite(self, query: str) -> RewrittenQuery:
        """
        Applies the full rewrite pipeline to a raw query.

        Returns a RewrittenQuery with intent, expansions, and sub-queries.
        """
        intent = self._classify_intent(query)
        expansions = self._expand_synonyms(query)
        sub_queries = self._decompose(query, intent)

        # Build the rewritten query by appending expansions
        rewritten = query
        if expansions:
            expansion_text = " ".join(f"({exp})" for exp in expansions[:3])
            rewritten = f"{query} {expansion_text}"

        return RewrittenQuery(
            original=query,
            rewritten=rewritten,
            sub_queries=sub_queries,
            expansions=expansions,
            intent=intent,
            confidence=0.85 if expansions or sub_queries else 1.0
        )

    def _classify_intent(self, query: str) -> str:
        """Classifies the query intent using regex pattern matching."""
        query_lower = query.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        return "informational"

    def _expand_synonyms(self, query: str) -> List[str]:
        """Finds matching acronyms/terms and returns their expansions."""
        expansions = []
        query_lower = query.lower()
        tokens = set(re.findall(r'\b\w+(?:-\w+)*\b', query_lower))

        for term, synonyms in self.SYNONYM_MAP.items():
            if term in tokens:
                expansions.extend(synonyms)

        return expansions

    def _decompose(self, query: str, intent: str) -> List[str]:
        """
        Decomposes complex queries into sub-queries.

        Comparative queries are split into individual components.
        Other complex queries are returned as-is.
        """
        if intent != "comparative":
            return []

        # Try to split on comparison operators
        parts = re.split(
            r'\s+(?:vs\.?|versus|compared?\s+to|or)\s+',
            query,
            flags=re.IGNORECASE
        )

        if len(parts) >= 2:
            base_pattern = self._extract_question_stem(query)
            return [f"{base_pattern} {part.strip()}?" for part in parts if part.strip()]

        return []

    def _extract_question_stem(self, query: str) -> str:
        """Extracts the question stem before the comparison subjects."""
        match = re.match(
            r'((?:how|what|why|when|which)\s+\w+\s+)',
            query,
            re.IGNORECASE
        )
        return match.group(1).strip() if match else "What is"
