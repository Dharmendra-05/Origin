"""
LLM Provider interface supporting OpenAI API, local Ollama endpoints, and deterministic Mock models.
"""

from typing import List
from origin_rag.config import settings
from origin_rag.chunker import TextChunk


class LLMGenerator:
    """Generates context-constrained responses using configured LLM backend."""

    def __init__(self, provider: str = None):
        self.provider = provider or settings.llm_provider

    def generate(self, query: str, context_chunks: List[TextChunk]) -> str:
        """Synthesizes an answer based strictly on retrieved context chunks."""
        context_str = "\n---\n".join(
            [f"[Source: {c.file_name}#L{c.start_line}-L{c.end_line}]\n{c.content}" for c in context_chunks]
        )
        
        if self.provider == "mock":
            return self._mock_generate(query, context_chunks)
        elif self.provider == "openai":
            return self._openai_generate(query, context_str)
        elif self.provider == "ollama":
            return self._ollama_generate(query, context_str)
        else:
            return self._mock_generate(query, context_chunks)

    def _mock_generate(self, query: str, context_chunks: List[TextChunk]) -> str:
        """Deterministic mock generator for offline execution and testing."""
        if not context_chunks:
            return "I could not find relevant information in the provided knowledge base."

        primary = context_chunks[0]
        return (
            f"Based on the system documentation, {primary.content.strip()} "
            f"[Source: {primary.file_name}#L{primary.start_line}-L{primary.end_line}]"
        )

    def _openai_generate(self, query: str, context_str: str) -> str:
        """Generates answer via OpenAI Chat Completions API."""
        try:
            import openai
            client = openai.OpenAI(api_key=settings.openai_api_key)
            prompt = (
                f"You are Origin-RAG, a precise AI assistant. Answer the user query strictly using "
                f"the provided context chunks. Include line citations where applicable.\n\n"
                f"Context:\n{context_str}\n\nQuery: {query}"
            )
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI Error: {str(e)}] Fallback: " + self._mock_generate(query, [])

    def _ollama_generate(self, query: str, context_str: str) -> str:
        """Generates answer via local Ollama API."""
        import urllib.request
        import json
        
        url = f"{settings.ollama_base_url}/api/generate"
        prompt = f"Context:\n{context_str}\n\nQuery: {query}\nAnswer:"
        payload = json.dumps({"model": settings.ollama_model, "prompt": prompt, "stream": False}).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except Exception as e:
            return f"[Ollama Error: {str(e)}]"
