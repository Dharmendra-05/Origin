"""
Origin-RAG Configuration settings and environment handlers.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Global configuration settings for Origin-RAG engine."""
    
    app_name: str = "Origin-RAG Engine"
    environment: str = Field(default_factory=lambda: os.getenv("ORIGIN_ENV", "development"))
    debug: bool = Field(default_factory=lambda: os.getenv("ORIGIN_DEBUG", "false").lower() == "true")
    
    # Vector store config
    vector_store_type: str = Field(default_factory=lambda: os.getenv("ORIGIN_VECTOR_STORE", "hybrid"))
    embedding_dim: int = Field(default=384)
    similarity_top_k: int = Field(default=4)
    score_threshold: float = Field(default=0.25)
    
    # Text Chunker config
    chunk_size: int = Field(default=300)      # Characters per chunk
    chunk_overlap: int = Field(default=60)     # Overlap characters
    preserve_line_numbers: bool = Field(default=True)
    
    # Attribution engine thresholds
    min_attribution_score: float = Field(default=0.70)
    hallucination_warning_threshold: float = Field(default=0.35)
    
    # LLM Settings
    llm_provider: str = Field(default_factory=lambda: os.getenv("ORIGIN_LLM_PROVIDER", "mock"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", None))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))


settings = Settings()
