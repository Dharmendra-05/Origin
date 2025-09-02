"""
Document loader module for parsing text, markdown, and code files with precise line-level tracking.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentLine(BaseModel):
    """Represents a single line of text with index and metadata."""
    line_number: int
    content: str


class Document(BaseModel):
    """Represents a loaded document with source provenance and line mapping."""
    doc_id: str
    file_path: str
    file_name: str
    file_type: str
    total_lines: int
    content: str
    lines: List[DocumentLine]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_hash: str


class DocumentLoader:
    """Loads and parses local files into structured Document objects preserving line origins."""
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".rst"}

    @classmethod
    def load_file(cls, file_path: str) -> Document:
        """Reads a file and returns a structured Document with line-level indices."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
            
        raw_lines = raw_text.splitlines()
        doc_lines = [
            DocumentLine(line_number=idx + 1, content=line)
            for idx, line in enumerate(raw_lines)
        ]
        
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
        doc_id = f"doc_{content_hash}"
        
        return Document(
            doc_id=doc_id,
            file_path=os.path.abspath(file_path),
            file_name=file_name,
            file_type=ext if ext else "unknown",
            total_lines=len(raw_lines),
            content=raw_text,
            lines=doc_lines,
            metadata={
                "file_size_bytes": os.path.getsize(file_path),
                "extension": ext
            },
            content_hash=content_hash
        )

    @classmethod
    def load_directory(cls, dir_path: str, recursive: bool = True) -> List[Document]:
        """Scans a directory and loads all supported documents."""
        documents = []
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
            
        if recursive:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in cls.SUPPORTED_EXTENSIONS:
                        full_path = os.path.join(root, file)
                        documents.append(cls.load_file(full_path))
        else:
            for file in os.listdir(dir_path):
                full_path = os.path.join(dir_path, file)
                if os.path.isfile(full_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in cls.SUPPORTED_EXTENSIONS:
                        documents.append(cls.load_file(full_path))
                        
        return documents
