"""
Semantic chunking engine with overlapping character boundaries and precise line-range metadata tracking.
"""

import hashlib
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from origin_rag.document_loader import Document


class TextChunk(BaseModel):
    """Represents a discrete text chunk with line origin tracking."""
    chunk_id: str
    doc_id: str
    file_name: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    char_count: int
    word_count: int
    chunk_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextChunker:
    """Splits documents into overlapping chunks while maintaining precise line provenance."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 60):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> List[TextChunk]:
        """Chunks a Document into semantic blocks with start_line and end_line bounds."""
        chunks = []
        lines = doc.lines
        
        if not lines:
            return chunks

        current_lines = []
        current_char_count = 0
        
        for line_obj in lines:
            line_str = line_obj.content
            current_lines.append(line_obj)
            current_char_count += len(line_str) + 1  # include newline
            
            if current_char_count >= self.chunk_size:
                chunk_text = "\n".join([l.content for l in current_lines])
                start_line = current_lines[0].line_number
                end_line = current_lines[-1].line_number
                
                chunk_hash = hashlib.sha256(
                    f"{doc.doc_id}_{start_line}_{end_line}_{chunk_text}".encode("utf-8")
                ).hexdigest()[:16]
                
                chunks.append(
                    TextChunk(
                        chunk_id=f"chk_{chunk_hash}",
                        doc_id=doc.doc_id,
                        file_name=doc.file_name,
                        file_path=doc.file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=chunk_text,
                        char_count=len(chunk_text),
                        word_count=len(chunk_text.split()),
                        chunk_hash=chunk_hash,
                        metadata={
                            "file_type": doc.file_type,
                            "total_doc_lines": doc.total_lines
                        }
                    )
                )
                
                # Apply overlap by keeping last N lines that fit inside overlap boundary
                overlap_lines = []
                overlap_chars = 0
                for rev_line in reversed(current_lines):
                    if overlap_chars + len(rev_line.content) <= self.chunk_overlap:
                        overlap_lines.insert(0, rev_line)
                        overlap_chars += len(rev_line.content) + 1
                    else:
                        break
                        
                current_lines = overlap_lines
                current_char_count = overlap_chars

        # Leftover lines
        if current_lines:
            chunk_text = "\n".join([l.content for l in current_lines])
            start_line = current_lines[0].line_number
            end_line = current_lines[-1].line_number
            
            chunk_hash = hashlib.sha256(
                f"{doc.doc_id}_{start_line}_{end_line}_{chunk_text}".encode("utf-8")
            ).hexdigest()[:16]
            
            chunks.append(
                TextChunk(
                    chunk_id=f"chk_{chunk_hash}",
                    doc_id=doc.doc_id,
                    file_name=doc.file_name,
                    file_path=doc.file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_text,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    chunk_hash=chunk_hash,
                    metadata={
                        "file_type": doc.file_type,
                        "total_doc_lines": doc.total_lines
                    }
                )
            )

        return chunks
