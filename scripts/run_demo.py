"""
End-to-end demonstration script for Origin-RAG.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from origin_rag.pipeline import OriginRAGPipeline


def run_demo():
    print("=" * 60)
    print("📍 ORIGIN-RAG: SOURCE ATTRIBUTION & LINE CITATION DEMO")
    print("=" * 60)

    kb_path = "sample_data/knowledge_base"
    print(f"\n[1] Ingesting knowledge base from '{kb_path}'...")
    
    pipeline = OriginRAGPipeline(llm_provider="mock")
    chunk_count = pipeline.ingest_directory(kb_path)
    print(f"[+] Successfully indexed {len(pipeline.documents)} documents into {chunk_count} chunks.")

    query = "What is the attribution formula used in Origin-RAG?"
    print(f"\n[2] Executing query: '{query}'...")
    result = pipeline.query(query)

    print("\n[3] Generated Response:")
    print("-" * 40)
    print(result.answer)
    print("-" * 40)

    report = result.attribution_report
    print("\n[4] Attribution Telemetry:")
    print(f" - Attribution Coverage Ratio : {report.attribution_coverage * 100:.1f}%")
    print(f" - Hallucination Risk Score   : {report.hallucination_score:.4f}")
    print(f" - System Notes               : {report.summary_notes}")

    print("\n[5] Extracted Line-Level Citations:")
    for idx, cite in enumerate(report.citations, 1):
        print(f"  {idx}. {cite.citation_tag} (Confidence: {cite.confidence_score * 100:.1f}%)")

    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
