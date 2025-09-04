"""
Command-Line Interface (CLI) entry point for Origin-RAG.
"""

import sys
import argparse
from origin_rag.pipeline import OriginRAGPipeline


def main():
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        prog="origin-rag",
        description="Origin-RAG: High-Precision Source-Attribution & Line Citation Engine"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a directory of documents into vector store")
    ingest_parser.add_argument("path", help="Path to document directory")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Execute RAG query with line-level citations")
    query_parser.add_argument("prompt", help="User question or instruction prompt")
    query_parser.add_argument("--path", default="sample_data/knowledge_base", help="Knowledge base path")
    query_parser.add_argument("--verify", action="store_true", help="Print detailed attribution verification score")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    pipeline = OriginRAGPipeline(llm_provider="mock")

    if args.command == "ingest":
        print(f"[*] Ingesting documents from: {args.path}")
        count = pipeline.ingest_directory(args.path)
        print(f"[+] Ingestion complete! Indexed {count} text chunks.")

    elif args.command == "query":
        pipeline.ingest_directory(args.path)
        print(f"[*] Querying: '{args.prompt}'")
        res = pipeline.query(args.prompt)
        print("\n=== ANSWER ===")
        print(res.answer)
        
        if args.verify or True:
            report = res.attribution_report
            print("\n=== ATTRIBUTION & LINE CITATIONS ===")
            print(f"Coverage Ratio: {report.attribution_coverage * 100:.1f}%")
            print(f"Hallucination Risk Score: {report.hallucination_score:.4f}")
            print(f"Status: {report.summary_notes}")
            print("\nCitations:")
            for cite in report.citations:
                print(f" - {cite.citation_tag}")


if __name__ == "__main__":
    main()
