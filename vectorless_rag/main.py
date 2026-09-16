"""
CLI for Vectorless RAG — official PageIndex SDK workflow.

Reference: https://github.com/VectifyAI/PageIndex

Examples:
    python -m vectorless_rag.main demo --pdf data/pdf_files/report.pdf --question "What is RAG?"
    python -m vectorless_rag.main index --pdf data/pdf_files/report.pdf
    python -m vectorless_rag.main chat --doc-id <ID> --question "Summarize section 2"
    python -m vectorless_rag.main tree --doc-id <ID>
    python -m vectorless_rag.main list
"""

from __future__ import annotations

import argparse
import sys

from vectorless_rag.config import Settings
from vectorless_rag.pipeline import VectorlessRAGPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vectorless RAG with PageIndex (official SDK).",
        epilog="Docs: https://github.com/VectifyAI/PageIndex",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # demo — mirrors official GitHub example (index → metadata → chat)
    p_demo = sub.add_parser("demo", help="Run full 3-step demo (official example flow)")
    p_demo.add_argument("--pdf", required=True, help="Path to PDF")
    p_demo.add_argument("--question", required=True, help="Question to ask")

    p_index = sub.add_parser("index", help="Index a PDF (submit_document)")
    p_index.add_argument("--pdf", required=True)
    p_index.add_argument("--reuse", action="store_true", help="Skip if already indexed")

    p_chat = sub.add_parser("chat", help="Ask via client.chat()")
    p_chat.add_argument("--doc-id", required=True)
    p_chat.add_argument("--question", required=True)
    p_chat.add_argument("--stream", action="store_true")

    p_tree = sub.add_parser("tree", help="Print tree (pageindex.utils.print_tree)")
    p_tree.add_argument("--doc-id", required=True)
    p_tree.add_argument("--save", metavar="PATH", help="Save full tree JSON")

    p_meta = sub.add_parser("meta", help="Show document metadata")
    p_meta.add_argument("--doc-id", required=True)

    sub.add_parser("list", help="List indexed documents")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    rag = VectorlessRAGPipeline(settings)
    print(f"Mode: {settings.mode} | index={settings.index_model} | chat={settings.chat_model}\n")

    if args.command == "demo":
        answer = rag.run_demo(args.pdf, args.question)
        print("\n--- Answer ---\n")
        print(answer)
        return 0

    if args.command == "index":
        doc_id = (
            rag.index_or_load(args.pdf, wait=True)
            if args.reuse
            else rag.index_document(args.pdf, wait=True)
        )
        rag.print_tree(doc_id)
        print(f"\ndoc_id={doc_id}")
        return 0

    if args.command == "chat":
        result = rag.chat(args.question, args.doc_id, stream=args.stream)
        if args.stream:
            for chunk in result:
                print(chunk, end="", flush=True)
            print()
        else:
            print(result)
        return 0

    if args.command == "tree":
        rag.print_tree(args.doc_id)
        if args.save:
            rag.save_tree_json(args.doc_id, args.save)
        return 0

    if args.command == "meta":
        print(rag.get_document_metadata(args.doc_id))
        return 0

    if args.command == "list":
        docs = rag.list_documents()
        if not docs:
            print("No documents.")
            return 0
        for doc in docs:
            doc_id = doc.get("id") or doc.get("doc_id", "?")
            name = doc.get("name") or doc.get("doc_name", "?")
            status = doc.get("status", "")
            print(f"{doc_id} | {name} | {status}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
