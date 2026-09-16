"""
Vectorless RAG pipeline — official PageIndex SDK workflow.

Reference: https://github.com/VectifyAI/PageIndex

PageIndex replaces vector DB + chunking with:
  Step 1 — Index:  build a hierarchical tree index per document
  Step 2 — Retrieve: LLM reasons over the tree (not similarity search)
  Step 3 — Generate: answer from text in selected tree nodes

Compare with your vector RAG in notebook/pdf_loader.ipynb:
  Vector RAG:  chunk → embed → Chroma → cosine similarity → LLM
  PageIndex:   tree  → LLM tree search → fetch nodes → LLM
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pageindex.utils as pi_utils
from pageindex import PageIndexClient

from vectorless_rag.client_factory import create_client
from vectorless_rag.config import Settings


class VectorlessRAGPipeline:
    """
    Thin wrapper around the official PageIndex SDK.

    Minimal usage (same as GitHub README):
        settings = Settings.from_env()
        rag = VectorlessRAGPipeline(settings)

        doc_id = rag.index_document("report.pdf")
        answer = rag.chat("What are the key findings?", doc_id)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: PageIndexClient = create_client(settings)

    # ------------------------------------------------------------------
    # Step 1 — Index: generate tree-structure index (no embeddings)
    # ------------------------------------------------------------------

    def index_document(
        self,
        file_path: str | Path,
        *,
        wait: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Submit a PDF for PageIndex tree indexing.

        Official API: client.submit_document(path, wait=True)["doc_id"]

        Local mode:
            - Synchronous flash indexing on your machine
            - Tree stored under settings.storage_path

        Cloud mode:
            - Async upload; wait=True polls until status == "completed"
            - Tree stored on PageIndex Cloud (OCR for scanned docs)
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        result = self.client.submit_document(
            str(path),
            metadata=metadata,
            wait=wait,
        )
        doc_id = result["doc_id"]
        name = result.get("name", path.name)
        print(f"[index] doc_id={doc_id}  name={name}  mode={self.settings.mode}")
        return doc_id

    def find_cached_document(self, filename: str) -> str | None:
        """
        Reuse an already-indexed document by filename (official demo pattern).

        From agentic_vectorless_rag_demo.py:
            doc_id = next(doc["id"] for doc in client.list_documents()["documents"]
                          if doc["name"] == PDF_PATH.name)
        """
        response = self.client.list_documents(limit=100)
        documents = response.get("documents", []) if isinstance(response, dict) else response
        for doc in documents:
            if doc.get("name") == filename or doc.get("doc_name") == filename:
                return doc.get("id") or doc.get("doc_id")
        return None

    def index_or_load(self, file_path: str | Path, *, wait: bool = True) -> str:
        """Index PDF if missing, otherwise return existing doc_id (demo caching)."""
        path = Path(file_path)
        cached = self.find_cached_document(path.name)
        if cached:
            print(f"[index] Using cached doc_id={cached} for {path.name}")
            return cached
        return self.index_document(path, wait=wait)

    # ------------------------------------------------------------------
    # Inspect tree — the “smart table of contents”
    # ------------------------------------------------------------------

    def get_tree(
        self,
        doc_id: str,
        *,
        node_summary: bool = True,
        include_text: bool = False,
    ) -> dict[str, Any]:
        """
        Fetch tree index from PageIndex.

        Official API: client.get_tree(doc_id, node_summary=True)

        For retrieval, summaries without full text save tokens — the chat
        agent navigates structure first, then pulls text from chosen nodes.
        """
        return self.client.get_tree(
            doc_id,
            node_summary=node_summary,
            include_text=include_text,
        )

    def print_tree(self, doc_id: str) -> None:
        """
        Print tree structure using official pageindex.utils.print_tree().

        Same as agentic_vectorless_rag_demo.py Step 1.
        """
        tree_response = self.get_tree(doc_id, node_summary=True, include_text=False)
        structure = tree_response.get("result")
        if not structure:
            print("[tree] No structure yet — document may still be processing.")
            return
        print("[tree] Document structure:")
        pi_utils.print_tree(structure)

    def get_document_metadata(self, doc_id: str) -> dict[str, Any]:
        """Official API: client.get_document(doc_id) — status, page count, etc."""
        return self.client.get_document(doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> list[dict[str, Any]]:
        """
        Read specific pages after tree search identifies them.

        Official API: client.get_page_content(doc_id, "5-7")
        pages format: "5-7", "3,8", or "12"
        """
        return self.client.get_page_content(doc_id, pages)

    def save_tree_json(self, doc_id: str, output_path: str | Path) -> Path:
        """Export full tree (with summaries + text) to JSON for inspection."""
        tree_response = self.get_tree(doc_id, node_summary=True, include_text=True)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(tree_response, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[tree] Saved to {out}")
        return out

    # ------------------------------------------------------------------
    # Step 2 + 3 — chat(): reasoning retrieval + answer generation
    # ------------------------------------------------------------------

    def chat(
        self,
        question: str,
        doc_id: str,
        *,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Ask a question — official PageIndex document-QA agent.

        Official API (README quickstart):
            answer = client.chat("What was the margin?", doc_id=doc_id)

        Internally the chat model:
            1. Reads the tree index
            2. Reasons which sections are relevant (not vector similarity)
            3. Pulls page/node text and composes the answer
        """
        return self.client.chat(
            messages=question,
            doc_id=doc_id,
            stream=stream,
        )

    def chat_with_citations(self, question: str, doc_id: str) -> dict[str, Any]:
        """
        Full chat_completions envelope — includes citations when enabled.

        Official API: client.chat_completions(..., enable_citations=True)
        """
        return self.client.chat_completions(
            messages=question,
            doc_id=doc_id,
            enable_citations=True,
        )

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        """List indexed documents (local storage or cloud library)."""
        response = self.client.list_documents(limit=limit)
        if isinstance(response, dict):
            return response.get("documents", [])
        return list(response)

    # ------------------------------------------------------------------
    # End-to-end demo (README + official example combined)
    # ------------------------------------------------------------------

    def run_demo(self, pdf_path: str | Path, question: str) -> str:
        """
        Full vectorless RAG demo matching the official GitHub example flow:

        Step 1 — Index PDF and view tree structure
        Step 2 — View document metadata
        Step 3 — Ask question via client.chat()
        """
        path = Path(pdf_path)

        print("=" * 60)
        print("Step 1: Index PDF and view tree structure")
        print("=" * 60)
        doc_id = self.index_or_load(path, wait=True)
        self.print_tree(doc_id)

        print("\n" + "=" * 60)
        print("Step 2: View document metadata")
        print("=" * 60)
        print(self.get_document_metadata(doc_id))

        print("\n" + "=" * 60)
        print("Step 3: Ask question (vectorless RAG via client.chat)")
        print("=" * 60)
        print(f"Question: {question}")
        answer = self.chat(question, doc_id)
        if not isinstance(answer, str):
            answer = "".join(answer)
        return answer
