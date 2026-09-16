"""
Configuration for Vectorless RAG (official PageIndex SDK).

Follows the PageIndex GitHub quickstart:
  https://github.com/VectifyAI/PageIndex

Two modes (set PAGEINDEX_MODE in .env):
  - local  — index on your machine, no PageIndex API key (text PDFs)
  - cloud  — index on PageIndex Cloud (OCR, scanned PDFs, managed storage)

LLM keys:
  - Google Gemini via LiteLLM: use model names with the `gemini/` prefix
    (e.g. gemini/gemini-2.0-flash) and set GOOGLE_API_KEY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """
    Runtime settings aligned with PageIndexClient(index=..., chat=...).

    Attributes:
        mode: "local" or "cloud".
        pageindex_api_key: Required only for cloud mode.
        google_api_key: Google AI Studio key (GOOGLE_API_KEY / GEMINI_API_KEY).
        index_model: Model that builds the tree (local) or "cloud" (cloud mode).
        chat_model: Model that searches the tree and writes answers.
        storage_path: Local tree storage directory (local mode only).
        documents_dir: Default folder to discover PDFs.
    """

    mode: str
    google_api_key: str
    index_model: str
    chat_model: str
    pageindex_api_key: str = ""
    storage_path: Path = _PROJECT_ROOT / "data" / "pageindex"
    documents_dir: Path = _PROJECT_ROOT / "data" / "pdf_files"

    @property
    def is_cloud(self) -> bool:
        return self.mode.lower() == "cloud"

    @property
    def is_local(self) -> bool:
        return self.mode.lower() == "local"

    @classmethod
    def from_env(cls) -> Settings:
        """
        Load settings from environment variables.

        Required:
            GOOGLE_API_KEY — Gemini API key for index + chat (LiteLLM)

        Cloud mode (PAGEINDEX_MODE=cloud):
            PAGEINDEX_API_KEY — from https://developer.pageindex.ai/

        Optional:
            PAGEINDEX_MODE      — "local" (default) or "cloud"
            PAGEINDEX_INDEX_MODEL — local index model (default: gemini/gemini-2.0-flash)
            PAGEINDEX_CHAT_MODEL  — chat model (default: gemini/gemini-2.0-flash)
            PAGEINDEX_STORAGE_PATH — local storage dir (default: data/pageindex)
            PDF_DOCUMENTS_DIR
        """
        mode = os.getenv("PAGEINDEX_MODE", "cloud").strip().lower()
        google_key = os.getenv("GOOGLE_API_KEY", "").strip()

        if not google_key:
            raise ValueError(
                "Missing GOOGLE_API_KEY. Set it in .env (see vectorless_rag/env.example)."
            )

        pageindex_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
        if mode == "cloud" and not pageindex_key:
            raise ValueError(
                "Cloud mode requires PAGEINDEX_API_KEY. "
                "Get one at https://developer.pageindex.ai/ "
                "or set PAGEINDEX_MODE=local."
            )

        index_model = os.getenv("PAGEINDEX_INDEX_MODEL", "gemini/gemini-2.0-flash").strip()
        chat_model = os.getenv("PAGEINDEX_CHAT_MODEL", "gemini/gemini-2.0-flash").strip()

        if mode == "cloud":
            # Official cloud quickstart: index="cloud", chat=<your model>
            index_model = "cloud"

        storage = Path(
            os.getenv("PAGEINDEX_STORAGE_PATH", str(_PROJECT_ROOT / "data" / "pageindex"))
        )
        docs_dir = Path(
            os.getenv("PDF_DOCUMENTS_DIR", str(_PROJECT_ROOT / "data" / "pdf_files"))
        )

        return cls(
            mode=mode,
            google_api_key=google_key,
            pageindex_api_key=pageindex_key,
            index_model=index_model,
            chat_model=chat_model,
            storage_path=storage,
            documents_dir=docs_dir,
        )
