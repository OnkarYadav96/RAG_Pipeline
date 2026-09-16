"""
PageIndex client factory — mirrors the official GitHub quickstart.

Official README (local):
    client = PageIndexClient(
        index="gpt-5.6-luna",
        chat="gpt-5.6-sol",
    )

Official README (cloud):
    client = PageIndexClient(
        index="cloud",
        chat="gpt-5.6-sol",
    )

Official agent demo (local dedicated client):
    from pageindex import PageIndexLocalClient
    client = PageIndexLocalClient(storage_path="./.pageindex")

We default to PageIndexClient(index=, chat=) because it is the documented
entry point for both local and cloud modes.
"""

from __future__ import annotations

import os

from pageindex import PageIndexClient, PageIndexLocalClient

from vectorless_rag.config import Settings


def configure_llm_env(settings: Settings) -> None:
    """
    Set environment variables consumed by PageIndex / LiteLLM.

    Use `gemini/` prefix so LiteLLM routes to Google AI Studio, not Vertex.
    """
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    os.environ["GEMINI_API_KEY"] = settings.google_api_key
    if settings.pageindex_api_key:
        os.environ["PAGEINDEX_API_KEY"] = settings.pageindex_api_key


def create_client(settings: Settings) -> PageIndexClient:
    """
    Build PageIndexClient per official GitHub documentation.

    Local:  index=<llm model for tree build>, chat=<llm model for Q&A>
    Cloud:  index="cloud", chat=<your llm model>
    """
    configure_llm_env(settings)

    if settings.is_local:
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PAGEINDEX_STORAGE_PATH", str(settings.storage_path))
        # Local README: index=<model that builds tree>, chat=<model that searches>
        return PageIndexClient(
            index=settings.index_model,
            chat=settings.chat_model,
        )

    # Cloud README: index="cloud", chat=<your model>
    return PageIndexClient(
        index="cloud",
        chat=settings.chat_model,
    )


def create_local_demo_client(settings: Settings) -> PageIndexLocalClient:
    """
    Alternative: exact pattern from examples/agentic_vectorless_rag_demo.py.

    Use when you want the dedicated local client class from the official repo.
    """
    configure_llm_env(settings)
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    return PageIndexLocalClient(storage_path=str(settings.storage_path))
