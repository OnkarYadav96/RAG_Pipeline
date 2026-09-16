"""
Vectorless RAG — PageIndex official SDK wrapper.

GitHub: https://github.com/VectifyAI/PageIndex
Docs:   https://docs.pageindex.ai/getting-started
"""

from vectorless_rag.client_factory import create_client
from vectorless_rag.config import Settings
from vectorless_rag.pipeline import VectorlessRAGPipeline

__all__ = ["Settings", "VectorlessRAGPipeline", "create_client"]
