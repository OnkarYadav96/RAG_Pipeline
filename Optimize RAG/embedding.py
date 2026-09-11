"""
embedding.py
------------
RAG pipeline step 3: convert text chunks into vector embeddings.

Uses HuggingFace sentence-transformers locally (all-MiniLM-L6-v2 by default).
Embeddings capture semantic meaning so similar questions match relevant chunks.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL

# Cache the model instance so we do not reload weights on every call
_embedding_model: Embeddings | None = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> Embeddings:
    """
    Return a singleton HuggingFaceEmbeddings instance.

    The model runs locally; no API key is required for embeddings.
    First call downloads the model weights (one-time).
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Embed a list of plain strings into vectors.

    Useful for debugging or custom pipelines outside the vector store.
    """
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(query: str, model_name: str = EMBEDDING_MODEL) -> list[float]:
    """Embed a single user question for similarity search."""
    model = get_embedding_model(model_name)
    return model.embed_query(query)
