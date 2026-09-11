"""
vectorstore.py
--------------
RAG pipeline step 4: persist chunk embeddings in a vector database.

Uses ChromaDB with on-disk persistence so uploaded documents survive
app restarts. Supports creating, loading, and appending to the store.
"""

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config import CHROMA_COLLECTION_NAME, VECTORSTORE_DIR
from embedding import get_embedding_model


def create_vectorstore(
    chunks: list[Document],
    persist_directory: str | Path = VECTORSTORE_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> VectorStore:
    """
    Create a new Chroma vector store from document chunks.

    Embeddings are computed automatically via the embedding model.
    Data is saved under `persist_directory` for later reuse.
    """
    if not chunks:
        raise ValueError("Cannot create vector store: no document chunks provided.")

    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    return vectorstore


def load_vectorstore(
    persist_directory: str | Path = VECTORSTORE_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> VectorStore | None:
    """
    Load an existing Chroma store from disk.

    Returns None if the persistence folder does not exist yet.
    """
    path = Path(persist_directory)
    if not path.exists() or not any(path.iterdir()):
        return None

    embeddings = get_embedding_model()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(path),
    )


def add_documents_to_vectorstore(
    vectorstore: VectorStore,
    chunks: list[Document],
) -> int:
    """
    Append new chunks to an existing vector store (incremental ingestion).

    Returns the number of chunks added.
    """
    if not chunks:
        return 0
    vectorstore.add_documents(chunks)
    return len(chunks)


def get_or_create_vectorstore(chunks: list[Document] | None = None) -> VectorStore | None:
    """
    Load existing store, or create one if chunks are provided.

    If no store exists and chunks is None/empty, returns None.
    """
    store = load_vectorstore()
    if store is not None:
        if chunks:
            add_documents_to_vectorstore(store, chunks)
        return store

    if chunks:
        return create_vectorstore(chunks)
    return None


def clear_vectorstore(persist_directory: str | Path = VECTORSTORE_DIR) -> None:
    """Delete persisted vector data (reset knowledge base)."""
    import shutil

    path = Path(persist_directory)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
