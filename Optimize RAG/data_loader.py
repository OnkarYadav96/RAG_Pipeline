"""
data_loader.py
--------------
RAG pipeline steps 1 & 2: document loading and chunking.

- Loads PDF and plain-text files into LangChain Document objects.
- Enriches each document with metadata (source file, type, chunk index).
- Splits long documents into smaller chunks for embedding and retrieval.
"""

from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def _detect_file_type(file_path: Path) -> str:
    """Return a short file-type label used in metadata filters."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".txt", ".md"}:
        return "text"
    return "unknown"


def load_file(file_path: str | Path) -> list[Document]:
    """
    Load a single file (PDF or text) into LangChain Documents.

    Each page (PDF) or whole file (text) becomes one Document with
    metadata: source_file, file_type, loaded_at.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_type = _detect_file_type(path)
    loaded_at = datetime.now(timezone.utc).isoformat()

    if file_type == "pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()
    elif file_type == "text":
        loader = TextLoader(str(path), encoding="utf-8")
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Attach consistent metadata for filtering during similarity search
    for doc in documents:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = file_type
        doc.metadata["loaded_at"] = loaded_at
        doc.metadata["source"] = str(path)

    return documents


def load_files(file_paths: list[str | Path]) -> list[Document]:
    """Load multiple files and return a combined list of Documents."""
    all_documents: list[Document] = []
    for file_path in file_paths:
        documents = load_file(file_path)
        all_documents.extend(documents)
    return all_documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks with overlap.

    RecursiveCharacterTextSplitter tries to break on paragraphs, then
    sentences, then words — keeping semantic boundaries where possible.

    Each chunk gets chunk_index and total_chunks in metadata.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Group chunks by source file to add per-file chunk counters
    per_file_count: dict[str, int] = {}
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        per_file_count[source_file] = per_file_count.get(source_file, 0) + 1

    per_file_index: dict[str, int] = {}
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        idx = per_file_index.get(source_file, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["total_chunks"] = per_file_count[source_file]
        per_file_index[source_file] = idx + 1

    return chunks


def ingest_files(file_paths: list[str | Path]) -> list[Document]:
    """
    End-to-end data injection: load files then chunk with metadata.

    Returns chunk-level Documents ready for embedding.
    """
    documents = load_files(file_paths)
    return chunk_documents(documents)
