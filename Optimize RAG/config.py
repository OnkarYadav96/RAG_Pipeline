"""
config.py
---------
Central configuration for the Optimize RAG pipeline.

Loads API keys from the project-root `.env` file and defines paths,
chunking defaults, embedding model name, and LLM settings.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

# Project root is one level above this folder (VTRAG/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the repo `.env` (GROQ_API_KEY, etc.)
load_dotenv(PROJECT_ROOT / ".env")

# --- API keys (read from .env) ---
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
VECTORSTORE_DIR = Path(__file__).resolve().parent / "chroma_db"

# Ensure runtime folders exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# --- Chunking (RAG step 2: split documents) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Embeddings (RAG step 3: text -> vectors) ---
# Local HuggingFace model via sentence-transformers (no extra API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector store (RAG step 4) ---
CHROMA_COLLECTION_NAME = "optimize_rag_docs"

# --- Retrieval (RAG step 5) ---
TOP_K = 4

# --- LLM for answer generation (uses GROQ_API_KEY from .env) ---
GROQ_MODEL = "qwen/qwen3.6-27b"
