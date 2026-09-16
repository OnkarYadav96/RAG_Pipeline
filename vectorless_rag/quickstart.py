"""
Minimal copy of the official GitHub quickstart (cloud + Gemini).

This file is intentionally close to the PageIndex README so you can compare
side-by-side with: https://github.com/VectifyAI/PageIndex

Run after setting .env:
    python vectorless_rag/quickstart.py data/pdf_files/your.pdf "Your question?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexClient

# Load secrets from project root .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# --- Same pattern as official cloud quickstart ---
os.environ["PAGEINDEX_API_KEY"] = os.environ["PAGEINDEX_API_KEY"]
os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]
os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

CHAT_MODEL = os.getenv("PAGEINDEX_CHAT_MODEL", "gemini/gemini-2.0-flash")

client = PageIndexClient(
    index="cloud",       # build and store index in PageIndex Cloud
    chat=CHAT_MODEL,       # your Gemini model searches the tree + answers
)

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/pdf_files/sample.pdf"
question = sys.argv[2] if len(sys.argv) > 2 else "Summarize this document."

# Cloud indexing is async — wait=True blocks until ready (official README)
doc_id = client.submit_document(pdf_path, wait=True)["doc_id"]
print(f"doc_id: {doc_id}")

# Official API: string question in, answer string out
answer = client.chat(question, doc_id=doc_id)
print(answer)
