"""
Run vectorless RAG from inside the vectorless_rag/ folder.

When your shell is already in vectorless_rag/, use this instead of
`python -m vectorless_rag.main` (which only works from the project root).

Examples (from vectorless_rag/):
    python run.py demo --pdf ../data/pdf_files/report.pdf --question "What is RAG?"
    python run.py index --pdf ../data/pdf_files/report.pdf
    python run.py list
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root = parent of this folder (VTRAG/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vectorless_rag.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
