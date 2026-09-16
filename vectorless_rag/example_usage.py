"""
Official PageIndex quickstart — adapted for Google Gemini.

Mirrors https://github.com/VectifyAI/PageIndex README:

    from pageindex import PageIndexClient

    client = PageIndexClient(index="cloud", chat="gemini/gemini-2.0-flash")
    doc_id = client.submit_document("report.pdf", wait=True)["doc_id"]
    answer = client.chat("What are the key findings?", doc_id=doc_id)

Run:
    pip install -U pageindex litellm python-dotenv
    python vectorless_rag/example_usage.py
"""

from pathlib import Path

from vectorless_rag.config import Settings
from vectorless_rag.pipeline import VectorlessRAGPipeline


def main() -> None:
    settings = Settings.from_env()
    rag = VectorlessRAGPipeline(settings)

    pdf_dir = settings.documents_dir
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Add a PDF under {pdf_dir}")

    pdf_path = pdf_files[0]
    question = "What is RAG and why is it used?"

    # Official 3-step demo from agentic_vectorless_rag_demo.py + README
    answer = rag.run_demo(pdf_path, question)
    print("\n--- Answer ---\n")
    print(answer)


if __name__ == "__main__":
    main()
