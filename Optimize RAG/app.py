"""
app.py
------
Gradio web UI for the Optimize RAG chatbot.

Workflow:
  1. Upload PDF/TXT files -> saved to uploads/ -> chunked -> embedded -> Chroma DB
  2. Ask questions in the chat -> semantic search -> Groq LLM answer with sources

Run:
  cd "Optimize RAG"
  python app.py
"""

import shutil
import sys
from pathlib import Path

# Allow imports when launched from repo root or this folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from config import UPLOAD_DIR, VECTORSTORE_DIR
from data_loader import ingest_files
from search import format_sources_for_ui, search_and_answer
from vectorstore import clear_vectorstore, get_or_create_vectorstore, load_vectorstore

# Shared app state: active vector store after ingestion
_vectorstore = None


def _save_uploaded_files(files: list | None) -> list[Path]:
    """Persist uploaded files from Gradio to the local uploads folder."""
    if not files:
        return []

    saved_paths: list[Path] = []
    for file_obj in files:
        # Gradio may pass a path string or a tempfile-like object
        src = Path(file_obj if isinstance(file_obj, (str, Path)) else file_obj.name)
        dest = UPLOAD_DIR / src.name
        shutil.copy2(src, dest)
        saved_paths.append(dest)
    return saved_paths


def process_uploads(files) -> tuple[str, str]:
    """
    Handle file upload: ingest, chunk, embed, and store in Chroma.

    Returns status message and source summary for the UI.
    """
    global _vectorstore

    saved = _save_uploaded_files(files)
    if not saved:
        return "No files uploaded.", ""

    try:
        chunks = ingest_files(saved)
        if not chunks:
            return "Files loaded but produced no text chunks.", ""

        _vectorstore = get_or_create_vectorstore(chunks)
        file_names = ", ".join(p.name for p in saved)
        return (
            f"Indexed **{len(chunks)}** chunks from: {file_names}. "
            "You can now ask questions in the chat.",
            f"Chunks stored in `{VECTORSTORE_DIR}`",
        )
    except Exception as exc:
        return f"Error during ingestion: {exc}", ""


def reset_knowledge_base() -> str:
    """Clear vector DB and in-memory store."""
    global _vectorstore
    clear_vectorstore()
    _vectorstore = None
    return "Knowledge base cleared. Upload new documents to continue."


def chat(message: str, history: list) -> tuple[str, list]:
    """
    Chat handler: RAG search + Groq answer.

    Appends user/assistant messages to Gradio 6 chat history (messages format).
    """
    global _vectorstore

    if not message or not message.strip():
        return "", history

    # Lazy-load store if app restarted but Chroma data exists on disk
    if _vectorstore is None:
        _vectorstore = load_vectorstore()

    if _vectorstore is None:
        reply = (
            "Please upload PDF or TXT documents first using the **Upload Documents** "
            "panel, then ask your question."
        )
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return "", history

    try:
        result = search_and_answer(_vectorstore, message.strip())
        answer = result["answer"]
        sources = format_sources_for_ui(result["sources"])
        full_reply = f"{answer}\n\n---\n**Sources:**\n{sources}"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_reply},
        ]
    except Exception as exc:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error: {exc}"},
        ]

    return "", history


def build_ui() -> gr.Blocks:
    """Construct the Gradio layout: upload panel + chatbot."""
    with gr.Blocks(title="Optimize RAG Chatbot") as demo:
        gr.Markdown(
            """
            # Optimize RAG Chatbot
            Upload **PDF** or **TXT** files, then ask questions grounded in your documents.
            Uses local embeddings + **Groq** LLM (API key from `.env`).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Documents")
                file_input = gr.File(
                    label="Select files (PDF, TXT, MD)",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"],
                )
                upload_btn = gr.Button("Index Documents", variant="primary")
                clear_btn = gr.Button("Clear Knowledge Base", variant="secondary")
                upload_status = gr.Markdown("")
                store_info = gr.Markdown("")

            with gr.Column(scale=2):
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(label="RAG Chatbot", height=480)
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What is RAG? What is LangChain?",
                    lines=2,
                )
                send_btn = gr.Button("Send", variant="primary")

        # Wire events
        upload_btn.click(
            fn=process_uploads,
            inputs=[file_input],
            outputs=[upload_status, store_info],
        )
        clear_btn.click(fn=reset_knowledge_base, outputs=[upload_status])

        msg.submit(fn=chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
        send_btn.click(fn=chat, inputs=[msg, chatbot], outputs=[msg, chatbot])

    return demo


def main() -> None:
    """Launch the Gradio server."""
    global _vectorstore
    _vectorstore = load_vectorstore()
    if _vectorstore is not None:
        print(f"Loaded existing vector store from {VECTORSTORE_DIR}")

    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
