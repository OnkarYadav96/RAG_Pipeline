"""
search.py
---------
RAG pipeline step 5: retrieve relevant chunks and generate answers.

- Similarity search over the vector store (semantic retrieval).
- Context assembly from top-K chunks.
- Answer generation with Groq LLM using retrieved context (grounded RAG).
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K

# System prompt: instruct the LLM to answer only from provided context
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using ONLY the "
            "context below. If the answer is not in the context, say you do not "
            "have enough information in the uploaded documents. Be concise and cite "
            "the source file name when possible.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def get_llm() -> ChatGroq:
    """
    Initialize the Groq chat model using GROQ_API_KEY from `.env`.

    Raises a clear error if the API key is missing.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to the project `.env` file at the repo root."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )


def retrieve_documents(
    vectorstore: VectorStore,
    query: str,
    top_k: int = TOP_K,
) -> list[Document]:
    """
    Run semantic similarity search and return the top-K matching chunks.

    `similarity_search` compares the query embedding against stored chunk
    embeddings and returns the closest matches.
    """
    return vectorstore.similarity_search(query, k=top_k)


def format_context(documents: list[Document]) -> str:
    """Build a single context string from retrieved chunks for the LLM prompt."""
    if not documents:
        return "No relevant documents found."

    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def search_and_answer(
    vectorstore: VectorStore,
    question: str,
    top_k: int = TOP_K,
) -> dict[str, str | list[Document]]:
    """
    Full RAG query: retrieve context then generate an answer.

    Returns:
        - answer: LLM response string
        - sources: list of retrieved Document chunks
        - context: formatted context passed to the LLM
    """
    retrieved = retrieve_documents(vectorstore, question, top_k=top_k)
    context = format_context(retrieved)

    llm = get_llm()
    messages = RAG_PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "sources": retrieved,
        "context": context,
    }


def format_sources_for_ui(documents: list[Document]) -> str:
    """Format source snippets for display in the chat UI."""
    if not documents:
        return "_No sources retrieved._"

    lines: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_file", "unknown")
        preview = doc.page_content[:200].replace("\n", " ")
        if len(doc.page_content) > 200:
            preview += "..."
        lines.append(f"**{i}. {source}** — {preview}")
    return "\n\n".join(lines)
