"""
The main RAG pipeline — connects all pieces together.
ingest → hybrid search → reranking → Groq answer with citations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from rag.ingest import build_vector_store
from rag.hybrid_retriever import hybrid_search
from rag.reranker import rerank
from rag.generator import generate_answer


def ingest_documents(pdf_paths: list[str], collection: chromadb.Collection = None):
    """Load PDFs into vector store. Uses provided collection if given (per-session)."""
    return build_vector_store(pdf_paths, collection=collection)


def ask(question: str, top_k: int = 5, collection: chromadb.Collection = None) -> dict:
    """
    Full pipeline: question → retrieve → rerank → generate answer with citation.

    Returns:
        {
            "answer": "...[Source: file.pdf, Page X]",
            "sources": [{"source": "file.pdf", "page": X}],
            "chunks_used": [...]
        }
    """
    chunks = hybrid_search(question, top_k=10, collection=collection)
    chunks = rerank(question, chunks, top_n=top_k)
    result = generate_answer(question, chunks)
    result["chunks_used"] = chunks
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "ingest":
            ingest_documents(sys.argv[2:])
        else:
            question = sys.argv[1]
            result = ask(question)
            print("\n" + "="*50)
            print(f"Q: {question}")
            print("="*50)
            print(f"A: {result['answer']}")
            print(f"\nSources: {result['sources']}")
