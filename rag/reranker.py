"""
Day 3: Cross-Encoder Reranking.

After hybrid search gives us 10 chunks, the reranker picks the best 3.

Why cross-encoder is more accurate:
- Vector search encodes question and chunk SEPARATELY then compares
- Cross-encoder reads question + chunk TOGETHER → understands their relationship
- Much more accurate but slower — that's why we use it only on top 10, not all chunks

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained specifically for passage relevance scoring
- Lightweight — runs on CPU
- Returns a score: higher = more relevant
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import CrossEncoder

print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker ready.")

TOP_N = 3  # keep only the best 3 chunks after reranking


def rerank(question: str, chunks: list[dict], top_n: int = TOP_N) -> list[dict]:
    """
    Rerank chunks using cross-encoder.
    Takes 10 chunks from hybrid search → returns best 3.
    """
    if not chunks:
        return []

    # Build (question, chunk_text) pairs for the cross-encoder
    pairs = [(question, chunk["text"]) for chunk in chunks]

    # Score each pair — higher score = more relevant
    scores = reranker.predict(pairs)

    # Add reranker score to each chunk
    for i, chunk in enumerate(chunks):
        chunk["reranker_score"] = float(scores[i])

    # Sort by reranker score and return top_n
    reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_n]


if __name__ == "__main__":
    from rag.hybrid_retriever import hybrid_search

    question = "What is the premium increase percentage?"
    print(f"\nQuestion: {question}")

    print("\n--- BEFORE reranking (hybrid top 5) ---")
    chunks = hybrid_search(question, top_k=10)
    for i, c in enumerate(chunks[:5], 1):
        print(f"{i}. [Page {c['page']}] Hybrid score: {c['score']:.4f} | {c['text'][:80]}...")

    print("\n--- AFTER reranking (top 3) ---")
    reranked = rerank(question, chunks)
    for i, c in enumerate(reranked, 1):
        print(f"{i}. [Page {c['page']}] Reranker score: {c['reranker_score']:.4f} | {c['text'][:80]}...")
