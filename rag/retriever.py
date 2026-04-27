"""
Vector search retriever.

Takes a question, converts it to an embedding, finds the most
similar chunks in ChromaDB.

Tomorrow we'll add BM25 here to make it hybrid retrieval.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from rag.ingest import get_collection

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
TOP_K = 10  # retrieve more than we need — reranker will cut it down later


def vector_search(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Search ChromaDB for chunks semantically similar to the question.
    Returns list of chunks with text, page, source, and similarity score.
    """
    collection = get_collection()
    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "source": results["metadatas"][0][i]["source"],
            "score": 1 - results["distances"][0][i],  # convert distance to similarity
            "retrieval_method": "vector",
        })

    return chunks


if __name__ == "__main__":
    question = "What is the deductible for Plan B?"
    results = vector_search(question)
    print(f"\nTop {len(results)} results for: '{question}'\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [Page {r['page']}] Score: {r['score']:.3f}")
        print(f"   {r['text'][:150]}...")
        print()
