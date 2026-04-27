"""
Hybrid Retrieval = BM25 + Vector Search combined.

BM25 finds exact keywords, vector search finds semantic meaning.
Combined score = 0.4 * bm25 + 0.6 * vector
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
TOP_K = 10


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def hybrid_search(question: str, top_k: int = TOP_K, collection: chromadb.Collection = None) -> list[dict]:
    if collection is None:
        from rag.ingest import get_collection
        collection = get_collection()

    all_data = collection.get(include=["documents", "metadatas"])
    all_texts = all_data["documents"]
    all_metadatas = all_data["metadatas"]

    if not all_texts:
        return []

    # BM25 keyword search
    tokenized_corpus = [tokenize(text) for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    # Vector semantic search
    question_embedding = embedding_model.encode(question).tolist()
    vector_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=len(all_texts),
        include=["distances"],
    )

    vector_distances = np.array(vector_results["distances"][0])
    max_dist = vector_distances.max() if vector_distances.max() > 0 else 1
    vector_scores = 1 - (vector_distances / max_dist)

    result_ids = vector_results["ids"][0]
    all_ids = all_data["ids"]
    vector_score_map = {chunk_id: vector_scores[i] for i, chunk_id in enumerate(result_ids)}

    # Combine scores
    chunks = []
    for i, (text, metadata) in enumerate(zip(all_texts, all_metadatas)):
        chunk_id = all_ids[i]
        bm25_score = float(bm25_scores[i])
        v_score = float(vector_score_map.get(chunk_id, 0))
        combined_score = (BM25_WEIGHT * bm25_score) + (VECTOR_WEIGHT * v_score)

        chunks.append({
            "text": text,
            "page": metadata["page"],
            "source": metadata["source"],
            "bm25_score": round(bm25_score, 4),
            "vector_score": round(v_score, 4),
            "score": round(combined_score, 4),
            "retrieval_method": "hybrid",
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:top_k]


if __name__ == "__main__":
    question = "What is the premium increase percentage?"
    print(f"\nHybrid search for: '{question}'\n")
    results = hybrid_search(question)
    for i, r in enumerate(results, 1):
        print(f"{i}. [Page {r['page']}] Combined: {r['score']} | BM25: {r['bm25_score']} | Vector: {r['vector_score']}")
        print(f"   {r['text'][:120]}...")
        print()
