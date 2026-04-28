"""
Step 1: Load PDFs and store in ChromaDB with page numbers.

Supports both persistent (local dev) and ephemeral (per-session, HF Spaces) clients.
"""

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "benefits_docs"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_pages(pdf_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page": page_num,
                    "source": Path(pdf_path).name,
                })
    return pages


def chunk_page(page: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    text = page["text"]
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if end < len(text):
            last_period = chunk_text.rfind(". ")
            if last_period > chunk_size // 2:
                chunk_text = chunk_text[:last_period + 1]
                end = start + last_period + 1

        chunks.append({
            "text": chunk_text.strip(),
            "page": page["page"],
            "source": page["source"],
            "chunk_index": chunk_index,
        })
        chunk_index += 1
        start = end - overlap

    return chunks


def build_vector_store(pdf_paths: list[str], collection: chromadb.Collection = None) -> chromadb.Collection:
    """
    Embed all chunks and store in ChromaDB.
    If collection is provided (ephemeral session), use it directly.
    Otherwise use the persistent local client.
    """
    if collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(name=COLLECTION_NAME)

    all_chunks = []
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path}")
        pages = extract_pages(pdf_path)
        for page in pages:
            all_chunks.extend(chunk_page(page))

    print(f"Total chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    ids = [f"{c['source']}_p{c['page']}_c{c['chunk_index']}" for c in all_chunks]
    metadatas = [{"page": c["page"], "source": c["source"]} for c in all_chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"Stored {len(all_chunks)} chunks in ChromaDB.")
    return collection


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def make_ephemeral_collection() -> chromadb.Collection:
    """Create a fresh in-memory collection for a single user session."""
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name=COLLECTION_NAME)


if __name__ == "__main__":
    import sys
    pdfs = sys.argv[1:] if len(sys.argv) > 1 else []
    if not pdfs:
        print("Usage: python ingest.py file1.pdf file2.pdf")
    else:
        build_vector_store(pdfs)
