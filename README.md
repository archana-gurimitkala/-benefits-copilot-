---
title: Benefits Copilot
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: Chat with your benefits documents using AI
---

# 📋 Benefits Copilot

> Upload any employee benefits proposal PDF and ask questions in plain English — no more digging through pages to find premium costs, coverage details, or plan recommendations.

---

## The Story Behind This

My first RAG project was [StayEasy](https://github.com/archana-gurimitkala/stayeasy-rag) — a hotel booking assistant that answered questions about properties using basic RAG. It worked, but it was a demo. The pipeline was simple:

```
PDF → embed everything → vector search → answer
```

It got the job done for a demo, but I kept wondering — *would this actually hold up in a real use case?* What happens when the document has specific numbers, dollar amounts, plan names? Vector search alone struggles with exact keywords. And how do you even know if your RAG is any good?

That's what pushed me to build Benefits Copilot. Same concept, but built the right way for production.

---

## What's Different This Time

### StayEasy RAG (Basic)
```
PDF → embed → vector search → answer
```
- Single retrieval method (vector only)
- No way to measure quality
- No citation enforcement
- One shared database for all users

### Benefits Copilot (Production)
```
PDF → embed → hybrid search → reranking → answer with citations
                                        ↓
                              RAGAS quality evaluation
```
- **Hybrid retrieval** — combines BM25 keyword search + vector semantic search
- **Cross-encoder reranking** — picks the best chunks from the top results
- **Mandatory citations** — every answer includes the source file and page number
- **RAGAS evaluation** — automatically scores the system using an LLM as judge
- **Per-session isolation** — each user gets their own private document space

---

## How It Works

### 1. Hybrid Search — Two is Better Than One

Vector search is great at understanding meaning but struggles with exact terms like `$487.00` or `HDHP`. BM25 is great at exact keyword matching but misses semantic meaning.

So I combined both:

```
final_score = 0.4 × BM25_score + 0.6 × vector_score
```

Vector gets more weight because benefits questions are mostly semantic — but BM25 catches the exact numbers and plan names that vector search misses.

### 2. Cross-Encoder Reranking

After hybrid search returns the top 10 chunks, a cross-encoder model reads the question and each chunk **together** — like a human would — and picks the best 3. This is much more accurate than comparing them separately.

Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2` — lightweight, runs on CPU, specifically trained for passage relevance.

### 3. Citation Enforcement

The system prompt forces the LLM to include `[Source: filename, Page X]` in every answer. If the answer isn't in the document, it says so — it never guesses.

### 4. RAGAS Evaluation

I used RAGAS to measure quality with three metrics:

| Metric | What it measures | Score |
|--------|-----------------|-------|
| **Faithfulness** | Is the answer grounded in the chunks? (no hallucination) | **1.00** |
| **Answer Relevancy** | Does the answer actually address the question? | **0.85** |
| **Context Precision** | Were the retrieved chunks the right ones? | **1.00** |
| **Overall** | | **0.95** |

Getting here wasn't instant. Context Precision started at 0.43 — nearly half the retrieved chunks were irrelevant. Fixing it meant tuning chunk size, adjusting retrieval weights, tightening the prompt, and iterating on ground truth answers until the numbers made sense.

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF parsing | pdfplumber | Accurate text + page numbers |
| Vector DB | ChromaDB | Free, runs locally, per-session ephemeral mode |
| Embeddings | SentenceTransformer (all-MiniLM-L6-v2) | Free, runs on CPU |
| Keyword search | BM25 (rank-bm25) | Exact word + number matching |
| Reranker | CrossEncoder (ms-marco-MiniLM-L-6-v2) | Accurate relevance scoring |
| LLM | Groq + Llama 3.3 70B | Free API, fast responses — with automatic fallback to secondary key |
| Evaluation | RAGAS | Automatic quality scoring |
| UI | Chainlit | Built for LLM chat apps |
| Deployment | HF Spaces + Docker | Free hosting |

---

## What I Learned

- **Basic RAG is easy. Production RAG is about the details.** Chunk size, overlap, retrieval weights, prompt wording — each one affects quality more than you'd expect.
- **You can't improve what you don't measure.** Adding RAGAS evaluation changed everything. Without it, I would have thought the system was good when Context Precision was 0.43.
- **Hybrid search beats pure vector search** for documents with specific numbers and technical terms.
- **Reranking is worth the extra step.** The cross-encoder consistently surfaced better chunks than the hybrid scores alone.
- **Prompt engineering matters more than I expected.** A single line — "answer only what was asked, don't mix topics" — pushed Answer Relevancy from 0.76 to 0.85.

---

## Running Locally

```bash
git clone https://github.com/archana-gurimitkala/benefits-copilot
cd benefits-copilot
pip install -r requirements.txt

# Add your Groq API key (add a second key as fallback — optional but recommended)
echo "GROQ_API_KEY=your_primary_key_here" > .env
echo "GROQ_API_KEY_2=your_fallback_key_here" >> .env

# Ingest the sample document
python rag/pipeline.py ingest data/sample_benefits_proposal.pdf

# Run the app
chainlit run app.py
```

---

## Screenshots

![Welcome Screen](screenshots/Benefits%201.png)

![PDF Upload & Summary](screenshots/Benefits%202.png)

![Dental Coverage Answer](screenshots/Benefits%203.png)

![Key Recommendations](screenshots/Benefits%204.png)

---

## Live Demo

[🚀 Try it on Hugging Face Spaces](https://huggingface.co/spaces/Archanacreates/benefits-copilot)

---

Built by [Archana Gurimitkala](https://github.com/archana-gurimitkala)
