"""
Answer generator using Groq API (llama-3.3-70b-versatile).

Key features:
- Every answer MUST cite the source page
- Returns "I don't know" if answer not in chunks (prevents hallucination)
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_context(chunks: list[dict]) -> str:
    """Format chunks with page numbers for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Chunk {i} — Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Generate an answer with a mandatory citation.

    Returns:
        {
            "answer": "The deductible is $1,500. [Source: plan_2026.pdf, Page 4]",
            "sources": [{"source": "plan_2026.pdf", "page": 4}]
        }
    """
    context = build_context(chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a benefits document assistant. Answer ONLY the specific question asked using ONLY the document chunks provided.

RULES:
1. Read the question carefully. Answer ONLY what was asked — nothing more.
2. If the question is about dental, answer only dental. If about vision, answer only vision. Do NOT mix topics.
3. Use ONLY information from the provided chunks. No outside knowledge.
4. Include all relevant numbers, percentages, and specifics that directly answer the question.
5. Every answer MUST end with citations: [Source: filename, Page X] for every chunk you used.
6. If the answer is not in the chunks, say: "I could not find this information in the uploaded documents."
7. Do NOT guess or make up information.""",
            },
            {
                "role": "user",
                "content": f"Document chunks:\n{context}\n\nQuestion: {question}\n\nAnswer (include page citation):",
            },
        ],
        temperature=0.1,
        max_tokens=800,
    )

    answer = response.choices[0].message.content.strip()

    # Extract unique sources cited
    sources = []
    seen = set()
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            sources.append({"source": chunk["source"], "page": chunk["page"]})
            seen.add(key)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    test_chunks = [
        {"text": "Plan B has an individual deductible of $1,500 per year.", "page": 4, "source": "test.pdf"},
        {"text": "Family deductible for Plan B is $3,000 per year.", "page": 4, "source": "test.pdf"},
    ]
    result = generate_answer("What is the deductible for Plan B?", test_chunks)
    print(result["answer"])
    print("\nSources:", result["sources"])
