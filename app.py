"""
BenefitsAI — Chainlit UI
Chat with your employee benefits documents.
"""

import sys
import os
import shutil
import tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chainlit as cl
from rag.ingest import make_ephemeral_collection, build_vector_store
from rag.pipeline import ask

WELCOME = """
# 📋 BenefitsAI — Benefits Document Assistant

Upload your employee benefits proposal PDF to get started.

**What I can help with:**
- Premium costs and annual increases
- Coverage details (medical, dental, vision)
- Plan recommendations
- Employee vs employer contributions

**How to use:**
1. Click the 📎 attachment icon and upload your PDF
2. Ask any question about the document

**Example questions:**
- *"What is the total annual premium increase?"*
- *"What does the dental plan cover?"*
- *"What are the key recommendations?"*
- *"What is the employee-only monthly premium?"*
"""


@cl.on_chat_start
async def start():
    # Create a fresh in-memory ChromaDB collection for this user session
    collection = make_ephemeral_collection()
    cl.user_session.set("collection", collection)
    cl.user_session.set("pdf_loaded", False)

    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    collection = cl.user_session.get("collection")

    # Handle PDF uploads — preserve original filename for clean citations
    uploaded_pdfs = []
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path and element.name.lower().endswith(".pdf"):
                tmp_dir = tempfile.mkdtemp()
                clean_path = os.path.join(tmp_dir, element.name)
                shutil.copy2(element.path, clean_path)
                uploaded_pdfs.append(clean_path)

    if uploaded_pdfs:
        names = [os.path.basename(p) for p in uploaded_pdfs]
        await cl.Message(content=f"⏳ Ingesting **{', '.join(names)}**...").send()

        async with cl.Step(name="Processing PDF...") as step:
            build_vector_store(uploaded_pdfs, collection=collection)
            step.output = f"Ingested {len(uploaded_pdfs)} file(s)"

        cl.user_session.set("pdf_loaded", True)
        await cl.Message(content=f"✅ **{', '.join(names)}** loaded! Ask me anything about it.").send()

        if not message.content.strip():
            return

    question = message.content.strip()
    if not question:
        return

    if not cl.user_session.get("pdf_loaded"):
        await cl.Message(content="⚠️ Please upload a PDF first using the 📎 attachment icon.").send()
        return

    async with cl.Step(name="Searching document...") as step:
        result = ask(question, collection=collection)
        chunks = result["chunks_used"]
        step.output = f"Found {len(chunks)} relevant chunks"

    answer = result["answer"]

    # Deduplicated source list
    sources_seen = set()
    source_lines = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in sources_seen:
            sources_seen.add(key)
            source_lines.append(f"- **{chunk['source']}**, Page {chunk['page']}")

    sources_text = "\n".join(source_lines) if source_lines else "No sources found"

    # Chunk details with rank labels instead of raw scores
    rank_labels = ["🥇 Most relevant", "🥈 2nd", "🥉 3rd", "4th", "5th"]
    chunk_details = []
    for i, c in enumerate(chunks):
        label = rank_labels[i] if i < len(rank_labels) else f"#{i+1}"
        chunk_details.append(f"**{label}** — {c['source']}, Page {c['page']}\n> {c['text'][:250]}...")
    source_detail = "\n\n".join(chunk_details)

    await cl.Message(content=answer).send()

    elements = [
        cl.Text(
            name="Sources",
            content=f"### Retrieved Sources\n{sources_text}\n\n---\n### Chunk Details\n{source_detail}",
            display="side",
        )
    ]
    await cl.Message(content="📎 **Sources used** — click to expand", elements=elements).send()
