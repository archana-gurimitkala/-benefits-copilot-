"""
Day 4: RAGAS Evaluation Pipeline.

RAGAS automatically scores your RAG system using an LLM as a judge.

3 metrics we measure:
1. Faithfulness    — is the answer grounded in the retrieved chunks?
2. Answer Relevance — does the answer actually address the question?
3. Context Precision — were the retrieved chunks relevant to the question?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv

load_dotenv()

# Use Groq as the judge LLM (free)
groq_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.3-70b-versatile", temperature=0, n=1)
)

# Use local embeddings (free)
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Set LLM and embeddings on each metric
faithfulness.llm = groq_llm
answer_relevancy.llm = groq_llm
answer_relevancy.embeddings = embeddings
context_precision.llm = groq_llm

TEST_QUESTIONS = [
    "What is the total annual premium increase?",
    "What is the employee only monthly premium?",
    "What are the key recommendations in the proposal?",
    "What is the dental plan coverage?",
    "What is the vision plan benefit?",
]

# Ground truth answers for context_precision
GROUND_TRUTHS = [
    "The total annual increase is $23,400 which is +8.2%.",
    "The employee only monthly premium is $487.00 for medical.",
    "The recommendations include adding an HDHP with HSA option and reviewing dental rates.",
    "The dental plan covers preventive care at 100%, basic services at 80%, and major services at 50%.",
    "The vision plan provides an annual eye exam and $150 allowance for frames or contacts.",
]


def run_evaluation(ask_fn) -> dict:
    print("Running RAGAS evaluation on test questions...")
    print(f"Total questions: {len(TEST_QUESTIONS)}\n")

    questions = []
    answers = []
    contexts = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {question}")
        result = ask_fn(question)
        questions.append(question)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["chunks_used"]])

    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": GROUND_TRUTHS,
    })

    print("\nScoring with RAGAS (this takes a minute)...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        raise_exceptions=False,
    )

    df = results.to_pandas()
    scores = {
        "faithfulness": round(float(df["faithfulness"].mean()), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
        "context_precision": round(float(df["context_precision"].mean()), 4),
    }

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:      {scores['faithfulness']:.2f} / 1.0")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.2f} / 1.0")
    print(f"Context Precision: {scores['context_precision']:.2f} / 1.0")
    overall = sum(scores.values()) / len(scores)
    print(f"Overall Score:     {overall:.2f} / 1.0")
    print("="*50)

    scores["overall"] = round(overall, 4)
    return scores


if __name__ == "__main__":
    from rag.pipeline import ask
    run_evaluation(ask)
