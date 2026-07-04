"""
rag_project/evaluation/ragas_eval.py

RAG evaluation using RAGAS framework.

Metrics:
  - Faithfulness:       Does the answer follow from retrieved context?
  - Answer Relevancy:  Is the answer relevant to the question?
  - Context Recall:    Were the relevant documents retrieved?
  - Context Precision: How much of retrieved context is actually relevant?

Research basis:
  Es et al. (2023) "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  arXiv:2309.15217

Usage:
    python rag_project/evaluation/ragas_eval.py
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


# ── Test question set ─────────────────────────────────────────
TEST_SAMPLES = [
    EvalSample(
        question="What is Retrieval-Augmented Generation?",
        answer="",   # filled by the RAG pipeline at eval time
        contexts=[],
        ground_truth="RAG combines a retrieval system with a language model to generate answers grounded in retrieved documents.",
    ),
    EvalSample(
        question="How does BM25 differ from dense retrieval?",
        answer="",
        contexts=[],
        ground_truth="BM25 is a sparse lexical method using TF-IDF-like term matching, while dense retrieval uses neural embeddings to find semantically similar documents.",
    ),
    EvalSample(
        question="What is Reciprocal Rank Fusion?",
        answer="",
        contexts=[],
        ground_truth="RRF combines rankings from multiple retrieval systems by summing reciprocal ranks, giving a fused ranking that outperforms individual methods.",
    ),
]


def faithfulness_score(answer: str, contexts: List[str]) -> float:
    """
    Approximate faithfulness: fraction of answer sentences supported by context.
    In production: use an NLI model (e.g. cross-encoder/nli-deberta-v3-base).
    """
    if not answer or not contexts:
        return 0.0
    context_text = " ".join(contexts).lower()
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 0.0
    supported = sum(
        1 for s in sentences
        if any(word in context_text for word in s.lower().split() if len(word) > 4)
    )
    return supported / len(sentences)


def answer_relevancy_score(question: str, answer: str) -> float:
    """
    Approximate answer relevancy: keyword overlap between question and answer.
    In production: use question generation + embedding similarity.
    """
    if not answer:
        return 0.0
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    a_words = set(w.lower() for w in answer.split() if len(w) > 3)
    if not q_words:
        return 0.0
    return len(q_words & a_words) / len(q_words)


def context_precision_score(contexts: List[str], ground_truth: str) -> float:
    """Fraction of retrieved contexts that are relevant to the ground truth."""
    if not contexts:
        return 0.0
    gt_words = set(w.lower() for w in ground_truth.split() if len(w) > 3)
    relevant = sum(
        1 for c in contexts
        if len(set(w.lower() for w in c.split() if len(w) > 3) & gt_words) > 2
    )
    return relevant / len(contexts)


def evaluate_rag_pipeline(pipeline_fn, samples: List[EvalSample] = None) -> dict:
    """
    Run full RAGAS evaluation on the RAG pipeline.

    Args:
        pipeline_fn: callable(question) -> (answer, contexts)
        samples: list of EvalSample (default: TEST_SAMPLES)
    Returns:
        dict with per-metric scores and overall mean
    """
    if samples is None:
        samples = TEST_SAMPLES

    faithfulness_scores, relevancy_scores, precision_scores = [], [], []

    for sample in samples:
        answer, contexts = pipeline_fn(sample.question)

        f = faithfulness_score(answer, contexts)
        r = answer_relevancy_score(sample.question, answer)
        p = context_precision_score(contexts, sample.ground_truth)

        faithfulness_scores.append(f)
        relevancy_scores.append(r)
        precision_scores.append(p)

        print(f"Q: {sample.question[:60]}...")
        print(f"  Faithfulness:      {f:.2f}")
        print(f"  Answer Relevancy:  {r:.2f}")
        print(f"  Context Precision: {p:.2f}\n")

    report = {
        "faithfulness":      round(sum(faithfulness_scores) / len(faithfulness_scores), 3),
        "answer_relevancy":  round(sum(relevancy_scores) / len(relevancy_scores), 3),
        "context_precision": round(sum(precision_scores) / len(precision_scores), 3),
    }
    report["overall"] = round(sum(report.values()) / len(report), 3)

    print("=" * 50)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in report.items():
        label = "★ OVERALL" if k == "overall" else k.replace("_", " ").title()
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {label:25} {bar} {v:.3f}")

    return report


if __name__ == "__main__":
    # Demo: evaluate with a dummy pipeline
    def dummy_pipeline(question):
        return (
            "This is a sample answer about " + question.lower(),
            ["RAG combines retrieval with generation.",
             "Dense retrieval uses neural embeddings.",
             "BM25 is a sparse lexical retrieval method."]
        )

    evaluate_rag_pipeline(dummy_pipeline)
