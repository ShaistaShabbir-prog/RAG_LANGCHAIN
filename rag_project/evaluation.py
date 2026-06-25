"""
Issue #4: RAGAS evaluation metrics for RAG answer quality.
"""
import logging
log = logging.getLogger(__name__)


def run_evaluation(test_pairs: list[dict]) -> dict:
    """
    Evaluate RAG pipeline using RAGAS metrics.

    Args:
        test_pairs: list of {"question": str, "ground_truth": str}

    Returns:
        dict with faithfulness, answer_relevancy, context_precision scores.

    Install: pip install ragas
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        from rag_project.chain import get_chain
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Run: pip install ragas datasets"}

    chain = get_chain()
    records = []
    for pair in test_pairs:
        q = pair["question"]
        result = chain.invoke({"question": q})
        records.append({
            "question":     q,
            "answer":       result.get("answer", ""),
            "contexts":     [d.page_content for d in result.get("source_documents", [])],
            "ground_truth": pair.get("ground_truth", ""),
        })

    ds = Dataset.from_list(records)
    try:
        scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        return {
            "faithfulness":       round(float(scores["faithfulness"]),       3),
            "answer_relevancy":   round(float(scores["answer_relevancy"]),   3),
            "context_precision":  round(float(scores["context_precision"]),  3),
            "overall":            round(sum([scores["faithfulness"],
                                             scores["answer_relevancy"],
                                             scores["context_precision"]]) / 3, 3),
            "samples_evaluated":  len(records),
        }
    except Exception as e:
        log.error("RAGAS evaluation failed: %s", e)
        return {"error": str(e)}
