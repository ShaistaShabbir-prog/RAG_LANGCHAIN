"""
Issue #9: Hybrid retrieval — BM25 keyword + dense vector search combined.
"""
from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger(__name__)


def hybrid_search(query: str, docs: list[str], vectorstore=None,
                  bm25_weight: float = 0.4, dense_weight: float = 0.6,
                  k: int = 4) -> list[dict[str, Any]]:
    """
    Combine BM25 keyword retrieval with FAISS dense vector retrieval.
    Returns top-k results with combined relevance score.
    """
    bm25_results  = _bm25_search(query, docs, k=k)
    dense_results = _dense_search(query, vectorstore, k=k) if vectorstore else []

    # Merge and re-rank using reciprocal rank fusion
    scores: dict[str, float] = {}
    texts:  dict[str, str]   = {}

    for rank, result in enumerate(bm25_results):
        key = result["text"][:100]
        scores[key] = scores.get(key, 0) + bm25_weight * (1 / (rank + 1))
        texts[key]  = result["text"]

    for rank, result in enumerate(dense_results):
        key = result["text"][:100]
        scores[key] = scores.get(key, 0) + dense_weight * (1 / (rank + 1))
        texts[key]  = result["text"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": texts[k], "score": round(v, 4), "retrieval": "hybrid"}
            for k, v in ranked[:k]]


def _bm25_search(query: str, docs: list[str], k: int = 4) -> list[dict]:
    """BM25 keyword retrieval using rank_bm25."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.warning("rank_bm25 not installed — pip install rank-bm25")
        return _simple_keyword_search(query, docs, k)

    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{"text": docs[i], "score": float(scores[i]), "retrieval": "bm25"}
            for i in top_idx if scores[i] > 0]


def _dense_search(query: str, vectorstore, k: int = 4) -> list[dict]:
    """Dense vector retrieval using FAISS vectorstore."""
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        return [{"text": doc.page_content, "score": float(score), "retrieval": "dense"}
                for doc, score in results]
    except Exception as e:
        log.warning("Dense search failed: %s", e)
        return []


def _simple_keyword_search(query: str, docs: list[str], k: int = 4) -> list[dict]:
    """Fallback: simple substring keyword search."""
    terms = query.lower().split()
    scored = [(sum(t in d.lower() for t in terms), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"text": d, "score": float(s), "retrieval": "keyword"}
            for s, d in scored[:k] if s > 0]
