"""
Issue #9: HybridSearch — BM25 keyword + FAISS dense vector ensemble.
"""
from __future__ import annotations
import logging
from typing import Any
log = logging.getLogger(__name__)


def get_hybrid_retriever(vectorstore, docs: list, k: int = 4, bm25_weight: float = 0.4):
    """
    Build an ensemble retriever combining BM25 and FAISS.
    Falls back to FAISS-only if rank_bm25 not installed.
    """
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        bm25 = BM25Retriever.from_documents(docs, k=k)
        faiss_ret = vectorstore.as_retriever(search_kwargs={"k": k})
        return EnsembleRetriever(
            retrievers=[bm25, faiss_ret],
            weights=[bm25_weight, 1 - bm25_weight],
        )
    except ImportError:
        log.warning("rank_bm25 not installed — using FAISS-only. pip install rank_bm25")
        return vectorstore.as_retriever(search_kwargs={"k": k})
