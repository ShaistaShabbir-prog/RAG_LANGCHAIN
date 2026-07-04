"""
rag_project/retrieval/hybrid_search.py

Hybrid retrieval: BM25 (sparse) + Dense embeddings + Reciprocal Rank Fusion.

Research basis:
  Luan et al. (2021) "Sparse, Dense, and Attentional Representations for Text Retrieval"
  — hybrid search outperforms either method alone by 5-15% on standard benchmarks.

Usage:
    from rag_project.retrieval.hybrid_search import HybridRetriever
    retriever = HybridRetriever(documents)
    results = retriever.search("What is the capital of Germany?", top_k=5)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = None


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int
    method: str  # 'bm25', 'dense', or 'hybrid'


class HybridRetriever:
    """
    Combines BM25 sparse retrieval with dense embedding retrieval
    using Reciprocal Rank Fusion (RRF) for score combination.

    Args:
        documents: List of Document objects
        embedding_model: HuggingFace model name for dense embeddings
        rrf_k: RRF constant (default 60, from Cormack et al. 2009)
        bm25_weight: Weight for BM25 scores in fusion (0-1)
        dense_weight: Weight for dense scores in fusion (0-1)
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.documents = documents
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # Build BM25 index
        tokenized = [doc.text.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Build dense index
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(embedding_model)
            texts = [doc.text for doc in documents]
            self.dense_embeddings = self.encoder.encode(
                texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
            )
        except ImportError:
            print("sentence-transformers not installed — dense retrieval disabled")
            self.encoder = None
            self.dense_embeddings = None

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.documents[i], float(scores[i]), rank, "bm25")
                for rank, i in enumerate(top_indices, 1)]

    def _dense_search(self, query: str, top_k: int) -> List[SearchResult]:
        if self.encoder is None:
            return []
        q_emb = self.encoder.encode(query, normalize_embeddings=True)
        scores = (self.dense_embeddings @ q_emb).tolist()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.documents[i], float(scores[i]), rank, "dense")
                for rank, i in enumerate(top_indices, 1)]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Combine rankings using Reciprocal Rank Fusion (RRF)."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result in bm25_results:
            doc_id = result.document.id
            scores[doc_id] = scores.get(doc_id, 0) + self.bm25_weight * (1 / (self.rrf_k + result.rank))
            doc_map[doc_id] = result.document

        for result in dense_results:
            doc_id = result.document.id
            scores[doc_id] = scores.get(doc_id, 0) + self.dense_weight * (1 / (self.rrf_k + result.rank))
            doc_map[doc_id] = result.document

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [SearchResult(doc_map[doc_id], scores[doc_id], rank, "hybrid")
                for rank, doc_id in enumerate(sorted_ids, 1)]

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and dense retrieval.
        Returns top_k results ranked by Reciprocal Rank Fusion.
        """
        bm25_results  = self._bm25_search(query, top_k * 2)
        dense_results = self._dense_search(query, top_k * 2)

        if not dense_results:
            return bm25_results[:top_k]

        return self._reciprocal_rank_fusion(bm25_results, dense_results, top_k)

    def __repr__(self):
        return (f"HybridRetriever(docs={len(self.documents)}, "
                f"bm25_weight={self.bm25_weight}, dense_weight={self.dense_weight})")
