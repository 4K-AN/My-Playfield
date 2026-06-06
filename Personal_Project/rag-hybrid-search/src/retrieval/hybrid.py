"""
Hybrid retriever using Reciprocal Rank Fusion (RRF).

Why RRF?
--------
Dense and sparse retrievers return scores on completely different scales —
cosine similarity ∈ [−1, 1] vs BM25 ∈ [0, ∞). Simple score averaging
is unreliable. RRF fuses ranked lists instead of raw scores:

    RRF_score(d) = Σ  1 / (k + rank_i(d))
                   i

where k (default 60) smooths the influence of high-ranked documents.

This is parameter-free in the sense that k=60 works well empirically
across most retrieval benchmarks (Cormack et al., 2009).

We also support a simpler weighted linear blend (alpha) for comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .dense import DenseRetriever
    from .sparse import BM25Retriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Unified result object from the hybrid retriever."""

    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    chunk_index: int
    dense_rank: Optional[int] = None    # rank in dense results (1-based)
    sparse_rank: Optional[int] = None   # rank in sparse results (1-based)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fusion_score: float = 0.0          # RRF or weighted blend score
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievedChunk(id={self.chunk_id[:8]}, "
            f"fusion={self.fusion_score:.4f}, text='{preview}...')"
        )


class HybridRetriever:
    """
    Combines DenseRetriever + BM25Retriever with RRF fusion.

    Parameters
    ----------
    dense_retriever  : DenseRetriever
    sparse_retriever : BM25Retriever
    dense_top_k      : candidates pulled from dense search
    sparse_top_k     : candidates pulled from BM25 search
    rrf_k            : RRF smoothing constant (default 60)
    alpha            : weight for weighted blend (0=sparse only, 1=dense only)
                       Only used when fusion_method='weighted'
    fusion_method    : 'rrf' (default) | 'weighted'
    """

    def __init__(
        self,
        dense_retriever,   # DenseRetriever
        sparse_retriever,  # BM25Retriever
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.fusion_method = fusion_method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """
        Run hybrid search and return top_k fused results.

        Steps:
          1. Dense search  → ranked list D
          2. Sparse search → ranked list S
          3. Merge all unique chunks
          4. Compute fusion score (RRF or weighted)
          5. Sort descending, return top_k
        """
        dense_hits = self.dense.search(query, top_k=self.dense_top_k)
        sparse_hits = self.sparse.search(query, top_k=self.sparse_top_k)

        logger.debug(
            "Dense hits: %d | Sparse hits: %d", len(dense_hits), len(sparse_hits)
        )

        # Build lookup by chunk_id
        chunk_map: Dict[str, RetrievedChunk] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["chunk_id"]
            chunk_map[cid] = RetrievedChunk(
                chunk_id=cid,
                text=hit["text"],
                doc_id=hit["doc_id"],
                doc_title=hit["doc_title"],
                chunk_index=hit["chunk_index"],
                dense_rank=rank,
                dense_score=hit["score"],
                metadata=hit.get("metadata", {}),
            )

        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit["chunk_id"]
            if cid in chunk_map:
                chunk_map[cid].sparse_rank = rank
                chunk_map[cid].sparse_score = hit["score"]
            else:
                chunk_map[cid] = RetrievedChunk(
                    chunk_id=cid,
                    text=hit["text"],
                    doc_id=hit["doc_id"],
                    doc_title=hit["doc_title"],
                    chunk_index=hit["chunk_index"],
                    sparse_rank=rank,
                    sparse_score=hit["score"],
                    metadata=hit.get("metadata", {}),
                )

        # Compute fusion scores
        if self.fusion_method == "rrf":
            self._apply_rrf(chunk_map)
        else:
            self._apply_weighted(chunk_map, dense_hits, sparse_hits)

        ranked = sorted(chunk_map.values(), key=lambda c: c.fusion_score, reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Fusion methods
    # ------------------------------------------------------------------

    def _apply_rrf(self, chunk_map: Dict[str, RetrievedChunk]) -> None:
        """Reciprocal Rank Fusion score."""
        k = self.rrf_k
        for chunk in chunk_map.values():
            score = 0.0
            if chunk.dense_rank is not None:
                score += 1.0 / (k + chunk.dense_rank)
            if chunk.sparse_rank is not None:
                score += 1.0 / (k + chunk.sparse_rank)
            chunk.fusion_score = score

    def _apply_weighted(
        self,
        chunk_map: Dict[str, RetrievedChunk],
        dense_hits: List[dict],
        sparse_hits: List[dict],
    ) -> None:
        """
        Weighted linear combination after min-max normalisation.
        alpha=1.0 → pure dense, alpha=0.0 → pure sparse.
        """
        def _normalize(hits: List[dict]) -> Dict[str, float]:
            if not hits:
                return {}
            scores = [h["score"] for h in hits]
            mn, mx = min(scores), max(scores)
            rng = mx - mn if mx != mn else 1.0
            return {h["chunk_id"]: (h["score"] - mn) / rng for h in hits}

        dense_norm = _normalize(dense_hits)
        sparse_norm = _normalize(sparse_hits)

        for cid, chunk in chunk_map.items():
            d = dense_norm.get(cid, 0.0)
            s = sparse_norm.get(cid, 0.0)
            chunk.fusion_score = self.alpha * d + (1.0 - self.alpha) * s
