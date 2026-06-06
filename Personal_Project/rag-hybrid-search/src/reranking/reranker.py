"""
Cross-encoder reranker.

Why a cross-encoder?
--------------------
Bi-encoders (used for dense retrieval) encode query and document
independently — fast but they miss fine-grained query-document
interactions. A cross-encoder takes [query, document] as a single input
and outputs a relevance score directly. Much more accurate, but O(n)
inference cost — so we only run it on the top candidates from the
hybrid retriever (typically 20–40 chunks), not the whole corpus.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS-MARCO passage ranking
- 22M parameters, fast inference
- Scores ∈ (-∞, +∞); higher = more relevant
"""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import CrossEncoder

from src.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks a list of RetrievedChunk objects using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, chunks, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        logger.info("Loading cross-encoder model: %s on %s", model_name, device)
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )
        self.batch_size = batch_size
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Score all (query, chunk) pairs and return top_k by cross-encoder score.

        The original fusion_score is preserved. A new field `rerank_score` is
        added to each chunk.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        logger.debug("Cross-encoder scoring %d pairs...", len(pairs))

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 20,
        )

        # Annotate chunks with rerank score
        for chunk, score in zip(chunks, scores):
            chunk.metadata["rerank_score"] = float(score)

        # Sort by cross-encoder score (descending)
        reranked = sorted(
            chunks,
            key=lambda c: c.metadata.get("rerank_score", 0.0),
            reverse=True,
        )

        logger.debug(
            "Top reranked scores: %s",
            [f"{c.metadata['rerank_score']:.3f}" for c in reranked[:5]],
        )

        return reranked[:top_k]

    def score(self, query: str, text: str) -> float:
        """Score a single (query, passage) pair. Useful for eval."""
        return float(self._model.predict([(query, text)])[0])
