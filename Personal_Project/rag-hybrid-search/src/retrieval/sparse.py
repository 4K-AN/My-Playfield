"""
Sparse BM25 retriever.

BM25 (Best Match 25) scores documents based on term frequency and inverse
document frequency. It handles exact keyword matches that dense search often
misses — especially for rare terms, proper nouns, and technical jargon.

This module wraps rank_bm25 and adds:
- simple tokenisation (lowercase + punctuation removal)
- serialisation to disk so the index survives restarts
"""

from __future__ import annotations

import logging
import pickle
import re
import string
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.chunking.chunker import Chunk

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"[" + re.escape(string.punctuation) + r"]")


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = _PUNCT.sub(" ", text.lower())
    return text.split()


class BM25Retriever:
    """
    In-memory BM25 index with optional disk persistence.

    Usage:
        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.search("my query", top_k=10)
    """

    def __init__(self, index_path: Optional[str] = None):
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[Chunk] = []
        self._index_path = Path(index_path) if index_path else None

        if self._index_path and self._index_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: List[Chunk]) -> None:
        """Build BM25 index from a list of chunks."""
        if not chunks:
            return

        self._chunks = chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        logger.info("Building BM25 index over %d documents...", len(chunks))
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built.")

        if self._index_path:
            self._save()

    def add(self, chunks: List[Chunk]) -> None:
        """Incrementally add chunks and rebuild the index."""
        self._chunks.extend(chunks)
        self.index(self._chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Return top_k chunks ranked by BM25 score.

        Returns list of dicts:
            {chunk_id, text, score, doc_id, doc_title, chunk_index, metadata}

        BM25 scores are non-negative floats; higher is more relevant.
        """
        if self._bm25 is None or not self._chunks:
            logger.warning("BM25 index is empty. Call .index() first.")
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by descending score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        hits = []
        for idx in ranked_indices:
            chunk = self._chunks[idx]
            hits.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": float(scores[idx]),
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                }
            )
        return hits

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
        logger.info("BM25 index saved to %s", self._index_path)

    def _load(self) -> None:
        with open(self._index_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        logger.info(
            "BM25 index loaded from %s (%d docs)", self._index_path, len(self._chunks)
        )

    @property
    def count(self) -> int:
        return len(self._chunks)
