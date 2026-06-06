"""
Dense vector retriever backed by ChromaDB.

Uses a bi-encoder (sentence-transformers) to embed documents and queries.
ChromaDB handles approximate nearest-neighbour search via HNSW.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.chunking.chunker import Chunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Manages embedding + storage in ChromaDB.

    Usage:
        retriever = DenseRetriever(config)
        retriever.index(chunks)
        results = retriever.search("my query", top_k=10)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_docs",
        distance_metric: str = "cosine",
        device: str = "cpu",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.distance_metric = distance_metric

        logger.info("Loading embedding model: %s on %s", model_name, device)
        self._encoder = SentenceTransformer(model_name, device=device)

        logger.info("Connecting to ChromaDB at %s", persist_dir)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        logger.info(
            "ChromaDB collection '%s' has %d documents",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: List[Chunk], show_progress: bool = True) -> None:
        """Embed and upsert chunks into ChromaDB."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                **c.metadata,
            }
            for c in chunks
        ]

        logger.info("Encoding %d chunks with batch_size=%d", len(chunks), self.batch_size)
        embeddings = self._encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # cosine similarity = dot product after L2 norm
        ).tolist()

        # ChromaDB upsert handles duplicates gracefully
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks. Collection size: %d", len(chunks), self._collection.count())

    def delete_collection(self) -> None:
        """Wipe and recreate the collection."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": self.distance_metric},
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Return top_k most similar chunks.

        Returns list of dicts:
            {chunk_id, text, score, doc_id, doc_title, chunk_index, metadata}
        """
        query_embedding = self._encoder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for chunk_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            score = 1.0 - dist
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                    "doc_id": meta.get("doc_id", ""),
                    "doc_title": meta.get("doc_title", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "metadata": {k: v for k, v in meta.items()
                                 if k not in ("doc_id", "doc_title", "chunk_index", "token_count")},
                }
            )
        return hits

    @property
    def count(self) -> int:
        return self._collection.count()
