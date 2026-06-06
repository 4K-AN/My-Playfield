"""
Document ingestion pipeline.

Reads raw text/markdown/txt files, chunks them, then indexes into both
the dense vector store (ChromaDB) and the BM25 sparse index.

Run as a script:
    python -m src.pipeline.ingest --docs data/ --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from src.chunking.chunker import Chunk, ChunkerConfig, get_chunker
from src.pipeline.config import PipelineConfig, load_config
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst"}


class DocumentIngestor:
    """
    Orchestrates the full ingestion flow:
      read files → chunk → embed (dense) → BM25 index (sparse)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        chunker_cfg = ChunkerConfig(
            strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            min_chunk_size=config.chunking.min_chunk_size,
        )
        self.chunker = get_chunker(chunker_cfg)

        self.dense_retriever = DenseRetriever(
            model_name=config.embedding.model,
            persist_dir=config.vector_db.persist_dir,
            collection_name=config.vector_db.collection_name,
            distance_metric=config.vector_db.distance_metric,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
        )

        self.sparse_retriever = BM25Retriever(
            index_path=config.bm25_index_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_directory(self, docs_dir: str) -> int:
        """Ingest all supported files in a directory. Returns total chunks indexed."""
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Directory not found: {docs_dir}")

        files = [
            p for p in docs_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.warning("No supported files found in %s", docs_dir)
            return 0

        logger.info("Found %d files to ingest in %s", len(files), docs_dir)
        all_chunks: List[Chunk] = []

        for file_path in tqdm(files, desc="Reading files"):
            chunks = self.ingest_file(str(file_path))
            all_chunks.extend(chunks)

        logger.info("Total chunks produced: %d", len(all_chunks))
        self._index_chunks(all_chunks)
        return len(all_chunks)

    def ingest_file(self, file_path: str) -> List[Chunk]:
        """Read a single file and return chunks (does not index them)."""
        path = Path(file_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("Failed to read %s: %s", file_path, e)
            return []

        if not text.strip():
            logger.warning("Skipping empty file: %s", file_path)
            return []

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))
        chunks = self.chunker.chunk(
            text=text,
            doc_id=doc_id,
            doc_title=path.stem,
            metadata={"source": str(path), "file_type": path.suffix},
        )
        logger.debug("File '%s' → %d chunks", path.name, len(chunks))
        return chunks

    def ingest_texts(self, texts: List[str], titles: Optional[List[str]] = None) -> int:
        """Ingest raw strings directly (useful for testing)."""
        titles = titles or [f"doc_{i}" for i in range(len(texts))]
        all_chunks: List[Chunk] = []
        for text, title in zip(texts, titles):
            doc_id = str(uuid.uuid4())
            chunks = self.chunker.chunk(text=text, doc_id=doc_id, doc_title=title)
            all_chunks.extend(chunks)
        self._index_chunks(all_chunks)
        return len(all_chunks)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_chunks(self, chunks: List[Chunk]) -> None:
        logger.info("Indexing %d chunks into dense store...", len(chunks))
        self.dense_retriever.index(chunks)

        logger.info("Indexing %d chunks into BM25...", len(chunks))
        self.sparse_retriever.index(chunks)
        logger.info("Ingestion complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into RAG pipeline")
    parser.add_argument("--docs", required=True, help="Path to documents directory")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--reset", action="store_true", help="Wipe existing index before ingesting")
    args = parser.parse_args()

    config = load_config(args.config)
    ingestor = DocumentIngestor(config)

    if args.reset:
        logger.info("Resetting existing vector store...")
        ingestor.dense_retriever.delete_collection()

    total = ingestor.ingest_directory(args.docs)
    print(f"\n✓ Ingested {total} chunks successfully.")


if __name__ == "__main__":
    main()
