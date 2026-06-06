# Lazy imports — DenseRetriever requires chromadb which may not be installed
# in all environments. Import directly from submodules as needed.
from .sparse import BM25Retriever
from .hybrid import HybridRetriever, RetrievedChunk

__all__ = ["BM25Retriever", "HybridRetriever", "RetrievedChunk"]

# DenseRetriever is available via: from src.retrieval.dense import DenseRetriever
