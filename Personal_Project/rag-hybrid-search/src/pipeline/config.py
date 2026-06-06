"""
Pipeline configuration — loaded from YAML or constructed programmatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 64


@dataclass
class ChunkingConfig:
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50


@dataclass
class VectorDBConfig:
    provider: str = "chromadb"
    persist_dir: str = "./chroma_db"
    collection_name: str = "rag_docs"
    distance_metric: str = "cosine"


@dataclass
class RetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    alpha: float = 0.5
    fusion_method: str = "rrf"


@dataclass
class RerankerConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class EvalConfig:
    faithfulness_threshold: float = 0.7
    relevance_threshold: float = 0.7
    sample_size: int = 100


@dataclass
class PipelineConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    bm25_index_path: Optional[str] = "./bm25_index.pkl"


def load_config(path: str = "configs/config.yaml") -> PipelineConfig:
    """Load config from YAML, falling back to defaults for missing keys."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at '{path}'. "
            "Copy configs/config.example.yaml → configs/config.yaml and fill in your API keys."
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    def _get(d: dict, *keys, default=None):
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, {})
        return d if d != {} else default

    cfg = PipelineConfig(
        llm=LLMConfig(
            provider=_get(raw, "llm", "provider", default="openai"),
            model=_get(raw, "llm", "model", default="gpt-4o-mini"),
            temperature=_get(raw, "llm", "temperature", default=0.0),
            max_tokens=_get(raw, "llm", "max_tokens", default=1024),
            api_key=_resolve_env(raw.get("llm", {}).get("api_key", "")),
        ),
        embedding=EmbeddingConfig(
            model=_get(raw, "embedding", "model", default="all-MiniLM-L6-v2"),
            device=_get(raw, "embedding", "device", default="cpu"),
            batch_size=_get(raw, "embedding", "batch_size", default=64),
        ),
        chunking=ChunkingConfig(
            strategy=_get(raw, "chunking", "strategy", default="recursive"),
            chunk_size=_get(raw, "chunking", "chunk_size", default=512),
            chunk_overlap=_get(raw, "chunking", "chunk_overlap", default=64),
            min_chunk_size=_get(raw, "chunking", "min_chunk_size", default=50),
        ),
        vector_db=VectorDBConfig(
            persist_dir=_get(raw, "vector_db", "persist_dir", default="./chroma_db"),
            collection_name=_get(raw, "vector_db", "collection_name", default="rag_docs"),
            distance_metric=_get(raw, "vector_db", "distance_metric", default="cosine"),
        ),
        retrieval=RetrievalConfig(
            dense_top_k=_get(raw, "retrieval", "dense_top_k", default=20),
            sparse_top_k=_get(raw, "retrieval", "sparse_top_k", default=20),
            final_top_k=_get(raw, "retrieval", "final_top_k", default=5),
            rrf_k=_get(raw, "retrieval", "rrf_k", default=60),
            alpha=_get(raw, "retrieval", "alpha", default=0.5),
        ),
        reranker=RerankerConfig(
            model=_get(raw, "reranker", "model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=_get(raw, "reranker", "device", default="cpu"),
            batch_size=_get(raw, "reranker", "batch_size", default=32),
        ),
        eval=EvalConfig(
            faithfulness_threshold=_get(raw, "eval", "faithfulness_threshold", default=0.7),
            relevance_threshold=_get(raw, "eval", "relevance_threshold", default=0.7),
        ),
    )
    return cfg


def _resolve_env(value: str) -> str:
    """Replace ${ENV_VAR} placeholders with actual env values."""
    if value and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.getenv(env_key, "")
    return value or ""
