"""Unit tests for retrieval modules (BM25, dense mock, hybrid fusion)."""

import pytest
from unittest.mock import MagicMock, patch

from src.chunking.chunker import Chunk
# Import modules directly to avoid triggering chromadb in __init__.py
from src.retrieval.sparse import BM25Retriever, _tokenize
from src.retrieval.hybrid import HybridRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            text=t,
            chunk_id=f"chunk_{i}",
            doc_id=f"doc_{i}",
            doc_title=f"Document {i}",
            chunk_index=i,
            token_count=len(t.split()),
        )
        for i, t in enumerate(texts)
    ]


CORPUS = [
    "BM25 is a bag-of-words retrieval function that ranks documents based on the query terms.",
    "Dense retrieval uses neural embeddings to find semantically similar documents.",
    "Hybrid search combines sparse and dense retrieval for better results.",
    "Reciprocal Rank Fusion merges multiple ranked lists into a single ranking.",
    "Cross-encoder rerankers provide more accurate relevance scoring than bi-encoders.",
    "Chunking strategies affect the quality of retrieved context in RAG systems.",
    "Vector databases like ChromaDB store embeddings for fast nearest-neighbour search.",
]


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_lowercase(self):
        tokens = _tokenize("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_punctuation_removed(self):
        tokens = _tokenize("hello, world! foo.")
        assert "," not in tokens
        assert "." not in tokens
        assert "!" not in tokens

    def test_splits_on_whitespace(self):
        tokens = _tokenize("one two three")
        assert tokens == ["one", "two", "three"]

    def test_empty_string(self):
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# BM25 retriever tests
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    @pytest.fixture
    def retriever(self):
        r = BM25Retriever()
        r.index(_make_chunks(CORPUS))
        return r

    def test_index_count(self, retriever):
        assert retriever.count == len(CORPUS)

    def test_search_returns_list(self, retriever):
        results = retriever.search("BM25 retrieval", top_k=3)
        assert isinstance(results, list)

    def test_search_top_k_respected(self, retriever):
        results = retriever.search("retrieval", top_k=3)
        assert len(results) <= 3

    def test_search_relevance(self, retriever):
        """BM25 document about BM25 should rank first for 'BM25 query'."""
        results = retriever.search("BM25 bag of words", top_k=5)
        assert len(results) > 0
        # The chunk about BM25 should be in the top results
        top_texts = [r["text"] for r in results[:3]]
        assert any("BM25" in t for t in top_texts)

    def test_search_returns_expected_fields(self, retriever):
        results = retriever.search("hybrid search", top_k=1)
        assert results
        r = results[0]
        assert "chunk_id" in r
        assert "text" in r
        assert "score" in r
        assert "doc_id" in r

    def test_empty_index_returns_empty(self):
        retriever = BM25Retriever()
        results = retriever.search("anything")
        assert results == []

    def test_scores_non_negative(self, retriever):
        results = retriever.search("some query", top_k=5)
        for r in results:
            assert r["score"] >= 0.0

    def test_scores_descending(self, retriever):
        results = retriever.search("dense retrieval", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Hybrid retriever tests (with mocked dense retriever)
# ---------------------------------------------------------------------------

def _make_hit(chunk_id: str, text: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "doc_id": chunk_id,
        "doc_title": chunk_id,
        "chunk_index": 0,
        "metadata": {},
    }


class TestHybridRetriever:
    @pytest.fixture
    def mock_dense(self):
        dense = MagicMock()
        dense.search.return_value = [
            _make_hit("c1", "Dense result 1", 0.95),
            _make_hit("c2", "Dense result 2", 0.80),
            _make_hit("c3", "Dense result 3", 0.75),
            _make_hit("c4", "Dense result 4", 0.60),
        ]
        return dense

    @pytest.fixture
    def mock_sparse(self):
        sparse = MagicMock()
        sparse.search.return_value = [
            _make_hit("c3", "Sparse result 3", 15.0),   # overlap with dense
            _make_hit("c5", "Sparse-only result", 12.0),
            _make_hit("c1", "Sparse result 1", 10.0),   # overlap with dense
            _make_hit("c6", "Another sparse", 5.0),
        ]
        return sparse

    @pytest.fixture
    def hybrid(self, mock_dense, mock_sparse):
        return HybridRetriever(
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            dense_top_k=4,
            sparse_top_k=4,
            rrf_k=60,
        )

    def test_returns_list_of_retrieved_chunks(self, hybrid):
        results = hybrid.search("test query", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_top_k_respected(self, hybrid):
        results = hybrid.search("test query", top_k=3)
        assert len(results) <= 3

    def test_fusion_merges_unique_chunks(self, hybrid):
        results = hybrid.search("test query", top_k=10)
        ids = [r.chunk_id for r in results]
        # Should include chunks from both dense and sparse
        assert "c1" in ids or "c3" in ids
        assert "c5" in ids  # sparse-only

    def test_rrf_scores_descending(self, hybrid):
        results = hybrid.search("test query", top_k=10)
        scores = [r.fusion_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_dense_and_sparse_rank_populated(self, hybrid):
        results = hybrid.search("test query", top_k=10)
        # c1 appears in both → should have both ranks populated
        c1 = next((r for r in results if r.chunk_id == "c1"), None)
        assert c1 is not None
        assert c1.dense_rank is not None
        assert c1.sparse_rank is not None

    def test_sparse_only_chunk_has_no_dense_rank(self, hybrid):
        results = hybrid.search("test query", top_k=10)
        c5 = next((r for r in results if r.chunk_id == "c5"), None)
        assert c5 is not None
        assert c5.dense_rank is None
        assert c5.sparse_rank is not None

    def test_weighted_fusion(self, mock_dense, mock_sparse):
        hybrid = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion_method="weighted",
            alpha=0.7,
        )
        results = hybrid.search("test query", top_k=5)
        assert len(results) > 0
        scores = [r.fusion_score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Retrieval metrics tests
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    def test_precision_perfect(self):
        from src.eval.metrics import RetrievalEvaluator
        ev = RetrievalEvaluator()
        assert ev.precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_precision_zero(self):
        from src.eval.metrics import RetrievalEvaluator
        ev = RetrievalEvaluator()
        assert ev.precision_at_k(["a", "b", "c"], {"x", "y", "z"}, k=3) == 0.0

    def test_recall_perfect(self):
        from src.eval.metrics import RetrievalEvaluator
        ev = RetrievalEvaluator()
        assert ev.recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_mrr_first_hit(self):
        from src.eval.metrics import RetrievalEvaluator
        ev = RetrievalEvaluator()
        assert ev.mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_mrr_third_hit(self):
        from src.eval.metrics import RetrievalEvaluator
        ev = RetrievalEvaluator()
        assert abs(ev.mrr(["x", "y", "a"], {"a"}) - 1/3) < 1e-9

    def test_mrr_no_hit(self):
        from src.eval.metrics import RetrievalEvaluator
        ev = RetrievalEvaluator()
        assert ev.mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_ndcg_perfect(self):
        from src.eval.metrics import RetrievalEvaluator
        ev = RetrievalEvaluator()
        score = ev.ndcg_at_k(["a", "b"], {"a", "b"}, k=2)
        assert abs(score - 1.0) < 1e-9
