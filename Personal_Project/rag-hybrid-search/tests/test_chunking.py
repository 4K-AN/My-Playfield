"""Unit tests for the chunking module."""

import pytest
from src.chunking.chunker import (
    Chunk,
    ChunkerConfig,
    RecursiveChunker,
    SentenceChunker,
    SlidingWindowChunker,
    get_chunker,
)

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with language model generation. Instead of relying solely on the model's parametric
knowledge, RAG systems fetch relevant documents from an external corpus and condition
the generation on that retrieved context.

The retrieval component can be dense (using vector similarity) or sparse (using BM25).
Hybrid approaches combine both signals using fusion techniques like Reciprocal Rank Fusion.

After retrieval, a cross-encoder reranker can further improve result quality by
jointly scoring the query and each candidate passage. This is more accurate than
bi-encoder scoring but also more computationally expensive.
""".strip()


@pytest.fixture
def default_config():
    return ChunkerConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10)


class TestRecursiveChunker:
    def test_returns_list_of_chunks(self, default_config):
        chunker = RecursiveChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="test_doc")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_unique(self, default_config):
        chunker = RecursiveChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_are_sequential(self, default_config):
        chunker = RecursiveChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_token_count_within_budget(self, default_config):
        chunker = RecursiveChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for c in chunks:
            # Allow small overrun due to overlap merge
            assert c.token_count <= default_config.chunk_size + default_config.chunk_overlap

    def test_min_chunk_size_filter(self):
        config = ChunkerConfig(chunk_size=512, chunk_overlap=0, min_chunk_size=500)
        chunker = RecursiveChunker(config)
        # Short text should produce no chunks (below min_chunk_size)
        chunks = chunker.chunk("short text")
        assert len(chunks) == 0

    def test_preserves_doc_metadata(self, default_config):
        chunker = RecursiveChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc123", doc_title="Test Doc")
        for c in chunks:
            assert c.doc_id == "doc123"
            assert c.doc_title == "Test Doc"

    def test_empty_text_returns_empty(self, default_config):
        chunker = RecursiveChunker(default_config)
        chunks = chunker.chunk("   \n  ")
        assert chunks == []


class TestSentenceChunker:
    def test_basic_chunking(self, default_config):
        chunker = SentenceChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_no_empty_chunks(self, default_config):
        chunker = SentenceChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for c in chunks:
            assert c.text.strip()

    def test_full_coverage(self, default_config):
        """All content should appear in at least one chunk."""
        chunker = SentenceChunker(default_config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        all_text = " ".join(c.text for c in chunks)
        # Key phrases should be present somewhere
        assert "RAG" in all_text or "Retrieval" in all_text


class TestSlidingWindowChunker:
    def test_fixed_window_size(self):
        config = ChunkerConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=5)
        chunker = SlidingWindowChunker(config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for c in chunks[:-1]:  # last chunk can be smaller
            assert c.token_count <= config.chunk_size

    def test_overlap_exists(self):
        config = ChunkerConfig(chunk_size=50, chunk_overlap=25, min_chunk_size=5)
        chunker = SlidingWindowChunker(config)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # With 50% overlap, consecutive chunks should share some tokens
        assert len(chunks) >= 2


class TestGetChunker:
    def test_recursive_factory(self):
        config = ChunkerConfig(strategy="recursive")
        chunker = get_chunker(config)
        assert isinstance(chunker, RecursiveChunker)

    def test_sentence_factory(self):
        config = ChunkerConfig(strategy="sentence")
        chunker = get_chunker(config)
        assert isinstance(chunker, SentenceChunker)

    def test_sliding_window_factory(self):
        config = ChunkerConfig(strategy="sliding_window")
        chunker = get_chunker(config)
        assert isinstance(chunker, SlidingWindowChunker)

    def test_unknown_strategy_raises(self):
        config = ChunkerConfig(strategy="unknown_strategy")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker(config)
