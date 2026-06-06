"""
Chunking strategies for RAG pipelines.

Three strategies are provided:
- RecursiveChunker  : splits on paragraph → sentence → word boundaries
                      (best general-purpose choice)
- SentenceChunker   : groups complete sentences up to a token budget
                      (preserves semantic units, good for QA)
- SlidingWindowChunker : fixed-size windows with overlap
                      (guarantees no context is lost at boundaries)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import tiktoken


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    doc_title: str = ""
    chunk_index: int = 0          # position within source document
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(id={self.chunk_id[:8]}, tokens={self.token_count}, text='{preview}...')"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ChunkerConfig:
    strategy: Literal["recursive", "sentence", "sliding_window"] = "recursive"
    chunk_size: int = 512          # max tokens per chunk
    chunk_overlap: int = 64        # overlap in tokens
    min_chunk_size: int = 50       # discard smaller chunks
    encoding_name: str = "cl100k_base"  # tiktoken encoding


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

class TokenCounter:
    """Wraps tiktoken for fast token counting."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseChunker:
    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.counter = TokenCounter(config.encoding_name)

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        doc_title: str = "",
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        raise NotImplementedError

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        index: int,
        start_char: int,
        end_char: int,
        metadata: dict,
    ) -> Optional[Chunk]:
        text = text.strip()
        token_count = self.counter.count(text)
        if token_count < self.config.min_chunk_size:
            return None
        return Chunk(
            text=text,
            doc_id=doc_id,
            doc_title=doc_title,
            chunk_index=index,
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Strategy 1: Recursive chunker
# ---------------------------------------------------------------------------

class RecursiveChunker(BaseChunker):
    """
    Tries to split on natural boundaries in order:
    double-newline (paragraphs) → single newline → sentence → word

    This is the default strategy — it produces the most coherent chunks
    because it respects document structure.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        doc_title: str = "",
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        raw_splits = self._split_recursive(text, self.SEPARATORS)
        chunks: List[Chunk] = []
        buffer = ""
        buffer_start = 0
        char_offset = 0

        for split in raw_splits:
            candidate = (buffer + " " + split).strip() if buffer else split
            if self.counter.count(candidate) <= self.config.chunk_size:
                buffer = candidate
            else:
                # flush buffer
                if buffer:
                    c = self._make_chunk(
                        buffer, doc_id, doc_title, len(chunks),
                        buffer_start, buffer_start + len(buffer), metadata
                    )
                    if c:
                        chunks.append(c)
                    # carry overlap into next buffer
                    overlap_text = self._tail_tokens(buffer, self.config.chunk_overlap)
                    buffer_start = buffer_start + len(buffer) - len(overlap_text)
                    buffer = (overlap_text + " " + split).strip()
                else:
                    # split is itself too big — force-split by tokens
                    for sub in self._split_by_tokens(split):
                        c = self._make_chunk(
                            sub, doc_id, doc_title, len(chunks),
                            char_offset, char_offset + len(sub), metadata
                        )
                        if c:
                            chunks.append(c)
                        char_offset += len(sub)
                    buffer = ""
            char_offset += len(split)

        if buffer:
            c = self._make_chunk(
                buffer, doc_id, doc_title, len(chunks),
                buffer_start, buffer_start + len(buffer), metadata
            )
            if c:
                chunks.append(c)

        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        if sep == "":
            return list(text)
        parts = text.split(sep)
        result = []
        for p in parts:
            if self.counter.count(p) > self.config.chunk_size:
                result.extend(self._split_recursive(p, separators[1:]))
            else:
                if p.strip():
                    result.append(p)
        return result

    def _tail_tokens(self, text: str, n_tokens: int) -> str:
        """Return the last n_tokens worth of text."""
        tokens = self.counter.encode(text)
        return self.counter.decode(tokens[-n_tokens:]) if len(tokens) > n_tokens else text

    def _split_by_tokens(self, text: str) -> list[str]:
        """Hard split a long string into token-budget chunks."""
        tokens = self.counter.encode(text)
        step = self.config.chunk_size - self.config.chunk_overlap
        parts = []
        for i in range(0, len(tokens), step):
            parts.append(self.counter.decode(tokens[i: i + self.config.chunk_size]))
        return parts


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-aware chunker
# ---------------------------------------------------------------------------

class SentenceChunker(BaseChunker):
    """
    Splits on sentence boundaries (regex-based, no nltk dependency),
    then groups sentences greedily until the token budget is full.

    Best when you need semantically complete units (QA, summarisation).
    """

    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        doc_title: str = "",
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        sentences = self._SENTENCE_END.split(text.strip())
        chunks: List[Chunk] = []
        buffer_sentences: list[str] = []
        buffer_tokens = 0

        for sent in sentences:
            sent_tokens = self.counter.count(sent)
            if buffer_tokens + sent_tokens > self.config.chunk_size and buffer_sentences:
                chunk_text = " ".join(buffer_sentences)
                c = self._make_chunk(
                    chunk_text, doc_id, doc_title, len(chunks), 0, len(chunk_text), metadata
                )
                if c:
                    chunks.append(c)
                # keep overlap sentences
                overlap: list[str] = []
                overlap_tokens = 0
                for s in reversed(buffer_sentences):
                    t = self.counter.count(s)
                    if overlap_tokens + t > self.config.chunk_overlap:
                        break
                    overlap.insert(0, s)
                    overlap_tokens += t
                buffer_sentences = overlap + [sent]
                buffer_tokens = overlap_tokens + sent_tokens
            else:
                buffer_sentences.append(sent)
                buffer_tokens += sent_tokens

        if buffer_sentences:
            chunk_text = " ".join(buffer_sentences)
            c = self._make_chunk(
                chunk_text, doc_id, doc_title, len(chunks), 0, len(chunk_text), metadata
            )
            if c:
                chunks.append(c)

        return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Sliding window chunker
# ---------------------------------------------------------------------------

class SlidingWindowChunker(BaseChunker):
    """
    Token-level sliding window with fixed stride.
    Guarantees all content is covered with overlap.
    Useful when you cannot afford to miss any context (legal, medical).
    """

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        doc_title: str = "",
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        tokens = self.counter.encode(text)
        step = self.config.chunk_size - self.config.chunk_overlap
        chunks: List[Chunk] = []

        for i in range(0, len(tokens), step):
            window = tokens[i: i + self.config.chunk_size]
            chunk_text = self.counter.decode(window)
            c = self._make_chunk(
                chunk_text, doc_id, doc_title, len(chunks),
                i, i + len(window), metadata
            )
            if c:
                chunks.append(c)

        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_chunker(config: ChunkerConfig) -> BaseChunker:
    """Return the right chunker for the configured strategy."""
    mapping = {
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
        "sliding_window": SlidingWindowChunker,
    }
    cls = mapping.get(config.strategy)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: {config.strategy!r}")
    return cls(config)
