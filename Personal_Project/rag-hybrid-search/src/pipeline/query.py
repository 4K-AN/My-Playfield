"""
End-to-end RAG query pipeline.

Flow:
  query → hybrid retrieval → cross-encoder rerank → LLM answer generation

Run as a script:
    python -m src.pipeline.query --question "What is retrieval-augmented generation?"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import openai

from src.pipeline.config import PipelineConfig, load_config
from src.pipeline.ingest import DocumentIngestor
from src.reranking.reranker import CrossEncoderReranker
from src.retrieval.hybrid import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
- Answer using ONLY information found in the context below.
- If the context does not contain enough information, say "I don't have enough information to answer this question."
- Be concise and factual.
- Cite the source document title when relevant.
- Do NOT make up information not present in the context."""

CONTEXT_TEMPLATE = """<context>
{context_blocks}
</context>

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)  # before reranking
    total_tokens_used: int = 0

    def __repr__(self) -> str:
        return (
            f"RAGResult(\n"
            f"  question='{self.question[:80]}',\n"
            f"  answer='{self.answer[:120]}...',\n"
            f"  sources={[s.doc_title for s in self.sources]}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Full RAG pipeline: retrieve → rerank → generate.

    Usage:
        pipeline = RAGPipeline(config)
        result = pipeline.query("What is BM25?")
        print(result.answer)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        ingestor = DocumentIngestor(config)
        self.dense_retriever = ingestor.dense_retriever
        self.sparse_retriever = ingestor.sparse_retriever

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            dense_top_k=config.retrieval.dense_top_k,
            sparse_top_k=config.retrieval.sparse_top_k,
            rrf_k=config.retrieval.rrf_k,
            alpha=config.retrieval.alpha,
            fusion_method=config.retrieval.fusion_method,
        )

        self.reranker = CrossEncoderReranker(
            model_name=config.reranker.model,
            device=config.reranker.device,
            batch_size=config.reranker.batch_size,
        )

        self._llm_client = openai.OpenAI(api_key=config.llm.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """
        Execute a full RAG query.

        Steps:
          1. Hybrid retrieval (dense + sparse + RRF)
          2. Cross-encoder reranking
          3. Build context string
          4. LLM generation
        """
        final_top_k = top_k or self.config.retrieval.final_top_k

        # Step 1: Hybrid retrieval
        candidate_chunks = self.hybrid_retriever.search(
            question, top_k=self.config.retrieval.dense_top_k
        )
        logger.info("Retrieved %d candidate chunks", len(candidate_chunks))

        # Step 2: Cross-encoder reranking
        final_chunks = self.reranker.rerank(question, candidate_chunks, top_k=final_top_k)
        logger.info("Reranked to %d final chunks", len(final_chunks))

        # Step 3: Build LLM context
        context_blocks = self._build_context(final_chunks)
        prompt = CONTEXT_TEMPLATE.format(
            context_blocks=context_blocks,
            question=question,
        )

        # Step 4: Generate answer
        answer, tokens_used = self._generate(prompt)

        return RAGResult(
            question=question,
            answer=answer,
            sources=final_chunks,
            retrieved_chunks=candidate_chunks,
            total_tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            rerank_score = chunk.metadata.get("rerank_score", 0.0)
            blocks.append(
                f"[{i}] Source: {chunk.doc_title or chunk.doc_id}\n"
                f"    Relevance score: {rerank_score:.3f}\n"
                f"    {chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _generate(self, prompt: str) -> tuple[str, int]:
        """Call the LLM and return (answer_text, tokens_used)."""
        try:
            response = self._llm_client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0
            return answer, tokens
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return f"[Error generating answer: {e}]", 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top-k", type=int, default=None, help="Override final top_k")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    pipeline = RAGPipeline(config)
    result = pipeline.query(args.question, top_k=args.top_k)

    print("\n" + "=" * 60)
    print(f"QUESTION: {result.question}")
    print("=" * 60)
    print(f"\nANSWER:\n{result.answer}")
    print("\nSOURCES:")
    for i, src in enumerate(result.sources, 1):
        score = src.metadata.get("rerank_score", 0.0)
        print(f"  [{i}] {src.doc_title or src.doc_id} (score: {score:.3f})")
    print(f"\nTokens used: {result.total_tokens_used}")


if __name__ == "__main__":
    main()
