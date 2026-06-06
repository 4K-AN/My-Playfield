"""
Evaluation metrics for RAG systems.

Three metrics are implemented:

1. FaithfulnessEvaluator
   ----------------------
   Measures whether every claim in the generated answer is supported by
   the retrieved context. Uses an LLM judge that decomposes the answer
   into atomic statements and checks each one.

   Score ∈ [0, 1]:  supported_statements / total_statements

2. RelevanceEvaluator (Answer Relevance)
   --------------------------------------
   Measures whether the answer actually addresses the question.
   Uses the cross-encoder to score (question, answer) similarity as a
   proxy, plus an optional LLM judge for nuanced assessment.

   Score ∈ [0, 1]

3. RetrievalEvaluator (Context Recall / Precision)
   -------------------------------------------------
   For offline eval where ground-truth relevant chunks are known.
   - Precision@K: fraction of top-K retrieved that are relevant
   - Recall@K:    fraction of all relevant docs retrieved in top-K
   - MRR:         mean reciprocal rank of first relevant result
   - NDCG@K:      normalised discounted cumulative gain
"""

from __future__ import annotations

import json
import logging
import math
from typing import List, Optional, Set

import openai

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are an expert evaluator. Your task is to assess the FAITHFULNESS of an answer
with respect to the provided context.

FAITHFULNESS means: every statement in the answer must be directly supported by the context.
Statements that introduce information NOT present in the context are considered hallucinations.

Context:
{context}

Answer to evaluate:
{answer}

Instructions:
1. Decompose the answer into individual atomic statements (one claim per line).
2. For each statement, decide if it is SUPPORTED or NOT_SUPPORTED by the context.
3. Return ONLY valid JSON in this exact format:
{{
  "statements": [
    {{"statement": "<text>", "supported": true}},
    {{"statement": "<text>", "supported": false}}
  ]
}}
"""


class FaithfulnessEvaluator:
    """
    LLM-based faithfulness checker.

    Uses an LLM judge to decompose the answer into atomic claims and
    verify each claim against the retrieved context.
    """

    def __init__(self, llm_client: openai.OpenAI, model: str = "gpt-4o-mini"):
        self._client = llm_client
        self._model = model

    def evaluate(self, answer: str, context: str) -> dict:
        """
        Returns:
            {
                score: float,          # 0.0–1.0
                supported: int,
                total: int,
                statements: list[dict]
            }
        """
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            statements = data.get("statements", [])
        except Exception as e:
            logger.error("Faithfulness eval failed: %s", e)
            return {"score": 0.0, "supported": 0, "total": 0, "statements": [], "error": str(e)}

        if not statements:
            return {"score": 1.0, "supported": 0, "total": 0, "statements": []}

        supported = sum(1 for s in statements if s.get("supported", False))
        score = supported / len(statements)

        return {
            "score": score,
            "supported": supported,
            "total": len(statements),
            "statements": statements,
        }


# ---------------------------------------------------------------------------
# 2. Answer Relevance
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """You are an expert evaluator. Rate how well the ANSWER addresses the QUESTION.

Question: {question}
Answer: {answer}

Rate on a scale from 0 to 1:
- 1.0: Answer directly and completely addresses the question
- 0.7: Answer mostly addresses the question with minor gaps
- 0.5: Answer partially addresses the question
- 0.3: Answer is related but doesn't really answer the question
- 0.0: Answer is completely irrelevant or refuses to answer

Return ONLY valid JSON: {{"score": <float>, "reason": "<one sentence explanation>"}}
"""


class RelevanceEvaluator:
    """LLM-based answer relevance evaluator."""

    def __init__(self, llm_client: openai.OpenAI, model: str = "gpt-4o-mini"):
        self._client = llm_client
        self._model = model

    def evaluate(self, question: str, answer: str) -> dict:
        """
        Returns:
            {score: float, reason: str}
        """
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return {
                "score": float(data.get("score", 0.0)),
                "reason": data.get("reason", ""),
            }
        except Exception as e:
            logger.error("Relevance eval failed: %s", e)
            return {"score": 0.0, "reason": "", "error": str(e)}


# ---------------------------------------------------------------------------
# 3. Retrieval Quality (offline, no LLM needed)
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """
    Offline retrieval metrics when ground-truth relevant chunk IDs are known.

    These metrics let you evaluate the retriever independently from the LLM.
    Useful for A/B testing chunking strategies or retrieval parameters.
    """

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Fraction of top-k retrieved that are relevant."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        return sum(1 for rid in top_k if rid in relevant_ids) / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Fraction of all relevant docs that appear in top-k."""
        if not relevant_ids:
            return 1.0
        top_k = retrieved_ids[:k]
        return sum(1 for rid in top_k if rid in relevant_ids) / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Mean Reciprocal Rank — rank of the first relevant result."""
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Normalised Discounted Cumulative Gain at k."""
        def dcg(ids: List[str]) -> float:
            return sum(
                (1.0 / math.log2(rank + 1))
                for rank, rid in enumerate(ids[:k], start=1)
                if rid in relevant_ids
            )

        actual_dcg = dcg(retrieved_ids)
        # Ideal: all relevant docs at the top
        ideal_ids = list(relevant_ids)[:k]
        ideal_dcg = dcg(ideal_ids)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def evaluate_all(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int = 5,
    ) -> dict:
        return {
            f"precision@{k}": self.precision_at_k(retrieved_ids, relevant_ids, k),
            f"recall@{k}": self.recall_at_k(retrieved_ids, relevant_ids, k),
            "mrr": self.mrr(retrieved_ids, relevant_ids),
            f"ndcg@{k}": self.ndcg_at_k(retrieved_ids, relevant_ids, k),
        }
