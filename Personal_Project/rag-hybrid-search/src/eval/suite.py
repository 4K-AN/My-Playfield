"""
Evaluation suite — runs the full set of metrics over a test dataset.

An eval dataset is a list of EvalSamples:
    {question, expected_answer (optional), relevant_chunk_ids (optional)}

The suite queries the RAG pipeline for each sample and computes:
    - faithfulness  (LLM judge)
    - answer relevance (LLM judge)
    - retrieval precision/recall/MRR/NDCG (if relevant_chunk_ids provided)

Results are returned as an EvalReport with per-sample breakdowns and
aggregate statistics.

Usage:
    suite = EvalSuite(pipeline, config)
    report = suite.run(samples)
    report.print_summary()
    report.to_csv("results.csv")
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set

import openai

from src.eval.metrics import FaithfulnessEvaluator, RelevanceEvaluator, RetrievalEvaluator
from src.pipeline.config import EvalConfig
from src.pipeline.query import RAGPipeline, RAGResult
from src.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One test case in the evaluation dataset."""
    question: str
    expected_answer: Optional[str] = None           # for reference (not used in scoring)
    relevant_chunk_ids: Optional[Set[str]] = None   # for retrieval metrics
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalSample":
        return cls(
            question=d["question"],
            expected_answer=d.get("expected_answer"),
            relevant_chunk_ids=set(d["relevant_chunk_ids"]) if d.get("relevant_chunk_ids") else None,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def load_jsonl(cls, path: str) -> List["EvalSample"]:
        """Load samples from a JSONL file (one JSON object per line)."""
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(cls.from_dict(json.loads(line)))
        return samples


@dataclass
class SampleResult:
    """Evaluation results for a single sample."""
    question: str
    answer: str
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_mrr: float = 0.0
    retrieval_ndcg: float = 0.0
    faithfulness_details: dict = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def passes(self, faithfulness_threshold: float, relevance_threshold: float) -> bool:
        return (
            self.faithfulness_score >= faithfulness_threshold
            and self.relevance_score >= relevance_threshold
        )


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    samples: List[SampleResult]
    config: EvalConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Aggregate stats (computed in __post_init__)
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg: float = 0.0
    pass_rate: float = 0.0

    def __post_init__(self) -> None:
        self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        n = len(self.samples)
        if n == 0:
            return
        self.avg_faithfulness = sum(s.faithfulness_score for s in self.samples) / n
        self.avg_relevance = sum(s.relevance_score for s in self.samples) / n
        self.avg_precision = sum(s.retrieval_precision for s in self.samples) / n
        self.avg_recall = sum(s.retrieval_recall for s in self.samples) / n
        self.avg_mrr = sum(s.retrieval_mrr for s in self.samples) / n
        self.avg_ndcg = sum(s.retrieval_ndcg for s in self.samples) / n
        passing = sum(
            1 for s in self.samples
            if s.passes(self.config.faithfulness_threshold, self.config.relevance_threshold)
        )
        self.pass_rate = passing / n

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("  RAG EVALUATION REPORT")
        print("=" * 55)
        print(f"  Samples evaluated : {len(self.samples)}")
        print(f"  Timestamp         : {self.timestamp}")
        print("-" * 55)
        print(f"  Faithfulness      : {self.avg_faithfulness:.3f}  "
              f"(threshold: {self.config.faithfulness_threshold})")
        print(f"  Answer Relevance  : {self.avg_relevance:.3f}  "
              f"(threshold: {self.config.relevance_threshold})")
        print("-" * 55)
        print(f"  Retrieval Precision@5: {self.avg_precision:.3f}")
        print(f"  Retrieval Recall@5   : {self.avg_recall:.3f}")
        print(f"  MRR                  : {self.avg_mrr:.3f}")
        print(f"  NDCG@5               : {self.avg_ndcg:.3f}")
        print("-" * 55)
        print(f"  Pass Rate         : {self.pass_rate:.1%}")
        print("=" * 55 + "\n")

    def to_csv(self, path: str) -> None:
        """Write per-sample results to a CSV file."""
        fieldnames = [
            "question", "answer", "faithfulness_score", "relevance_score",
            "retrieval_precision", "retrieval_recall", "retrieval_mrr",
            "retrieval_ndcg", "sources", "error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.samples:
                writer.writerow({
                    "question": s.question,
                    "answer": s.answer,
                    "faithfulness_score": round(s.faithfulness_score, 4),
                    "relevance_score": round(s.relevance_score, 4),
                    "retrieval_precision": round(s.retrieval_precision, 4),
                    "retrieval_recall": round(s.retrieval_recall, 4),
                    "retrieval_mrr": round(s.retrieval_mrr, 4),
                    "retrieval_ndcg": round(s.retrieval_ndcg, 4),
                    "sources": "; ".join(s.sources),
                    "error": s.error or "",
                })
        logger.info("Eval results saved to %s", path)

    def to_json(self, path: str) -> None:
        data = {
            "timestamp": self.timestamp,
            "summary": {
                "avg_faithfulness": self.avg_faithfulness,
                "avg_relevance": self.avg_relevance,
                "avg_precision": self.avg_precision,
                "avg_recall": self.avg_recall,
                "avg_mrr": self.avg_mrr,
                "avg_ndcg": self.avg_ndcg,
                "pass_rate": self.pass_rate,
            },
            "samples": [
                {
                    "question": s.question,
                    "answer": s.answer,
                    "faithfulness_score": s.faithfulness_score,
                    "relevance_score": s.relevance_score,
                    "retrieval_precision": s.retrieval_precision,
                    "retrieval_recall": s.retrieval_recall,
                    "retrieval_mrr": s.retrieval_mrr,
                    "retrieval_ndcg": s.retrieval_ndcg,
                    "sources": s.sources,
                    "error": s.error,
                }
                for s in self.samples
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

class EvalSuite:
    """
    Orchestrates evaluation over a list of EvalSamples.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        eval_config: EvalConfig,
        llm_client: Optional[openai.OpenAI] = None,
        llm_model: str = "gpt-4o-mini",
    ):
        self.pipeline = pipeline
        self.eval_config = eval_config

        client = llm_client or openai.OpenAI(api_key=pipeline.config.llm.api_key)
        self.faithfulness_eval = FaithfulnessEvaluator(client, model=llm_model)
        self.relevance_eval = RelevanceEvaluator(client, model=llm_model)
        self.retrieval_eval = RetrievalEvaluator()

    def run(
        self,
        samples: List[EvalSample],
        k: int = 5,
        verbose: bool = True,
    ) -> EvalReport:
        """
        Run all metrics on each sample.

        Parameters
        ----------
        samples : list of EvalSample
        k       : top-k for retrieval metrics
        verbose : print progress
        """
        results: List[SampleResult] = []

        for i, sample in enumerate(samples):
            if verbose:
                print(f"[{i+1}/{len(samples)}] Evaluating: {sample.question[:60]}...")

            result = self._eval_sample(sample, k=k)
            results.append(result)

            if verbose:
                print(
                    f"  → faithfulness={result.faithfulness_score:.3f} "
                    f"relevance={result.relevance_score:.3f}"
                )

        report = EvalReport(samples=results, config=self.eval_config)
        return report

    def _eval_sample(self, sample: EvalSample, k: int) -> SampleResult:
        try:
            rag_result: RAGResult = self.pipeline.query(sample.question)
        except Exception as e:
            logger.error("Pipeline query failed for '%s': %s", sample.question, e)
            return SampleResult(
                question=sample.question,
                answer="",
                error=str(e),
            )

        # Build context string from retrieved chunks
        context = "\n\n".join(c.text for c in rag_result.sources)

        # Faithfulness
        faith = self.faithfulness_eval.evaluate(rag_result.answer, context)

        # Answer relevance
        rel = self.relevance_eval.evaluate(sample.question, rag_result.answer)

        # Retrieval metrics (only if ground truth is provided)
        ret_metrics = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
        if sample.relevant_chunk_ids:
            retrieved_ids = [c.chunk_id for c in rag_result.retrieved_chunks]
            metrics = self.retrieval_eval.evaluate_all(
                retrieved_ids, sample.relevant_chunk_ids, k=k
            )
            ret_metrics = {
                "precision": metrics[f"precision@{k}"],
                "recall": metrics[f"recall@{k}"],
                "mrr": metrics["mrr"],
                "ndcg": metrics[f"ndcg@{k}"],
            }

        return SampleResult(
            question=sample.question,
            answer=rag_result.answer,
            faithfulness_score=faith["score"],
            relevance_score=rel["score"],
            retrieval_precision=ret_metrics["precision"],
            retrieval_recall=ret_metrics["recall"],
            retrieval_mrr=ret_metrics["mrr"],
            retrieval_ndcg=ret_metrics["ndcg"],
            faithfulness_details=faith,
            sources=[c.doc_title or c.doc_id for c in rag_result.sources],
        )
