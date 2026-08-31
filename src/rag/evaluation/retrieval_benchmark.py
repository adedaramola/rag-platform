"""Deterministic retrieval benchmark with ranking, latency, scale, and data-quality metrics.

The benchmark deliberately keeps classical retrieval metrics separate from LLM-as-judge
metrics. Relevance is evaluated at the finest label granularity present in each golden row:
chunk ID, source+page, or source. No-answer rows are reported separately because Recall@K and
MRR are undefined when a query has no relevant documents.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog

from rag.exceptions import EvaluationError
from rag.interfaces.retriever import CorpusStats, RetrievedChunk, RetrieverProtocol

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GoldenQuery:
    """One query and its graded or source-level relevance labels."""

    question: str
    ground_truth: str
    expected_sources: tuple[str, ...]
    expected_pages: tuple[int, ...]
    relevant_chunk_ids: tuple[str, ...]

    @property
    def has_relevance_labels(self) -> bool:
        """Return whether this is a positive query with at least one qrel."""
        return bool(self.expected_sources or self.expected_pages or self.relevant_chunk_ids)

    @property
    def granularity(self) -> str:
        """Return the finest available relevance-judgment granularity."""
        if self.relevant_chunk_ids:
            return "chunk"
        if self.expected_pages:
            return "page"
        if self.expected_sources:
            return "source"
        return "none"


class RetrievalBenchmarkRunner:
    """Evaluate all retriever stages from one retrieval execution per query."""

    def __init__(self, retriever: RetrieverProtocol, ks: tuple[int, ...] = (1, 3, 5)) -> None:
        if not ks or any(k < 1 for k in ks):
            raise EvaluationError("Benchmark K values must all be positive integers")
        self._retriever = retriever
        self._ks = tuple(sorted(set(ks)))

    def run(self, golden_path: Path, limit: int | None = None) -> dict[str, Any]:
        """Run the benchmark and return a JSON-serializable report."""
        queries = _load_golden(golden_path)
        if limit is not None:
            if limit < 1:
                raise EvaluationError("Benchmark query limit must be positive")
            queries = queries[:limit]
        if not queries:
            raise EvaluationError(f"Golden dataset is empty: {golden_path}")

        stages: dict[str, list[tuple[GoldenQuery, list[RetrievedChunk]]]] = {
            "dense": [],
            "bm25": [],
            "hybrid_pre_rerank": [],
            "hybrid_post_rerank": [],
        }
        timing_samples: dict[str, list[float]] = {}

        logger.info("retrieval_benchmark_start", queries=len(queries), ks=self._ks)
        for index, query in enumerate(queries, start=1):
            diagnostics = self._retriever.retrieve_with_diagnostics(query.question)
            stages["dense"].append((query, diagnostics.dense))
            stages["bm25"].append((query, diagnostics.bm25))
            stages["hybrid_pre_rerank"].append((query, diagnostics.hybrid_pre_rerank))
            stages["hybrid_post_rerank"].append((query, diagnostics.hybrid_post_rerank))
            for name, value in diagnostics.timings_ms.items():
                timing_samples.setdefault(name, []).append(value)
            logger.info(
                "retrieval_benchmark_query_complete",
                query_index=index,
                query_count=len(queries),
            )

        corpus = self._retriever.corpus_stats()
        report: dict[str, Any] = {
            "schema_version": "1.0",
            "metric_scope": {
                "unit": "retrieved chunk",
                "relevance_granularity": _granularity_counts(queries),
                "notes": (
                    "Positive-query ranking metrics exclude no-answer rows. "
                    "No-answer retrieval is reported separately."
                ),
            },
            "dataset": {
                "query_count": len(queries),
                "positive_query_count": sum(q.has_relevance_labels for q in queries),
                "no_answer_query_count": sum(not q.has_relevance_labels for q in queries),
            },
            "corpus": asdict(corpus),
            "data_quality": _data_quality(queries, corpus),
            "methods": {
                method: _evaluate_rankings(rows, self._ks) for method, rows in stages.items()
            },
            "latency_ms": {
                stage: _latency_summary(samples) for stage, samples in timing_samples.items()
            },
        }
        logger.info("retrieval_benchmark_complete", queries=len(queries))
        return report

    @staticmethod
    def write_report(report: dict[str, Any], output_path: Path) -> None:
        """Persist a benchmark report as stable, human-readable JSON."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            raise EvaluationError(f"Failed to write benchmark report: {output_path}") from exc
        logger.info("retrieval_benchmark_report_written", path=str(output_path))


def _evaluate_rankings(
    rows: list[tuple[GoldenQuery, list[RetrievedChunk]]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    """Aggregate binary-ranking metrics for one retrieval method."""
    positives = [(query, hits) for query, hits in rows if query.has_relevance_labels]
    negatives = [(query, hits) for query, hits in rows if not query.has_relevance_labels]
    metrics: dict[str, Any] = {
        "evaluated_positive_queries": len(positives),
        "evaluated_no_answer_queries": len(negatives),
        "mrr": _mean([_reciprocal_rank(query, hits) for query, hits in positives]),
        "no_answer_nonempty_rate": _mean([float(bool(hits)) for _query, hits in negatives]),
    }
    for k in ks:
        precisions = [_precision_at_k(query, hits, k) for query, hits in positives]
        recalls = [_recall_at_k(query, hits, k) for query, hits in positives]
        metrics[f"precision_at_{k}"] = _mean(precisions)
        metrics[f"recall_at_{k}"] = _mean(recalls)
        metrics[f"f1_at_{k}"] = _harmonic(metrics[f"precision_at_{k}"], metrics[f"recall_at_{k}"])
        metrics[f"ndcg_at_{k}"] = _mean([_ndcg_at_k(query, hits, k) for query, hits in positives])
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def _precision_at_k(query: GoldenQuery, hits: list[RetrievedChunk], k: int) -> float:
    """Return relevant retrieved chunks divided by K."""
    return sum(_is_relevant(query, hit) for hit in hits[:k]) / k


def _recall_at_k(query: GoldenQuery, hits: list[RetrievedChunk], k: int) -> float:
    """Return the fraction of unique qrel identities retrieved by K."""
    expected = _expected_identities(query)
    observed = {_hit_identity(query, hit) for hit in hits[:k] if _is_relevant(query, hit)}
    return len(observed & expected) / len(expected) if expected else 0.0


def _reciprocal_rank(query: GoldenQuery, hits: list[RetrievedChunk]) -> float:
    """Return reciprocal rank of the first relevant chunk."""
    for rank, hit in enumerate(hits, start=1):
        if _is_relevant(query, hit):
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(query: GoldenQuery, hits: list[RetrievedChunk], k: int) -> float:
    """Return binary NDCG@K, deduplicating repeated source/page judgments."""
    expected = _expected_identities(query)
    seen: set[str] = set()
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        identity = _hit_identity(query, hit)
        if identity in expected and identity not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(identity)
    ideal_count = min(k, len(expected))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def _is_relevant(query: GoldenQuery, hit: RetrievedChunk) -> bool:
    """Apply the query's finest available qrel to one hit."""
    return _hit_identity(query, hit) in _expected_identities(query)


def _expected_identities(query: GoldenQuery) -> set[str]:
    """Normalize qrels to stable identity strings."""
    if query.relevant_chunk_ids:
        return {f"chunk:{chunk_id}" for chunk_id in query.relevant_chunk_ids}
    sources = {_source_name(source) for source in query.expected_sources}
    if query.expected_pages:
        return {f"page:{source}:{page}" for source in sources for page in query.expected_pages}
    return {f"source:{source}" for source in sources}


def _hit_identity(query: GoldenQuery, hit: RetrievedChunk) -> str:
    """Normalize a retrieved chunk at the query's qrel granularity."""
    if query.relevant_chunk_ids:
        return f"chunk:{hit.id}"
    source = _source_name(hit.source)
    if query.expected_pages:
        return f"page:{source}:{hit.page}"
    return f"source:{source}"


def _data_quality(queries: list[GoldenQuery], corpus: CorpusStats) -> dict[str, Any]:
    """Find measurable corpus/golden-set defects that can invalidate scores."""
    indexed_sources = set(corpus.source_names)
    expected_sources = {
        _source_name(source) for query in queries for source in query.expected_sources
    }
    missing_sources = sorted(expected_sources - indexed_sources)
    questions_with_missing_sources = sum(
        bool({_source_name(source) for source in query.expected_sources} - indexed_sources)
        for query in queries
    )
    normalized_questions = [query.question.strip().casefold() for query in queries]
    return {
        "expected_sources": len(expected_sources),
        "indexed_expected_sources": len(expected_sources & indexed_sources),
        "missing_expected_sources": missing_sources,
        "questions_with_missing_expected_sources": questions_with_missing_sources,
        "duplicate_questions": len(normalized_questions) - len(set(normalized_questions)),
        "empty_ground_truths": sum(not query.ground_truth.strip() for query in queries),
        "source_only_positive_labels": sum(
            query.granularity == "source" for query in queries if query.has_relevance_labels
        ),
        "passage_labeled_positive_queries": sum(
            query.granularity in {"page", "chunk"}
            for query in queries
            if query.has_relevance_labels
        ),
    }


def _load_golden(path: Path) -> list[GoldenQuery]:
    """Load and validate the supported JSONL relevance fields."""
    queries: list[GoldenQuery] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                question = str(row.get("question", "")).strip()
                if not question:
                    raise EvaluationError(f"Missing question at {path}:{line_number}")
                queries.append(
                    GoldenQuery(
                        question=question,
                        ground_truth=str(row.get("ground_truth", "")),
                        expected_sources=tuple(str(v) for v in row.get("expected_sources", [])),
                        expected_pages=tuple(int(v) for v in row.get("expected_pages", [])),
                        relevant_chunk_ids=tuple(str(v) for v in row.get("relevant_chunk_ids", [])),
                    )
                )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationError(f"Failed to load retrieval golden dataset: {path}") from exc
    return queries


def _latency_summary(samples: list[float]) -> dict[str, float | int]:
    """Return stable descriptive latency statistics."""
    return {
        "count": len(samples),
        "mean": round(statistics.fmean(samples), 3),
        "p50": round(_percentile(samples, 0.50), 3),
        "p95": round(_percentile(samples, 0.95), 3),
        "max": round(max(samples), 3),
    }


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without external dependencies."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _granularity_counts(queries: list[GoldenQuery]) -> dict[str, int]:
    """Count qrels by available granularity."""
    return {
        granularity: sum(query.granularity == granularity for query in queries)
        for granularity in ("chunk", "page", "source", "none")
    }


def _source_name(source: str) -> str:
    """Normalize local paths and URLs to a comparable final path component."""
    return Path(source).name.casefold()


def _mean(values: list[float]) -> float:
    """Return a safe arithmetic mean."""
    return statistics.fmean(values) if values else 0.0


def _harmonic(left: float, right: float) -> float:
    """Return the harmonic mean used for F1."""
    return 2 * left * right / (left + right) if left + right else 0.0
