"""CLI orchestration for deterministic retrieval benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from rag.config import get_settings
from rag.evaluation.retrieval_benchmark import RetrievalBenchmarkRunner
from rag.pipeline import build_pipeline

logger = structlog.get_logger(__name__)


def main() -> None:
    """Run dense, BM25, hybrid, reranked, latency, and data-quality evaluation."""
    parser = argparse.ArgumentParser(description="Benchmark every RAG retrieval stage")
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="JSONL query/qrel path (defaults to Settings.eval_golden_path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/retrieval_benchmark.json"),
        help="JSON report destination",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    pipeline = build_pipeline(settings)
    runner = RetrievalBenchmarkRunner(pipeline.retriever, ks=tuple(args.k))
    report = runner.run(args.golden or settings.eval_golden_path, limit=args.limit)
    runner.write_report(report, args.output)
    logger.info(
        "benchmark_summary",
        output=str(args.output),
        queries=report["dataset"]["query_count"],
        corpus_chunks=report["corpus"]["chunks"],
    )
