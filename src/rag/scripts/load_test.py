"""CLI orchestration for API load, latency, and semantic-cache measurement."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import structlog

from rag.config import get_settings
from rag.evaluation.load_test import run_load_test, write_load_report
from rag.exceptions import EvaluationError

logger = structlog.get_logger(__name__)


def main() -> None:
    """Run a cold/warm concurrent workload against a deployed RAG API."""
    parser = argparse.ArgumentParser(description="Load test a deployed RAG Platform API")
    parser.add_argument("--url", required=True, help="API base URL, without /query")
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--query-limit", type=int, default=5)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/load_test.json"))
    args = parser.parse_args()

    settings = get_settings()
    golden_path = args.golden or settings.eval_golden_path
    try:
        rows = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines()]
        questions = [str(row["question"]) for row in rows[: args.query_limit]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvaluationError(f"Failed to load load-test queries: {golden_path}") from exc

    api_key = settings.api_key.get_secret_value() if settings.api_key is not None else None
    report = asyncio.run(
        run_load_test(
            endpoint=args.url,
            questions=questions,
            request_count=args.requests,
            concurrency=args.concurrency,
            api_key=api_key,
        )
    )
    write_load_report(report, args.output)
    logger.info(
        "load_test_complete",
        output=str(args.output),
        success_rate=report["warm_repeated"]["success_rate"],
        throughput_rps=report["warm_repeated"]["throughput_rps"],
    )
