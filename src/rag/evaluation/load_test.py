"""Async HTTP load test for end-to-end latency, concurrency, and cache benefit."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from rag.exceptions import EvaluationError

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RequestSample:
    """One observed API request."""

    status_code: int
    latency_ms: float
    error: str | None = None


async def run_load_test(
    endpoint: str,
    questions: list[str],
    request_count: int,
    concurrency: int,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Measure cold and warm API behavior using an exact-repeat query workload."""
    if not questions:
        raise EvaluationError("Load test requires at least one question")
    if request_count < 1 or concurrency < 1:
        raise EvaluationError("Load test request count and concurrency must be positive")

    headers = {"X-API-Key": api_key} if api_key else {}
    query_url = f"{endpoint.rstrip('/')}/query"
    metrics_url = f"{endpoint.rstrip('/')}/metrics"
    timeout = httpx.Timeout(120.0)

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        before_metrics = await _fetch_metrics(client, metrics_url)

        # One request per unique question records the cold path and primes semantic cache.
        cold_samples: list[RequestSample] = []
        for question in questions:
            cold_samples.append(await _request(client, query_url, question))

        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(index: int) -> RequestSample:
            async with semaphore:
                return await _request(client, query_url, questions[index % len(questions)])

        started = time.perf_counter()
        warm_samples = await asyncio.gather(*(_bounded(i) for i in range(request_count)))
        wall_seconds = time.perf_counter() - started
        after_metrics = await _fetch_metrics(client, metrics_url)

    cold = _summarize(cold_samples, wall_seconds=None)
    warm = _summarize(warm_samples, wall_seconds=wall_seconds)
    cold_p50 = float(cold["latency_ms"]["p50"])
    warm_p50 = float(warm["latency_ms"]["p50"])
    return {
        "schema_version": "1.0",
        "endpoint": endpoint,
        "unique_questions": len(questions),
        "request_count": request_count,
        "concurrency": concurrency,
        "cold": cold,
        "warm_repeated": warm,
        "cache_latency_speedup": round(cold_p50 / warm_p50, 3) if warm_p50 else None,
        "server_metrics_before": before_metrics,
        "server_metrics_after": after_metrics,
        "server_metrics_delta": _metrics_delta(before_metrics, after_metrics),
    }


def write_load_report(report: dict[str, Any], output_path: Path) -> None:
    """Write the load-test report as JSON."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Failed to write load-test report: {output_path}") from exc
    logger.info("load_test_report_written", path=str(output_path))


async def _request(client: httpx.AsyncClient, url: str, question: str) -> RequestSample:
    """Send one query and always return an observation instead of raising."""
    started = time.perf_counter()
    try:
        response = await client.post(url, json={"question": question})
        return RequestSample(
            status_code=response.status_code,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=None if response.is_success else response.text[:200],
        )
    except httpx.HTTPError as exc:
        return RequestSample(
            status_code=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(exc),
        )


async def _fetch_metrics(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    """Fetch optional server metrics without invalidating a load test."""
    try:
        response = await client.get(url)
        if response.is_success:
            payload = response.json()
            return dict(payload) if isinstance(payload, dict) else None
    except (httpx.HTTPError, json.JSONDecodeError):
        return None
    return None


def _summarize(samples: list[RequestSample], wall_seconds: float | None) -> dict[str, Any]:
    """Aggregate response status and end-to-end latency statistics."""
    latencies = [sample.latency_ms for sample in samples]
    successes = sum(200 <= sample.status_code < 300 for sample in samples)
    status_counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample.status_code)
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "requests": len(samples),
        "successes": successes,
        "errors": len(samples) - successes,
        "success_rate": round(successes / len(samples), 6) if samples else 0.0,
        "throughput_rps": (
            round(len(samples) / wall_seconds, 3) if wall_seconds and wall_seconds > 0 else None
        ),
        "status_counts": status_counts,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "p99": round(_percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
    }


def _metrics_delta(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, float] | None:
    """Calculate numeric server-counter changes when both snapshots exist."""
    if before is None or after is None:
        return None
    keys = ("total_queries", "cache_hit_count", "cache_miss_count")
    return {
        key: float(after.get(key, 0)) - float(before.get(key, 0))
        for key in keys
        if isinstance(after.get(key, 0), (int, float))
        and isinstance(before.get(key, 0), (int, float))
    }


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
