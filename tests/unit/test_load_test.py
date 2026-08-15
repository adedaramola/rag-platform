"""Unit tests for HTTP load-test aggregation and cache metric deltas."""

from __future__ import annotations

import pytest

from rag.evaluation.load_test import RequestSample, _metrics_delta, _summarize


def test_summarize_reports_status_latency_and_throughput() -> None:
    """Successful and failed samples produce stable operational statistics."""
    samples = [
        RequestSample(status_code=200, latency_ms=10.0),
        RequestSample(status_code=200, latency_ms=20.0),
        RequestSample(status_code=500, latency_ms=30.0, error="boom"),
    ]

    result = _summarize(samples, wall_seconds=1.5)

    assert result["successes"] == 2
    assert result["errors"] == 1
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["throughput_rps"] == 2.0
    assert result["status_counts"] == {"200": 2, "500": 1}
    assert result["latency_ms"]["p50"] == 20.0
    assert result["latency_ms"]["p95"] == 29.0


def test_metrics_delta_tracks_query_and_cache_counters() -> None:
    """Counter snapshots are converted to numeric deltas."""
    before = {"total_queries": 2, "cache_hit_count": 1, "cache_miss_count": 1}
    after = {"total_queries": 12, "cache_hit_count": 8, "cache_miss_count": 4}

    assert _metrics_delta(before, after) == {
        "total_queries": 10.0,
        "cache_hit_count": 7.0,
        "cache_miss_count": 3.0,
    }


def test_metrics_delta_requires_both_snapshots() -> None:
    """Unavailable server metrics remain explicitly unknown."""
    assert _metrics_delta(None, {}) is None
    assert _metrics_delta({}, None) is None
