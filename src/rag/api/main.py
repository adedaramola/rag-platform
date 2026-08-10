"""FastAPI application. Exposes POST /query, GET /health, and GET /metrics.

The pipeline — including cross-encoder model load and store connection — is
initialised once at startup via the lifespan context manager, not on first request.
"""

from __future__ import annotations

import contextlib
import statistics
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from rag.config import get_settings
from rag.exceptions import RAGError
from rag.pipeline import RAGPipeline, build_pipeline

logger = structlog.get_logger(__name__)

# Claude Sonnet pricing (per token, USD).
# Source: https://www.anthropic.com/pricing — update if pricing changes.
_INPUT_COST_PER_TOKEN = 3.00 / 1_000_000  # $3.00 / MTok
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000  # $15.00 / MTok

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming question."""

    question: str


class CitationOut(BaseModel):
    """One source cited in the answer."""

    index: int
    source: str
    page: int | None = None


class QueryResponse(BaseModel):
    """Answer with inline citations and retrieval metadata."""

    answer: str
    citations: list[CitationOut]
    context_chunks: int


class MetricsResponse(BaseModel):
    """Operational metrics snapshot."""

    total_queries: int
    p95_latency_ms: float | None  # None when fewer than 20 queries recorded
    avg_cost_usd: float | None  # None when no token data available
    citation_coverage: float | None  # fraction of queries with ≥1 citation; None when 0 queries


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------


class _Metrics:
    """Thread-safe in-memory metrics store.

    Tracks the last 1 000 query latencies for P95 calculation, plus running
    totals for cost and citation coverage. No external dependencies required.
    """

    _WINDOW = 1_000  # number of recent latencies to keep

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=self._WINDOW)
        self._total_queries = 0
        self._cited_queries = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def record(
        self,
        latency_ms: float,
        cited: bool,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            self._total_queries += 1
            if cited:
                self._cited_queries += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            total = self._total_queries
            p95 = (
                statistics.quantiles(sorted(self._latencies), n=100)[94]
                if len(self._latencies) >= 20
                else None
            )
            coverage = self._cited_queries / total if total > 0 else None
            if self._total_input_tokens + self._total_output_tokens > 0:
                total_cost = (
                    self._total_input_tokens * _INPUT_COST_PER_TOKEN
                    + self._total_output_tokens * _OUTPUT_COST_PER_TOKEN
                )
                avg_cost: float | None = total_cost / total if total > 0 else None
            else:
                avg_cost = None

        return MetricsResponse(
            total_queries=total,
            p95_latency_ms=round(p95, 1) if p95 is not None else None,
            avg_cost_usd=round(avg_cost, 6) if avg_cost is not None else None,
            citation_coverage=round(coverage, 4) if coverage is not None else None,
        )


# ---------------------------------------------------------------------------
# Application lifespan — builds the pipeline once at startup
# ---------------------------------------------------------------------------

_pipeline: RAGPipeline | None = None
_metrics = _Metrics()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Build the RAGPipeline on startup; release on shutdown."""
    global _pipeline
    settings = get_settings()
    logger.info("rag_platform_api_startup", store_backend=settings.store_backend)
    try:
        _pipeline = build_pipeline(settings)
        logger.info("rag_platform_api_ready")
    except Exception as exc:
        logger.error("rag_platform_api_startup_failed", error=str(exc))
        raise
    yield
    _pipeline = None
    logger.info("rag_platform_api_shutdown")


app = FastAPI(
    title="RAG Platform RAG API",
    description="Citation-grounded document Q&A powered by hybrid retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> dict[str, str]:
    """Liveness probe — used by the ALB health check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    """Run a question through the RAG pipeline and return a cited answer."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised — startup may still be in progress.",
        )
    log = logger.bind(question=body.question[:80])
    t0 = time.perf_counter()
    try:
        result = _pipeline.query(body.question)
        latency_ms = (time.perf_counter() - t0) * 1000
        _metrics.record(
            latency_ms=latency_ms,
            cited=len(result.citations) > 0,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        log.info(
            "query_ok",
            citations=len(result.citations),
            latency_ms=round(latency_ms, 1),
        )
        return QueryResponse(
            answer=result.answer,
            citations=[
                CitationOut(index=c.index, source=c.source, page=c.page) for c in result.citations
            ],
            context_chunks=len(result.raw_context),
        )
    except RAGError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        _metrics.record(latency_ms=latency_ms, cited=False, input_tokens=0, output_tokens=0)
        log.warning("query_rag_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        _metrics.record(latency_ms=latency_ms, cited=False, input_tokens=0, output_tokens=0)
        log.error("query_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Operational metrics: P95 latency, average cost per request, citation coverage.

    P95 latency is computed over the last 1 000 queries and returns null until
    at least 20 queries have been recorded. Cost and citation coverage return
    null until the first successful query completes.
    """
    return _metrics.snapshot()
