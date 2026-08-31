"""FastAPI application for cited answers and Agent-only evidence retrieval.

The pipeline — including cross-encoder model load and store connection — is
initialised once at startup via the lifespan context manager, not on first request.

Security:
  - /health is open (required for ALB health checks).
  - /query and /metrics require X-API-Key when RAG_PLATFORM_API_KEY is configured.
    When RAG_PLATFORM_API_KEY is unset the check is skipped (safe for local dev).
  - /query is rate-limited to 10 requests/minute per API key (or client IP as
    fallback). Exceeding the limit returns HTTP 429.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import re
import statistics
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

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

    question: str = Field(min_length=1, max_length=2_000)


class SearchRequest(BaseModel):
    """Bounded retrieval-only request for an authenticated Agent."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    source_ids: list[
        Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._-]+$")]
    ] = Field(default_factory=list, max_length=20)
    max_chunks: int = Field(default=5, ge=1, le=8)


class SearchChunkOut(BaseModel):
    """One approved evidence excerpt with a request-local citation ID."""

    citation_id: str
    source_id: str
    page: int | None
    text: str


class SearchResponse(BaseModel):
    """Bounded evidence response; it deliberately contains no generated answer."""

    query_id: str
    chunks: list[SearchChunkOut]


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
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_hit_rate: float | None = None  # None when cache disabled or no lookups yet


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
# Security — API key authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(
    provided: Annotated[str | None, Security(_api_key_header)] = None,
) -> None:
    """Dependency that enforces X-API-Key when RAG_PLATFORM_API_KEY is configured.

    When api_key is unset in Settings the check is a no-op, preserving
    zero-friction local development. Set RAG_PLATFORM_API_KEY to activate.
    """
    settings = get_settings()
    if settings.api_key is None:
        return
    if provided is None or not hmac.compare_digest(provided, settings.api_key.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )


def _verify_agent_api_key(
    provided: Annotated[str | None, Security(_api_key_header)] = None,
) -> None:
    """Require the configured service credential for the Agent-only search route."""
    settings = get_settings()
    if settings.api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent retrieval is not configured.",
        )
    if provided is None or not hmac.compare_digest(provided, settings.api_key.get_secret_value()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def _rate_limit_key(request: Request) -> str:
    """Rate-limit by API key when present, falling back to client IP."""
    key = request.headers.get("X-API-Key")
    if key:
        return f"apikey:{hashlib.sha256(key.encode()).hexdigest()[:16]}"
    if request.client:
        return f"ip:{request.client.host}"
    return "ip:unknown"


limiter = Limiter(key_func=_rate_limit_key)


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
        logger.error("rag_platform_api_startup_failed", error_type=type(exc).__name__)
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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type,unused-ignore]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> dict[str, str]:
    """Liveness probe — used by the ALB health check."""
    return {"status": "ok"}


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(_verify_api_key)],
)
@limiter.limit(lambda: get_settings().api_rate_limit)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Run a question through the RAG pipeline and return a cited answer."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised — startup may still be in progress.",
        )
    log = logger.bind(query_length=len(body.question))
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
        log.warning("query_rag_error", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        _metrics.record(latency_ms=latency_ms, cited=False, input_tokens=0, output_tokens=0)
        log.error("query_unexpected_error", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@app.post(
    "/v1/search",
    response_model=SearchResponse,
    dependencies=[Depends(_verify_agent_api_key)],
)
@limiter.limit(lambda: get_settings().api_rate_limit)
def search(
    request: Request,
    body: SearchRequest,
    traceparent: Annotated[
        str | None,
        Header(pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"),
    ] = None,
    workflow_id: Annotated[
        str | None,
        Header(alias="X-Workflow-ID", pattern=r"^[0-9a-f-]{36}$"),
    ] = None,
) -> SearchResponse:
    """Return approved retrieval evidence without invoking answer generation."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised — startup may still be in progress.",
        )
    settings = get_settings()
    approved = set(settings.approved_source_ids)
    if not approved:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No approved knowledge sources are configured.",
        )
    requested = set(body.source_ids)
    if requested and not requested.issubset(approved):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="One or more requested sources are not approved.",
        )
    allowed = requested or approved
    started = time.perf_counter()
    query_id = f"rag-{uuid.uuid4()}"
    try:
        trace_id = traceparent.split("-")[1] if traceparent is not None else None
        retrieved = _pipeline.search(
            body.query,
            trace_id=trace_id,
            workflow_id=workflow_id,
        )
    except RAGError as exc:
        logger.warning(
            "search_rag_error",
            query_id=query_id,
            query_length=len(body.query),
            trace_id=trace_id,
            workflow_id=workflow_id,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Knowledge retrieval failed.",
        ) from exc
    except Exception as exc:
        logger.error(
            "search_unexpected_error",
            query_id=query_id,
            query_length=len(body.query),
            trace_id=trace_id,
            workflow_id=workflow_id,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc

    chunks: list[SearchChunkOut] = []
    for chunk in retrieved:
        source_id = _source_id(chunk.source)
        if source_id not in allowed:
            continue
        excerpt = (chunk.parent_text or chunk.text).strip()[: settings.search_max_excerpt_chars]
        if not excerpt:
            continue
        chunks.append(
            SearchChunkOut(
                citation_id=f"C{len(chunks) + 1}",
                source_id=source_id,
                page=chunk.page,
                text=excerpt,
            )
        )
        if len(chunks) >= body.max_chunks:
            break
    logger.info(
        "search_ok",
        query_id=query_id,
        query_length=len(body.query),
        trace_id=trace_id,
        workflow_id=workflow_id,
        result_count=len(chunks),
        latency_ms=round((time.perf_counter() - started) * 1_000, 1),
    )
    return SearchResponse(query_id=query_id, chunks=chunks)


def _source_id(source: str) -> str:
    """Derive a stable public identifier from an ingested file path or URL."""
    parsed_path = urlparse(source).path if "://" in source else source
    candidate = Path(parsed_path).stem or Path(parsed_path).name or source
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-.").lower()
    return normalized[:200] or "unknown-source"


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    dependencies=[Depends(_verify_api_key)],
)
def metrics() -> MetricsResponse:
    """Operational metrics: P95 latency, average cost per request, citation coverage.

    P95 latency is computed over the last 1 000 queries and returns null until
    at least 20 queries have been recorded. Cost and citation coverage return
    null until the first successful query completes. Cache stats are included
    when a semantic cache backend is configured.
    """
    snap = _metrics.snapshot()
    if _pipeline is not None and _pipeline.cache is not None:
        hits = _pipeline.cache.hit_count
        misses = _pipeline.cache.miss_count
        total = hits + misses
        return MetricsResponse(
            total_queries=snap.total_queries,
            p95_latency_ms=snap.p95_latency_ms,
            avg_cost_usd=snap.avg_cost_usd,
            citation_coverage=snap.citation_coverage,
            cache_hit_count=hits,
            cache_miss_count=misses,
            cache_hit_rate=round(hits / total, 4) if total > 0 else None,
        )
    return snap
