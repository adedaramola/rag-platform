"""Langfuse tracing integration.

Tracing is fully opt-in: when RAG_PLATFORM_LANGFUSE_PUBLIC_KEY and
RAG_PLATFORM_LANGFUSE_SECRET_KEY are absent, all calls are no-ops and the
system runs identically to an untraced deployment.

A tracing failure must never propagate to the user — all Langfuse calls
are wrapped defensively. The pipeline works correctly whether tracing is
enabled or not.

Usage:
    tracer = get_tracer(settings)
    with tracer.trace("query", input=question) as span:
        with span.span("retrieval") as s:
            chunks = retriever.retrieve(question)
            s.update(output={"chunks": len(chunks)})
        with span.span("generation") as s:
            answer = generator.generate(question, chunks)
            s.update(output={"citations": len(answer.citations)})
    tracer.flush()
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from rag.config import Settings

logger = structlog.get_logger(__name__)


class _NoOpSpan:
    """A span that does nothing. Used when tracing is disabled."""

    def update(self, **kwargs: Any) -> None:
        pass

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """Tracer that does nothing. Active when Langfuse keys are not configured."""

    @contextmanager
    def trace(self, name: str, **kwargs: Any) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()

    def flush(self) -> None:
        pass

    @property
    def enabled(self) -> bool:
        return False


class _LangfuseSpan:
    """Thin wrapper around a Langfuse span or trace object."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def update(self, **kwargs: Any) -> None:
        with contextlib.suppress(Exception):
            self._span.update(**kwargs)

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Generator[_LangfuseSpan, None, None]:
        child: Any = None
        try:
            child = self._span.span(name=name, **kwargs)
            wrapped: _LangfuseSpan = _LangfuseSpan(child)
        except Exception:
            wrapped = _LangfuseSpan(_NoOpSpan())
        try:
            yield wrapped
        finally:
            if child is not None:
                with contextlib.suppress(Exception):
                    child.end()

    def __enter__(self) -> _LangfuseSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        with contextlib.suppress(Exception):
            self._span.end()


class _LangfuseTracer:
    """Active tracer backed by a Langfuse client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @contextmanager
    def trace(self, name: str, **kwargs: Any) -> Generator[_LangfuseSpan, None, None]:
        trace: Any = None
        try:
            trace = self._client.trace(name=name, **kwargs)
            yield _LangfuseSpan(trace)
        except Exception as exc:
            logger.warning("langfuse_trace_failed", name=name, error=str(exc))
            yield _LangfuseSpan(_NoOpSpan())
        finally:
            if trace is not None:
                with contextlib.suppress(Exception):
                    trace.update(status_message="ok")

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._client.flush()

    @property
    def enabled(self) -> bool:
        return True


def get_tracer(settings: Settings) -> _LangfuseTracer | _NoOpTracer:
    """Return a Langfuse tracer if keys are configured, otherwise a no-op tracer.

    Never raises — falls back to no-op on any import or auth failure.
    """
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return _NoOpTracer()

    try:
        from langfuse import Langfuse  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "langfuse_not_installed",
            hint="pip install 'rag-platform[observability]'",
        )
        return _NoOpTracer()

    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
        logger.info("langfuse_tracing_enabled", host=settings.langfuse_host)
        return _LangfuseTracer(client)
    except Exception as exc:
        logger.warning("langfuse_init_failed", error=str(exc))
        return _NoOpTracer()
