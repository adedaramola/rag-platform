"""Unit tests for the Langfuse tracing integration.

All tests must run without Langfuse installed and without any API keys set.
The tracing module must behave identically whether tracing is enabled or not —
failures are suppressed and no-ops fall through transparently.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from rag.config import Settings
from rag.tracing import _LangfuseSpan, _LangfuseTracer, _NoOpSpan, _NoOpTracer, get_tracer

# ---------------------------------------------------------------------------
# _NoOpSpan
# ---------------------------------------------------------------------------


def test_noop_span_update_is_silent() -> None:
    """update() on a _NoOpSpan must never raise."""
    span = _NoOpSpan()
    span.update(output={"chunks": 3}, metadata={"model": "x"})  # should not raise


def test_noop_span_context_manager() -> None:
    """_NoOpSpan must work as a context manager and return itself."""
    span = _NoOpSpan()
    with span as s:
        assert s is span


def test_noop_span_child_span_yields_noop() -> None:
    """span.span() must yield a _NoOpSpan child, callable recursively."""
    parent = _NoOpSpan()
    with parent.span("child") as child:
        assert isinstance(child, _NoOpSpan)
        child.update(output={"x": 1})
        with child.span("grandchild") as gc:
            assert isinstance(gc, _NoOpSpan)


# ---------------------------------------------------------------------------
# _NoOpTracer
# ---------------------------------------------------------------------------


def test_noop_tracer_is_disabled() -> None:
    tracer = _NoOpTracer()
    assert tracer.enabled is False


def test_noop_tracer_flush_is_silent() -> None:
    _NoOpTracer().flush()


def test_noop_tracer_trace_yields_noop_span() -> None:
    tracer = _NoOpTracer()
    with tracer.trace("query", input="hello") as span:
        assert isinstance(span, _NoOpSpan)
        with span.span("retrieval") as s:
            s.update(output={"chunks": 2})
        with span.span("generation") as s:
            s.update(output={"citations": 1})
    tracer.flush()


# ---------------------------------------------------------------------------
# get_tracer — key-absent path
# ---------------------------------------------------------------------------


def _settings_no_keys() -> Settings:
    return Settings(anthropic_api_key=SecretStr("test"))


def test_get_tracer_returns_noop_when_no_keys() -> None:
    """get_tracer must return _NoOpTracer when Langfuse keys are absent."""
    tracer = get_tracer(_settings_no_keys())
    assert isinstance(tracer, _NoOpTracer)
    assert tracer.enabled is False


def test_get_tracer_returns_noop_when_only_public_key_set() -> None:
    s = Settings(
        anthropic_api_key=SecretStr("test"),
        langfuse_public_key=SecretStr("pk"),
        langfuse_secret_key=None,
    )
    assert isinstance(get_tracer(s), _NoOpTracer)


def test_get_tracer_returns_noop_when_only_secret_key_set() -> None:
    s = Settings(
        anthropic_api_key=SecretStr("test"),
        langfuse_public_key=None,
        langfuse_secret_key=SecretStr("sk"),
    )
    assert isinstance(get_tracer(s), _NoOpTracer)


# ---------------------------------------------------------------------------
# get_tracer — langfuse not installed path
# ---------------------------------------------------------------------------


def test_get_tracer_falls_back_to_noop_when_langfuse_missing() -> None:
    """If langfuse is not installed, get_tracer must return _NoOpTracer (no ImportError)."""
    s = Settings(
        anthropic_api_key=SecretStr("test"),
        langfuse_public_key=SecretStr("pk"),
        langfuse_secret_key=SecretStr("sk"),
    )
    with patch.dict("sys.modules", {"langfuse": None}):
        tracer = get_tracer(s)
    assert isinstance(tracer, _NoOpTracer)


# ---------------------------------------------------------------------------
# get_tracer — Langfuse init failure path
# ---------------------------------------------------------------------------


def test_get_tracer_falls_back_to_noop_on_init_failure() -> None:
    """If Langfuse raises during __init__, get_tracer must return _NoOpTracer."""
    s = Settings(
        anthropic_api_key=SecretStr("test"),
        langfuse_public_key=SecretStr("pk"),
        langfuse_secret_key=SecretStr("sk"),
    )
    mock_langfuse_module = MagicMock()
    mock_langfuse_module.Langfuse.side_effect = RuntimeError("connection refused")

    with patch.dict("sys.modules", {"langfuse": mock_langfuse_module}):
        tracer = get_tracer(s)
    assert isinstance(tracer, _NoOpTracer)


# ---------------------------------------------------------------------------
# get_tracer — happy path (Langfuse available and init succeeds)
# ---------------------------------------------------------------------------


def test_get_tracer_returns_langfuse_tracer_when_keys_set() -> None:
    """When both keys are set and Langfuse init succeeds, return _LangfuseTracer."""
    s = Settings(
        anthropic_api_key=SecretStr("test"),
        langfuse_public_key=SecretStr("pk-test"),
        langfuse_secret_key=SecretStr("sk-test"),
    )
    mock_client = MagicMock()
    mock_langfuse_module = MagicMock()
    mock_langfuse_module.Langfuse.return_value = mock_client

    with patch.dict("sys.modules", {"langfuse": mock_langfuse_module}):
        tracer = get_tracer(s)

    assert isinstance(tracer, _LangfuseTracer)
    assert tracer.enabled is True


# ---------------------------------------------------------------------------
# _LangfuseTracer / _LangfuseSpan — defensive failure handling
# ---------------------------------------------------------------------------


def test_langfuse_tracer_suppresses_trace_exception() -> None:
    """If the Langfuse client raises on trace(), the context manager yields a no-op span."""
    mock_client = MagicMock()
    mock_client.trace.side_effect = RuntimeError("network error")
    tracer = _LangfuseTracer(mock_client)

    with tracer.trace("query", input="test") as span:
        # Should receive a no-op span, not propagate the exception
        assert isinstance(span._span, _NoOpSpan)
        span.update(output={"x": 1})  # must not raise


def test_langfuse_span_suppresses_child_span_exception() -> None:
    """If creating a child span raises, it falls back to a no-op span."""
    mock_inner = MagicMock()
    mock_inner.span.side_effect = RuntimeError("langfuse error")
    parent = _LangfuseSpan(mock_inner)

    with parent.span("child") as child:
        assert isinstance(child._span, _NoOpSpan)


def test_langfuse_span_end_is_called_even_on_exception() -> None:
    """span.end() must be called in the finally block even if the body raises."""
    mock_inner = MagicMock()
    mock_child = MagicMock()
    mock_inner.span.return_value = mock_child
    parent = _LangfuseSpan(mock_inner)

    with pytest.raises(ValueError), parent.span("step"):  # noqa: PT011
        raise ValueError("oops")

    mock_child.end.assert_called_once()


def test_langfuse_tracer_flush_suppresses_exception() -> None:
    """flush() must never raise even if the client errors."""
    mock_client = MagicMock()
    mock_client.flush.side_effect = RuntimeError("flush failed")
    tracer = _LangfuseTracer(mock_client)
    tracer.flush()  # must not raise


# ---------------------------------------------------------------------------
# Pipeline integration — tracer wired through RAGPipeline
# ---------------------------------------------------------------------------


def test_pipeline_uses_noop_tracer_by_default() -> None:
    """RAGPipeline.tracer should default to a _NoOpTracer instance."""
    from rag.pipeline import RAGPipeline

    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator, embedder=MagicMock())
    assert isinstance(pipeline.tracer, _NoOpTracer)


def test_pipeline_query_calls_tracer_trace() -> None:
    """query() must call tracer.trace() wrapping the retrieval and generation steps."""
    from rag.interfaces.generator import CitationRef, CitedAnswer
    from rag.pipeline import RAGPipeline

    spy_tracer = MagicMock()
    spy_span = MagicMock()
    spy_child = MagicMock()

    # trace() is a context manager returning spy_span
    spy_tracer.trace.return_value.__enter__ = MagicMock(return_value=spy_span)
    spy_tracer.trace.return_value.__exit__ = MagicMock(return_value=False)

    # span.span() is a context manager returning spy_child
    spy_span.span.return_value.__enter__ = MagicMock(return_value=spy_child)
    spy_span.span.return_value.__exit__ = MagicMock(return_value=False)
    spy_child.update = MagicMock()

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_generator = MagicMock()
    mock_generator.generate.return_value = CitedAnswer(
        answer="Test answer [src 1]",
        citations=[CitationRef(index=1, source="doc.pdf", page=1)],
        raw_context=[],
    )

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        embedder=MagicMock(),
        tracer=spy_tracer,
    )
    pipeline.query("test question")

    spy_tracer.trace.assert_called_once()
    assert spy_tracer.trace.call_args.kwargs == {"metadata": {"query_length": 13}}
    spy_tracer.flush.assert_called_once()
