"""Unit tests for the domain exception hierarchy."""

from __future__ import annotations

import pytest

from rag.exceptions import (
    ChunkingError,
    CitationGroundingError,
    DocumentLoadError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IngestionError,
    RAGError,
    RetrievalError,
    SchemaError,
    StoreConnectionError,
    StoreError,
    ThresholdViolationError,
    VectorIndexError,
)

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


def test_rag_error_message_and_context() -> None:
    e = RAGError("something broke", context={"key": "val"})
    assert str(e) == "something broke"
    assert e.context == {"key": "val"}


def test_rag_error_default_context_is_empty() -> None:
    e = RAGError("oops")
    assert e.context == {}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_document_load_error_default_message() -> None:
    e = DocumentLoadError(source="file.pdf")
    assert "file.pdf" in str(e)
    assert e.source == "file.pdf"
    assert isinstance(e, IngestionError)


def test_document_load_error_custom_message() -> None:
    e = DocumentLoadError(source="x.pdf", message="custom msg")
    assert str(e) == "custom msg"


def test_chunking_error() -> None:
    e = ChunkingError(document_id="doc-123")
    assert "doc-123" in str(e)
    assert e.document_id == "doc-123"
    assert isinstance(e, IngestionError)


def test_chunking_error_custom_message() -> None:
    e = ChunkingError(document_id="doc-456", message="custom chunk error")
    assert str(e) == "custom chunk error"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieval_error() -> None:
    e = RetrievalError(query="what is RAG?", step="dense_search")
    assert "dense_search" in str(e)
    assert e.query_length == len("what is RAG?")
    assert "what is RAG?" not in str(e)
    assert e.context == {"query_length": len("what is RAG?"), "step": "dense_search"}
    assert e.step == "dense_search"


def test_embedding_error() -> None:
    e = EmbeddingError(model="text-embedding-3-small")
    assert "text-embedding-3-small" in str(e)
    assert e.model == "text-embedding-3-small"
    assert isinstance(e, RetrievalError)


def test_vector_index_error() -> None:
    e = VectorIndexError(query="q", step="bm25")
    assert isinstance(e, RetrievalError)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generation_error() -> None:
    e = GenerationError("Claude API timed out")
    assert "Claude API timed out" in str(e)
    assert isinstance(e, RAGError)


def test_citation_grounding_error() -> None:
    e = CitationGroundingError(answer="Some text with no citations.", reason="no [src N] found")
    assert "no [src N] found" in str(e)
    assert e.answer == "Some text with no citations."
    assert e.reason == "no [src N] found"
    assert isinstance(e, GenerationError)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_error() -> None:
    e = StoreError("disk full")
    assert isinstance(e, RAGError)


def test_store_connection_error() -> None:
    e = StoreConnectionError(host="localhost", port=8080)
    assert "localhost" in str(e)
    assert "8080" in str(e)
    assert e.host == "localhost"
    assert e.port == 8080
    assert isinstance(e, StoreError)


def test_store_connection_error_custom_message() -> None:
    e = StoreConnectionError(host="h", port=9999, message="custom conn error")
    assert str(e) == "custom conn error"


def test_schema_error() -> None:
    e = SchemaError(expected="DocumentChunk", actual="OldCollection")
    assert "DocumentChunk" in str(e)
    assert "OldCollection" in str(e)
    assert e.expected == "DocumentChunk"
    assert e.actual == "OldCollection"
    assert isinstance(e, StoreError)


def test_schema_error_custom_message() -> None:
    e = SchemaError(expected="A", actual="B", message="mismatch!")
    assert str(e) == "mismatch!"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluation_error() -> None:
    e = EvaluationError("eval runner crashed")
    assert isinstance(e, RAGError)


def test_threshold_violation_error_str() -> None:
    e = ThresholdViolationError(metric="faithfulness", actual=0.72, required=0.85)
    s = str(e)
    assert "faithfulness" in s
    assert "0.720" in s
    assert "0.850" in s


def test_threshold_violation_error_attributes() -> None:
    e = ThresholdViolationError(metric="answer_relevancy", actual=0.6, required=0.8)
    assert e.metric == "answer_relevancy"
    assert pytest.approx(e.actual) == 0.6
    assert pytest.approx(e.required) == 0.8
    assert isinstance(e, EvaluationError)
