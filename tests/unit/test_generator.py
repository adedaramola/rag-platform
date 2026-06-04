"""Unit tests for CitationGroundedGenerator generation behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.exceptions import CitationGroundingError, GenerationError
from rag.generation.generator import CitationGroundedGenerator
from rag.interfaces.generator import CitedAnswer
from rag.interfaces.retriever import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(id="1", text="Parent chunk one.", source="doc1.pdf", page=1, score=0.9),
        RetrievedChunk(
            id="2",
            text="Web chunk two.",
            source="https://example.com",
            page=None,
            score=0.8,
        ),
    ]


def _response(answer: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(text=answer)],
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
    )


def test_build_context_block_formats_pdf_and_web_chunks() -> None:
    """Context block should include numbered sources and omit page labels for web chunks."""
    block = CitationGroundedGenerator._build_context_block(_chunks())
    assert "[src 1] (doc1.pdf p.1)" in block
    assert "Parent chunk one." in block
    assert "[src 2] (https://example.com)" in block
    assert "p.None" not in block


def test_generate_returns_cited_answer() -> None:
    """Successful generation should return a fully populated CitedAnswer."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _response("Draft answer [src 1]."),
        _response("Grounded answer [src 1]."),
    ]

    with patch("anthropic.Anthropic", return_value=mock_client):
        generator = CitationGroundedGenerator(
            model="claude-test", api_key="test-key", max_tokens=400
        )
        result = generator.generate("What happened?", _chunks())

    assert isinstance(result, CitedAnswer)
    assert result.answer == "Grounded answer [src 1]."
    assert len(result.citations) == 1
    assert result.citations[0].source == "doc1.pdf"
    assert result.input_tokens == 24
    assert result.output_tokens == 14
    assert mock_client.messages.create.call_count == 2
    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["max_tokens"] == 400
    assert kwargs["temperature"] == 0


def test_generate_falls_back_to_cited_draft_when_repair_loses_citations() -> None:
    """A cited draft should be returned if the repair pass drops citations."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _response("Draft answer [src 1]."),
        _response("Repair accidentally removed citations."),
    ]

    with patch("anthropic.Anthropic", return_value=mock_client):
        generator = CitationGroundedGenerator(model="claude-test", api_key="test-key")
        result = generator.generate("What happened?", _chunks())

    assert result.answer == "Draft answer [src 1]."
    assert len(result.citations) == 1


def test_generate_raises_when_draft_and_repair_lack_citations() -> None:
    """An uncited draft and uncited repair should fail closed."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _response("Draft answer without grounding."),
        _response("Repair answer still has no grounding."),
    ]

    with patch("anthropic.Anthropic", return_value=mock_client):
        generator = CitationGroundedGenerator(model="claude-test", api_key="test-key")

    with pytest.raises(CitationGroundingError, match="without valid \\[src N\\] citations"):
        generator.generate("Unsupported?", _chunks())


def test_generate_without_chunks_raises_generation_error() -> None:
    """generate() should reject calls with no retrieval context."""
    mock_client = MagicMock()

    with patch("anthropic.Anthropic", return_value=mock_client):
        generator = CitationGroundedGenerator(model="claude-test", api_key="test-key")

    with pytest.raises(GenerationError, match="no context chunks"):
        generator.generate("Empty?", [])


def test_generate_wraps_malformed_response_as_generation_error() -> None:
    """Responses without a text block should raise GenerationError."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        SimpleNamespace(
            content=[object()],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )
    ]

    with patch("anthropic.Anthropic", return_value=mock_client):
        generator = CitationGroundedGenerator(model="claude-test", api_key="test-key")

    with pytest.raises(GenerationError, match="Claude API call failed"):
        generator.generate("Malformed?", _chunks())


def test_generate_wraps_client_error_as_generation_error() -> None:
    """API client failures should be surfaced as GenerationError."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("upstream timeout")

    with patch("anthropic.Anthropic", return_value=mock_client):
        generator = CitationGroundedGenerator(model="claude-test", api_key="test-key")

    with pytest.raises(GenerationError, match="Claude API call failed"):
        generator.generate("Timeout?", _chunks())
