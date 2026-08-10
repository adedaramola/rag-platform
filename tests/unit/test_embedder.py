"""Unit tests for embedder factory and implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from rag.config import Settings
from rag.exceptions import EmbeddingError
from rag.ingestion.embedder import BGEEmbedder, OpenAIEmbedder, get_embedder
from rag.interfaces.embedder import EmbedderProtocol


def _settings(**kwargs: object) -> Settings:
    return Settings(  # type: ignore[call-arg]
        anthropic_api_key=SecretStr("test"),
        openai_api_key=SecretStr("test-openai"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_embedder_openai_returns_protocol() -> None:
    s = _settings(embed_backend="openai")
    embedder = get_embedder(s)
    assert isinstance(embedder, EmbedderProtocol)
    assert isinstance(embedder, OpenAIEmbedder)


def test_get_embedder_local_returns_bge() -> None:
    s = _settings(embed_backend="local")
    embedder = get_embedder(s)
    assert isinstance(embedder, BGEEmbedder)


def test_get_embedder_openai_missing_key_raises() -> None:
    s = _settings(embed_backend="openai")
    object.__setattr__(s, "openai_api_key", None)
    with pytest.raises(ValueError, match="RAG_PLATFORM_OPENAI_API_KEY"):
        get_embedder(s)


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


def test_openai_embedder_embed_batch() -> None:
    """embed_batch calls the OpenAI API once per batch and returns vectors."""
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_client.embeddings.create.return_value = MagicMock(data=[mock_embedding, mock_embedding])

    # OpenAI is imported inside __init__ — patch the source module
    with patch("openai.OpenAI", return_value=mock_client):
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        result = embedder.embed_batch(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    mock_client.embeddings.create.assert_called_once()


def test_openai_embedder_embed_single() -> None:
    """embed() delegates to embed_batch and returns the first vector."""
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.5, 0.6]
    mock_client.embeddings.create.return_value = MagicMock(data=[mock_embedding])

    with patch("openai.OpenAI", return_value=mock_client):
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        result = embedder.embed("single text")

    assert result == [0.5, 0.6]


def test_openai_embedder_api_error_raises_embedding_error() -> None:
    """API failure is wrapped as EmbeddingError."""
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = RuntimeError("api down")

    with patch("openai.OpenAI", return_value=mock_client):
        embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="test-key")
        with pytest.raises(EmbeddingError):
            embedder.embed_batch(["text"])


# ---------------------------------------------------------------------------
# BGEEmbedder
# ---------------------------------------------------------------------------


def test_bge_embedder_missing_package_raises_import_error() -> None:
    """BGEEmbedder._load_model raises ImportError if sentence_transformers absent."""
    embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")
    with (
        patch.dict("sys.modules", {"sentence_transformers": None}),
        pytest.raises(ImportError, match="local-embed"),
    ):
        embedder._load_model()


def test_bge_embedder_embed_batch() -> None:
    """embed_batch encodes with the lazy-loaded model."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    # SentenceTransformer is imported inside _load_model — patch the source module
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")
        result = embedder.embed_batch(["a", "b"])

    assert len(result) == 2
    assert result[0] == pytest.approx([0.1, 0.2])
