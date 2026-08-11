"""Unit tests for the store factory."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from rag.config import Settings
from rag.exceptions import StoreError
from rag.interfaces.store import DocumentStoreProtocol
from rag.store.factory import get_store


def _settings(**kwargs: Any) -> Settings:
    return Settings(
        anthropic_api_key=SecretStr("test"),
        openai_api_key=SecretStr("test"),
        **kwargs,
    )


def test_get_store_chroma_returns_protocol() -> None:
    """get_store with chroma backend returns a DocumentStoreProtocol."""
    with tempfile.TemporaryDirectory() as d:
        s = _settings(store_backend="chroma", chroma_persist_dir=Path(d))
        store = get_store(s)
        assert isinstance(store, DocumentStoreProtocol)


def test_get_store_chroma_bm25_round_trip() -> None:
    """ChromaStore from factory can store and load BM25 corpus."""
    with tempfile.TemporaryDirectory() as d:
        s = _settings(store_backend="chroma", chroma_persist_dir=Path(d))
        store = get_store(s)
        corpus = [{"id": "1", "text": "hello world"}]
        store.store_bm25_corpus(corpus)
        assert store.load_bm25_corpus() == corpus


def test_get_store_unknown_backend_raises() -> None:
    """Unknown store_backend raises StoreError."""
    with tempfile.TemporaryDirectory() as d:
        s = _settings(store_backend="chroma", chroma_persist_dir=Path(d))
        # Bypass Pydantic Literal validation by monkey-patching the attribute
        object.__setattr__(s, "store_backend", "unknown_backend")
        with pytest.raises(StoreError):
            get_store(s)
