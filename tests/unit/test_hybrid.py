"""Unit tests for HybridRetriever (build_index, retrieve) with mocked dependencies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.interfaces.retriever import RetrievedChunk
from rag.interfaces.store import Chunk
from rag.retrieval.hybrid import HybridRetriever


def _make_retriever(mock_store: object, mock_embedder: object) -> HybridRetriever:
    """Build a HybridRetriever with the CrossEncoder patched out."""
    import numpy as np

    mock_ce = MagicMock()
    # predict() must return something with .tolist() — use numpy array
    mock_ce.predict.side_effect = lambda pairs: np.array([0.9 - i * 0.1 for i in range(len(pairs))])

    with patch("sentence_transformers.CrossEncoder", return_value=mock_ce):
        retriever = HybridRetriever(
            store=mock_store,  # type: ignore[arg-type]
            embedder=mock_embedder,  # type: ignore[arg-type]
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k_dense=10,
            top_k_rerank=3,
        )
    return retriever


def _seed_store(store: object, embedder: object, n: int = 5) -> None:
    """Populate mock_store with n parent-child pairs."""
    from tests.conftest import MockEmbedder, MockStore

    assert isinstance(store, MockStore)
    assert isinstance(embedder, MockEmbedder)

    parents = [
        Chunk(
            id=f"p{i}",
            text=f"parent passage {i}",
            metadata={"source": "doc.pdf", "page": i},
            parent_id=None,
            token_count=50,
        )
        for i in range(n)
    ]
    children = [
        Chunk(
            id=f"c{i}",
            text=f"child chunk {i} about topic",
            metadata={"source": "doc.pdf", "page": i},
            parent_id=f"p{i}",
            token_count=20,
        )
        for i in range(n)
    ]
    vecs = embedder.embed_batch([c.text for c in children])
    store.upsert_chunks(parents)
    store.upsert_chunks(children, embeddings=vecs)
    corpus = [
        {"id": c.id, "text": c.text, "source": "doc.pdf", "page": c.metadata["page"]}
        for c in children
    ]
    store.store_bm25_corpus(corpus)


def test_build_index_empty_corpus(mock_store: object, mock_embedder: object) -> None:
    """build_index with empty corpus must not raise."""
    retriever = _make_retriever(mock_store, mock_embedder)
    retriever.build_index([])  # must not raise


def test_build_index_populates_bm25(mock_store: object, mock_embedder: object) -> None:
    """After build_index, BM25 object is set."""
    retriever = _make_retriever(mock_store, mock_embedder)
    corpus = [{"id": "1", "text": "hello world", "source": "s", "page": 0}]
    retriever.build_index(corpus)
    assert retriever._bm25 is not None


def test_retrieve_returns_reranked_chunks(mock_store: object, mock_embedder: object) -> None:
    """retrieve() returns a list of RetrievedChunks sorted by score."""
    retriever = _make_retriever(mock_store, mock_embedder)
    _seed_store(mock_store, mock_embedder)
    retriever.build_index(retriever._store.load_bm25_corpus())

    results = retriever.retrieve("topic question")
    assert len(results) > 0
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_respects_top_k_rerank(mock_store: object, mock_embedder: object) -> None:
    """retrieve() returns at most top_k_rerank results."""
    retriever = _make_retriever(mock_store, mock_embedder)
    _seed_store(mock_store, mock_embedder, n=10)
    retriever.build_index(retriever._store.load_bm25_corpus())

    results = retriever.retrieve("any question")
    assert len(results) <= 3  # top_k_rerank=3


def test_retrieve_empty_store_returns_empty(mock_store: object, mock_embedder: object) -> None:
    """retrieve() against an empty store returns an empty list."""
    retriever = _make_retriever(mock_store, mock_embedder)
    results = retriever.retrieve("anything")
    assert results == []


def test_retrieve_results_have_parent_text(mock_store: object, mock_embedder: object) -> None:
    """Parent expansion should preserve child text and carry parent context separately."""
    retriever = _make_retriever(mock_store, mock_embedder)
    _seed_store(mock_store, mock_embedder)
    retriever.build_index(retriever._store.load_bm25_corpus())

    results = retriever.retrieve("parent passage topic")
    assert any("child chunk" in r.text for r in results)
    assert any((r.parent_text or "").startswith("parent passage") for r in results)
