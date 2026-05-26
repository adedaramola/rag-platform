"""Shared fixtures. Unit tests must never require a live network connection or API keys."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import SecretStr

from rag.config import Settings
from rag.interfaces.generator import CitationRef, CitedAnswer, GeneratorProtocol
from rag.interfaces.retriever import RetrievedChunk
from rag.interfaces.store import Chunk, DocumentStoreProtocol
from rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------


class MockEmbedder:
    """Deterministic embedder for unit tests. No network calls."""

    _DIM = 1536

    def embed(self, text: str) -> list[float]:
        """Return a deterministic unit vector derived from hash(text)."""
        seed = hash(text) % (10**6)
        raw = [(seed * (i + 1)) % 100 / 100.0 for i in range(self._DIM)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Mock store
# ---------------------------------------------------------------------------


class MockStore:
    """In-memory document store for unit tests. Satisfies DocumentStoreProtocol."""

    def __init__(self) -> None:
        self._children: dict[str, dict[str, Any]] = {}
        self._parents: dict[str, dict[str, Any]] = {}
        self._bm25_corpus: list[dict[str, Any]] = []

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        embs = embeddings or []
        children = [c for c in chunks if not c.is_parent()]
        parents = [c for c in chunks if c.is_parent()]

        for chunk in parents:
            self._parents[chunk.id] = {
                "id": chunk.id,
                "text": chunk.text,
                **chunk.metadata,
            }

        for i, chunk in enumerate(children):
            entry: dict[str, Any] = {
                "id": chunk.id,
                "text": chunk.text,
                "parent_id": chunk.parent_id,
                **chunk.metadata,
            }
            if i < len(embs):
                entry["_vector"] = embs[i]
            self._children[chunk.id] = entry

    def semantic_search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return top_k children by cosine similarity to query_vector."""
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)

        scored = [
            (chunk_id, cosine(query_vector, entry.get("_vector", query_vector)))
            for chunk_id, entry in self._children.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[dict[str, Any]] = []
        for chunk_id, score in scored[:top_k]:
            entry = dict(self._children[chunk_id])
            entry["distance"] = 1.0 - score
            results.append(entry)
        return results

    def fetch_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        return [self._parents[pid] for pid in parent_ids if pid in self._parents]

    def store_bm25_corpus(self, corpus: list[dict[str, Any]]) -> None:
        self._bm25_corpus = list(corpus)

    def load_bm25_corpus(self) -> list[dict[str, Any]]:
        return list(self._bm25_corpus)


# ---------------------------------------------------------------------------
# Mock generator
# ---------------------------------------------------------------------------


class MockGenerator:
    """Deterministic generator for unit tests. No API calls."""

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> CitedAnswer:
        answer = f"Answer to '{query}' based on {len(chunks)} sources. [src 1]"
        citations = [CitationRef(index=1, source=chunks[0].source, page=chunks[0].page)]
        return CitedAnswer(answer=answer, citations=citations, raw_context=chunks)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def mock_store() -> MockStore:
    return MockStore()


@pytest.fixture
def mock_pipeline(mock_store: MockStore, mock_embedder: MockEmbedder) -> RAGPipeline:
    """RAGPipeline wired with in-memory mocks. Zero network calls, no model downloads."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from rag.retrieval.hybrid import HybridRetriever

    mock_ce = MagicMock()
    # predict() must return something with .tolist() — use numpy array
    mock_ce.predict.side_effect = lambda pairs: np.array([0.9 - i * 0.1 for i in range(len(pairs))])

    with patch("sentence_transformers.CrossEncoder", return_value=mock_ce):
        retriever = HybridRetriever(
            store=mock_store,
            embedder=mock_embedder,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k_dense=5,
            top_k_rerank=3,
        )
    generator = MockGenerator()
    return RAGPipeline(retriever=retriever, generator=generator, embedder=mock_embedder)


@pytest.fixture
def settings() -> Settings:
    """Settings with safe test values. No real API keys needed."""
    return Settings(
        store_backend="chroma",
        embed_backend="openai",
        anthropic_api_key=SecretStr("test-anthropic-key"),
        openai_api_key=SecretStr("test-openai-key"),
        eval_warn_only=True,
    )


# ---------------------------------------------------------------------------
# Protocol compliance checks (run implicitly via isinstance)
# ---------------------------------------------------------------------------

_embedder_mod = __import__("rag.interfaces.embedder", fromlist=["EmbedderProtocol"])
assert isinstance(MockEmbedder(), _embedder_mod.EmbedderProtocol)
assert isinstance(MockStore(), DocumentStoreProtocol)
assert isinstance(MockGenerator(), GeneratorProtocol)
