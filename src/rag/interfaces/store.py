"""Protocol defining the document store contract.

BM25 corpus persistence is part of this contract — not a separate concern.
Storing the BM25 corpus alongside the vector index ensures both are always
in sync. A cold restart restores the BM25 index automatically via load_bm25_corpus().

Design note: upsert_chunks accepts embeddings as list[list[float]] | None.
Pass None (or an empty list) for parent chunks — they have no vector index.
Implementations must handle this without raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class Chunk:
    """A document chunk produced by HierarchicalChunker."""

    id: str
    text: str
    metadata: dict[str, Any]
    parent_id: str | None
    token_count: int

    def is_parent(self) -> bool:
        """Return True if this chunk has no parent (i.e., it IS a parent chunk)."""
        return self.parent_id is None


@runtime_checkable
class DocumentStoreProtocol(Protocol):
    """Contract for vector + document store backends.

    Implementations: ChromaStore (default), WeaviateStore (production).
    Swap via RAG_PLATFORM_STORE_BACKEND env var — all downstream code uses this Protocol.
    """

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Insert or update chunks. Pass embeddings=None for parent chunks (no vectors)."""
        ...

    def semantic_search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """ANN search over child chunks. Never returns parent chunks."""
        ...

    def fetch_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch parent chunks by ID. Used for context expansion after retrieval."""
        ...

    def store_bm25_corpus(self, corpus: list[dict[str, Any]]) -> None:
        """Persist the BM25 corpus alongside the vector index."""
        ...

    def load_bm25_corpus(self) -> list[dict[str, Any]]:
        """Load the BM25 corpus. Returns [] if no corpus has been stored yet."""
        ...
