"""Semantic cache protocol. All cache backends must satisfy this contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.interfaces.generator import CitedAnswer


@runtime_checkable
class SemanticCacheProtocol(Protocol):
    """Contract for semantic query caches.

    Implementations store (embedding, CitedAnswer) pairs and return a cached
    answer when an incoming query embedding is sufficiently similar to a stored
    one. The similarity threshold is an implementation detail.
    """

    def get(self, query_embedding: list[float]) -> CitedAnswer | None:
        """Return a cached answer if a similar query exists, else None."""
        ...

    def set(self, query_embedding: list[float], answer: CitedAnswer) -> None:
        """Store the embedding→answer mapping for future lookups."""
        ...

    def clear(self) -> None:
        """Evict all cached entries and reset counters."""
        ...

    @property
    def hit_count(self) -> int:
        """Number of successful cache lookups since creation or last clear."""
        ...

    @property
    def miss_count(self) -> int:
        """Number of unsuccessful cache lookups since creation or last clear."""
        ...
