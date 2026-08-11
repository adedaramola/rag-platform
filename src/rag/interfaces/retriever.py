"""Protocol defining the retrieval contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, ready to pass to the generator."""

    id: str
    text: str  # retrieval unit text (typically the child chunk)
    source: str
    page: int | None  # None for web-sourced chunks
    score: float  # cross-encoder score after re-ranking
    parent_text: str | None = None  # expanded parent passage for generation context


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Contract for retrieval backends."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Run the full retrieval pipeline for a query. Returns re-ranked chunks."""
        ...

    def build_index(self, chunks: list[dict[str, object]]) -> None:
        """Build or rebuild the BM25 sparse index from a corpus of chunk dicts."""
        ...
