"""Protocol defining the retrieval contract."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class RetrievalDiagnostics:
    """Ranked outputs and timings captured at each retrieval stage."""

    dense: list[RetrievedChunk]
    bm25: list[RetrievedChunk]
    hybrid_pre_rerank: list[RetrievedChunk]
    hybrid_post_rerank: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class CorpusStats:
    """Scale and metadata-quality summary for the indexed child corpus."""

    chunks: int
    unique_chunk_ids: int
    documents: int
    source_names: list[str]
    pages: int
    duplicate_text_chunks: int
    empty_text_chunks: int
    missing_source_chunks: int
    missing_page_chunks: int
    missing_parent_id_chunks: int


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Contract for retrieval backends."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Run the full retrieval pipeline for a query. Returns re-ranked chunks."""
        ...

    def retrieve_with_diagnostics(self, query: str) -> RetrievalDiagnostics:
        """Run retrieval and expose comparable ranked stages plus timings."""
        ...

    def corpus_stats(self) -> CorpusStats:
        """Return corpus scale and metadata-quality measurements."""
        ...

    def build_index(self, chunks: list[dict[str, object]]) -> None:
        """Build or rebuild the BM25 sparse index from a corpus of chunk dicts."""
        ...
