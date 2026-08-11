"""Unit tests for HierarchicalChunker. No network, no file I/O."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from rag.config import Settings
from rag.ingestion.chunker import HierarchicalChunker


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key=SecretStr("test"),
        openai_api_key=SecretStr("test"),
        parent_token_size=200,
        child_token_size=50,
        overlap_sentences=2,
    )


@pytest.fixture
def chunker(settings: Settings) -> HierarchicalChunker:
    return HierarchicalChunker(settings)


def _make_text(approx_tokens: int) -> str:
    """Generate a text of approximately the given token count (1 token ~ 4 chars)."""
    sentence = "This is a sample sentence for testing purposes."
    repeat = max(1, (approx_tokens * 4) // len(sentence))
    return " ".join([sentence] * repeat)


def test_parent_child_relationship(chunker: HierarchicalChunker) -> None:
    """Every child.parent_id must exist in the set of parent IDs."""
    text = _make_text(3000)
    chunks = list(chunker.chunk_document(text, {"source": "test.pdf", "page": 1}))
    parents = {c.id for c in chunks if c.is_parent()}
    children = [c for c in chunks if not c.is_parent()]

    assert len(parents) > 0, "Expected at least one parent chunk"
    assert len(children) > 0, "Expected at least one child chunk"
    for child in children:
        assert child.parent_id in parents, (
            f"Child {child.id} has parent_id {child.parent_id!r} not found in parent set"
        )


def test_no_orphan_parents(chunker: HierarchicalChunker) -> None:
    """Every parent chunk must have at least one child."""
    text = _make_text(3000)
    chunks = list(chunker.chunk_document(text, {"source": "test.pdf"}))
    parent_ids = {c.id for c in chunks if c.is_parent()}
    child_parent_ids = {c.parent_id for c in chunks if not c.is_parent()}

    for pid in parent_ids:
        assert pid in child_parent_ids, f"Parent {pid} has no children"


def test_sentence_overlap(chunker: HierarchicalChunker) -> None:
    """Adjacent sibling children should share at least one sentence."""
    text = _make_text(3000)
    chunks = list(chunker.chunk_document(text, {"source": "test.pdf"}))
    parents = [c for c in chunks if c.is_parent()]

    for parent in parents:
        siblings = [c for c in chunks if c.parent_id == parent.id]
        if len(siblings) < 2:
            continue
        for prev, nxt in zip(siblings, siblings[1:], strict=False):
            prev_sentences = set(prev.text.split(". "))
            next_sentences = set(nxt.text.split(". "))
            overlap = prev_sentences & next_sentences
            assert len(overlap) >= 1, f"Adjacent children of parent {parent.id} share no sentences"


def test_token_budget(chunker: HierarchicalChunker) -> None:
    """No chunk should exceed 1.2x its target token size."""
    child_budget = chunker._child_token_size
    parent_budget = chunker._parent_token_size

    text = _make_text(3000)
    chunks = list(chunker.chunk_document(text, {"source": "test.pdf"}))

    for chunk in chunks:
        if chunk.is_parent():
            assert chunk.token_count <= parent_budget * 1.2, (
                f"Parent chunk {chunk.id} has {chunk.token_count} tokens "
                f"(budget: {parent_budget * 1.2})"
            )
        else:
            assert chunk.token_count <= child_budget * 1.2, (
                f"Child chunk {chunk.id} has {chunk.token_count} tokens "
                f"(budget: {child_budget * 1.2})"
            )


def test_metadata_propagation(chunker: HierarchicalChunker) -> None:
    """Children must inherit all metadata keys from the source document."""
    metadata = {"source": "test.pdf", "page": 3, "author": "Alice"}
    text = _make_text(3000)
    chunks = list(chunker.chunk_document(text, metadata))
    children = [c for c in chunks if not c.is_parent()]

    assert len(children) > 0
    for child in children:
        for key in metadata:
            assert key in child.metadata, f"Child {child.id} is missing metadata key '{key}'"


def test_empty_input(chunker: HierarchicalChunker) -> None:
    """chunk_document on empty string returns an empty iterator without raising."""
    result = list(chunker.chunk_document("", {}))
    assert result == []


def test_whitespace_only_input(chunker: HierarchicalChunker) -> None:
    """chunk_document on whitespace-only string returns empty iterator."""
    result = list(chunker.chunk_document("   \n\t  ", {}))
    assert result == []
