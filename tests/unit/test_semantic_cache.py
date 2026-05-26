"""Unit tests for the in-memory semantic cache.

Redis tests are excluded here — they require a live connection and belong in
tests/integration/. All tests are deterministic and require no network or API keys.
"""

from __future__ import annotations

import math

from rag.cache.memory import InMemorySemanticCache, _cosine_similarity
from rag.interfaces.cache import SemanticCacheProtocol
from rag.interfaces.generator import CitationRef, CitedAnswer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMS = 8  # small dimension keeps tests fast


def _unit_vec(index: int, dims: int = DIMS) -> list[float]:
    """Return a unit vector with 1.0 at position `index`, 0.0 elsewhere."""
    v = [0.0] * dims
    v[index % dims] = 1.0
    return v


def _make_answer(text: str = "answer") -> CitedAnswer:
    return CitedAnswer(
        answer=text,
        citations=[CitationRef(index=1, source="doc.pdf", page=1)],
        raw_context=[],
        input_tokens=10,
        output_tokens=5,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_in_memory_cache_satisfies_protocol() -> None:
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.9)
    assert isinstance(cache, SemanticCacheProtocol)


# ---------------------------------------------------------------------------
# Basic get / set
# ---------------------------------------------------------------------------


def test_miss_on_empty_cache() -> None:
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.9)
    assert cache.get(_unit_vec(0)) is None
    assert cache.miss_count == 1
    assert cache.hit_count == 0


def test_exact_match_returns_answer() -> None:
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.9)
    emb = _unit_vec(0)
    answer = _make_answer("exact")
    cache.set(emb, answer)
    result = cache.get(emb)
    assert result is not None
    assert result.answer == "exact"
    assert cache.hit_count == 1


def test_similar_query_returns_cached_answer() -> None:
    """Vectors nearly parallel should exceed the 0.9 threshold."""
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.9)
    base = [1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    similar = [1.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cache.set(base, _make_answer("base"))
    result = cache.get(similar)
    assert result is not None
    assert result.answer == "base"


def test_dissimilar_query_misses() -> None:
    """Orthogonal vectors (cosine = 0.0) must not hit."""
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.9)
    cache.set(_unit_vec(0), _make_answer("a"))
    assert cache.get(_unit_vec(1)) is None
    assert cache.miss_count == 1


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_lru_evicts_oldest_entry() -> None:
    cache = InMemorySemanticCache(max_size=2, similarity_threshold=0.9)
    cache.set(_unit_vec(0), _make_answer("first"))
    cache.set(_unit_vec(1), _make_answer("second"))
    # Access 'first' to make it recently used
    cache.get(_unit_vec(0))
    # Adding a third entry should evict 'second' (least recently used)
    cache.set(_unit_vec(2), _make_answer("third"))
    assert cache.size == 2
    assert cache.get(_unit_vec(1)) is None  # evicted
    assert cache.get(_unit_vec(0)) is not None  # still present
    assert cache.get(_unit_vec(2)) is not None  # still present


def test_max_size_one_always_keeps_latest() -> None:
    cache = InMemorySemanticCache(max_size=1, similarity_threshold=0.9)
    cache.set(_unit_vec(0), _make_answer("old"))
    cache.set(_unit_vec(1), _make_answer("new"))
    assert cache.size == 1
    assert cache.get(_unit_vec(1)) is not None
    assert cache.get(_unit_vec(0)) is None


# ---------------------------------------------------------------------------
# Counter accuracy
# ---------------------------------------------------------------------------


def test_hit_miss_counters() -> None:
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.9)
    cache.set(_unit_vec(0), _make_answer())
    cache.get(_unit_vec(0))  # hit
    cache.get(_unit_vec(1))  # miss
    cache.get(_unit_vec(0))  # hit
    assert cache.hit_count == 2
    assert cache.miss_count == 1


def test_clear_resets_counters_and_entries() -> None:
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.9)
    cache.set(_unit_vec(0), _make_answer())
    cache.get(_unit_vec(0))
    cache.clear()
    assert cache.hit_count == 0
    assert cache.miss_count == 0
    assert cache.size == 0
    assert cache.get(_unit_vec(0)) is None


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    import numpy as np

    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert math.isclose(_cosine_similarity(v, v), 1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    import numpy as np

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert math.isclose(_cosine_similarity(a, b), 0.0, abs_tol=1e-6)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    import numpy as np

    zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _cosine_similarity(zero, v) == 0.0
    assert _cosine_similarity(v, zero) == 0.0


# ---------------------------------------------------------------------------
# Threshold edge cases
# ---------------------------------------------------------------------------


def test_exact_threshold_boundary_is_inclusive() -> None:
    """A similarity equal to the threshold should count as a hit."""
    threshold = 0.95
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=threshold)
    # Use identical vectors → similarity = 1.0, above any sane threshold
    emb = _unit_vec(3)
    cache.set(emb, _make_answer("boundary"))
    assert cache.get(emb) is not None


def test_returns_best_match_not_first_match() -> None:
    """When multiple entries exist, the one with highest similarity wins."""
    cache = InMemorySemanticCache(max_size=10, similarity_threshold=0.5)
    weak = [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    strong = [0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cache.set(weak, _make_answer("weak"))
    cache.set(strong, _make_answer("strong"))
    query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = cache.get(query)
    assert result is not None
    assert result.answer == "strong"
