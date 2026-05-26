"""In-memory LRU semantic cache.

Uses an OrderedDict for O(1) LRU eviction and numpy for fast cosine similarity.
Thread-safe via a single RLock — safe for concurrent FastAPI requests.

Capacity is bounded by cache_max_size (number of entries). There is no TTL;
entries are evicted only when the cache is full. Use the Redis backend if you
need TTL-based expiry or persistence across restarts.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from rag.interfaces.generator import CitedAnswer


@dataclass
class _Entry:
    embedding: list[float]
    answer: CitedAnswer


class InMemorySemanticCache:
    """LRU semantic cache backed by an in-memory OrderedDict.

    Implements SemanticCacheProtocol. Similarity is cosine distance computed
    with numpy. The least-recently-used entry is evicted when the cache is full.
    """

    def __init__(self, max_size: int, similarity_threshold: float) -> None:
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._store: OrderedDict[int, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._next_id = 0
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------
    # SemanticCacheProtocol
    # ------------------------------------------------------------------

    def get(self, query_embedding: list[float]) -> CitedAnswer | None:
        """Return the best-matching cached answer, or None on a miss."""
        with self._lock:
            if not self._store:
                self._miss_count += 1
                return None

            q = np.array(query_embedding, dtype=np.float32)
            best_sim = -1.0
            best_key = -1

            for key, entry in self._store.items():
                sim = _cosine_similarity(q, np.array(entry.embedding, dtype=np.float32))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_key != -1 and best_sim >= self._threshold:
                self._store.move_to_end(best_key)
                self._hit_count += 1
                return self._store[best_key].answer

            self._miss_count += 1
            return None

    def set(self, query_embedding: list[float], answer: CitedAnswer) -> None:
        """Store the embedding→answer pair, evicting LRU entry if at capacity."""
        with self._lock:
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            key = self._next_id
            self._next_id += 1
            self._store[key] = _Entry(embedding=query_embedding, answer=answer)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hit_count = 0
            self._miss_count = 0

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._store)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 arrays. Returns 0.0 for zero vectors."""
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
