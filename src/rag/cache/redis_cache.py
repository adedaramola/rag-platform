"""Redis-backed semantic cache.

Opt-in via RAG_PLATFORM_CACHE_BACKEND=redis + RAG_PLATFORM_REDIS_URL=redis://...

Data layout (all under one configurable key prefix):
  {prefix}:emb  — Redis Hash  { item_id → base64(float32 bytes) }
  {prefix}:ans  — Redis Hash  { item_id → JSON(CitedAnswer)      }
  {prefix}:ts   — Redis Hash  { item_id → ISO-8601 timestamp     }

Why no RediSearch: the free Redis Cloud tier does not include the vector
search module. Embeddings are fetched in one HGETALL and similarity is
computed in Python — acceptable at free-tier scale (~3 750 entries × 6 KB).

Eviction strategy: TTL-based (remove entries older than cache_ttl_seconds on
each get() call). When no TTL is configured the cache grows until evicted by
a clear() call or external Redis management.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import struct
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

from rag.exceptions import StoreError
from rag.interfaces.generator import CitationRef, CitedAnswer
from rag.interfaces.retriever import RetrievedChunk

logger = structlog.get_logger(__name__)


class RedisSemanticCache:
    """Redis-backed semantic cache with optional TTL-based eviction.

    Implements SemanticCacheProtocol. Requires redis>=5.0.0 (pip install
    'rag-platform[cache]'). Connection errors raise StoreError.
    """

    def __init__(
        self,
        redis_url: str,
        similarity_threshold: float,
        ttl_seconds: int | None = None,
        key_prefix: str = "rag-platform:cache",
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0

        try:
            import redis  # noqa: PLC0415

            self._client: Any = redis.from_url(redis_url, decode_responses=False)
            self._client.ping()
        except ImportError as exc:
            raise ImportError(
                "Redis client not installed. Run: pip install 'rag-platform[cache]'"
            ) from exc
        except Exception as exc:
            raise StoreError(f"Redis connection failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @property
    def _emb_key(self) -> str:
        return f"{self._prefix}:emb"

    @property
    def _ans_key(self) -> str:
        return f"{self._prefix}:ans"

    @property
    def _ts_key(self) -> str:
        return f"{self._prefix}:ts"

    # ------------------------------------------------------------------
    # SemanticCacheProtocol
    # ------------------------------------------------------------------

    def get(self, query_embedding: list[float]) -> CitedAnswer | None:
        """Return the best-matching cached answer, or None on a miss."""
        with self._lock:
            if self._ttl is not None:
                self._evict_stale()

            try:
                raw_embs: dict[bytes, bytes] = self._client.hgetall(self._emb_key)
            except Exception as exc:
                logger.warning("redis_cache_get_error", error=str(exc))
                self._miss_count += 1
                return None

            if not raw_embs:
                self._miss_count += 1
                return None

            q = np.array(query_embedding, dtype=np.float32)
            best_sim = -1.0
            best_item_id: bytes | None = None

            for item_id_bytes, emb_bytes in raw_embs.items():
                stored = _decode_embedding(emb_bytes)
                sim = _cosine_similarity(q, stored)
                if sim > best_sim:
                    best_sim = sim
                    best_item_id = item_id_bytes

            if best_item_id is not None and best_sim >= self._threshold:
                try:
                    ans_bytes: bytes | None = self._client.hget(self._ans_key, best_item_id)
                except Exception as exc:
                    logger.warning("redis_cache_ans_fetch_error", error=str(exc))
                    self._miss_count += 1
                    return None

                if ans_bytes is not None:
                    answer = _deserialize_answer(ans_bytes.decode())
                    self._hit_count += 1
                    return answer

            self._miss_count += 1
            return None

    def set(self, query_embedding: list[float], answer: CitedAnswer) -> None:
        """Store the embedding→answer pair in Redis."""
        item_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        try:
            pipe = self._client.pipeline()
            pipe.hset(self._emb_key, item_id, _encode_embedding(query_embedding))
            pipe.hset(self._ans_key, item_id, _serialize_answer(answer))
            pipe.hset(self._ts_key, item_id, now.encode())
            pipe.execute()
        except Exception as exc:
            logger.warning("redis_cache_set_error", error=str(exc))

    def clear(self) -> None:
        """Delete all cache keys and reset hit/miss counters."""
        with self._lock:
            try:
                self._client.delete(self._emb_key, self._ans_key, self._ts_key)
            except Exception as exc:
                logger.warning("redis_cache_clear_error", error=str(exc))
            self._hit_count = 0
            self._miss_count = 0

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_stale(self) -> None:
        """Remove entries whose timestamp exceeds cache_ttl_seconds."""
        if self._ttl is None:
            return
        try:
            raw_ts: dict[bytes, bytes] = self._client.hgetall(self._ts_key)
        except Exception:
            return

        cutoff = datetime.now(UTC).timestamp() - self._ttl
        stale = [
            item_id
            for item_id, ts_bytes in raw_ts.items()
            if datetime.fromisoformat(ts_bytes.decode()).timestamp() < cutoff
        ]
        if stale:
            try:
                pipe = self._client.pipeline()
                pipe.hdel(self._emb_key, *stale)
                pipe.hdel(self._ans_key, *stale)
                pipe.hdel(self._ts_key, *stale)
                pipe.execute()
                logger.debug("redis_cache_evicted_stale", count=len(stale))
            except Exception as exc:
                logger.warning("redis_cache_evict_error", error=str(exc))


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------


def _encode_embedding(embedding: list[float]) -> bytes:
    """Pack a float list to raw bytes (float32 LE) then base64-encode."""
    raw = struct.pack(f"{len(embedding)}f", *embedding)
    return base64.b64encode(raw)


def _decode_embedding(data: bytes) -> np.ndarray:
    """Reverse of _encode_embedding."""
    raw = base64.b64decode(data)
    count = len(raw) // 4
    floats = struct.unpack(f"{count}f", raw)
    return np.array(floats, dtype=np.float32)


def _serialize_answer(answer: CitedAnswer) -> str:
    return json.dumps(dataclasses.asdict(answer))


def _deserialize_answer(raw: str) -> CitedAnswer:
    data = json.loads(raw)
    return CitedAnswer(
        answer=data["answer"],
        citations=[CitationRef(**c) for c in data["citations"]],
        raw_context=[RetrievedChunk(**r) for r in data["raw_context"]],
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
