"""Unit tests for Redis-backed semantic cache behavior."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta
from typing import Any

from rag.cache.redis_cache import RedisSemanticCache, _encode_embedding
from rag.interfaces.generator import CitationRef, CitedAnswer


class FakePipeline:
    def __init__(self, client: FakeRedisClient) -> None:
        self.client = client
        self.commands: list[tuple[str, Any]] = []

    def hset(self, key: str, item_id: str, value: Any) -> FakePipeline:
        self.commands.append(("hset", (key, item_id, value)))
        return self

    def hdel(self, key: str, *item_ids: Any) -> FakePipeline:
        self.commands.append(("hdel", (key, item_ids)))
        return self

    def execute(self) -> None:
        self.client.executed.append(self.commands)


class FakeRedisClient:
    def __init__(self) -> None:
        self.ping_calls = 0
        self._hgetall_result: dict[bytes, bytes] | None = None
        self._hget_result: bytes | None = None
        self._hgetall_error: Exception | None = None
        self.executed: list[list[tuple[str, Any]]] = []
        self.deleted: list[tuple[str, ...]] = []

    def ping(self) -> None:
        self.ping_calls += 1

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        if self._hgetall_error is not None:
            raise self._hgetall_error
        if self._hgetall_result is None:
            return {}
        return self._hgetall_result

    def hget(self, key: str, item_id: bytes | str) -> bytes | None:
        return self._hget_result

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    def delete(self, *keys: str) -> int:
        self.deleted.append(keys)
        return 1

    def hdel(self, key: str, *item_ids: Any) -> None:
        return None


def _make_answer(text: str = "answer") -> CitedAnswer:
    return CitedAnswer(
        answer=text,
        citations=[CitationRef(index=1, source="doc.pdf", page=1)],
        raw_context=[],
        input_tokens=10,
        output_tokens=5,
    )


def _install_fake_redis(monkeypatch: Any, client: FakeRedisClient) -> None:
    fake_redis_module = types.ModuleType("redis")
    fake_redis_module.from_url = lambda url, decode_responses=False: client
    monkeypatch.setitem(sys.modules, "redis", fake_redis_module)


def test_redis_cache_initializes_and_pings(monkeypatch: Any) -> None:
    client = FakeRedisClient()
    _install_fake_redis(monkeypatch, client)

    cache = RedisSemanticCache(
        redis_url="redis://localhost:6379/0",
        similarity_threshold=0.8,
        ttl_seconds=30,
        key_prefix="test-prefix",
    )

    assert cache.hit_count == 0
    assert cache.miss_count == 0
    assert client.ping_calls == 1
    assert cache._emb_key == "test-prefix:emb"
    assert cache._ttl == 30


def test_redis_cache_get_returns_none_when_lookup_fails(monkeypatch: Any) -> None:
    client = FakeRedisClient()
    client._hgetall_error = RuntimeError("boom")
    _install_fake_redis(monkeypatch, client)

    cache = RedisSemanticCache("redis://localhost:6379/0", similarity_threshold=0.8)
    result = cache.get([1.0, 0.0])

    assert result is None
    assert cache.miss_count == 1
    assert cache.hit_count == 0


def test_redis_cache_get_returns_best_matching_answer(monkeypatch: Any) -> None:
    client = FakeRedisClient()
    client._hgetall_result = {
        b"one": _encode_embedding([1.0, 0.0]),
        b"two": _encode_embedding([0.0, 1.0]),
    }
    client._hget_result = (
        b'{"answer": "cached", "citations": [], "raw_context": [], '
        b'"input_tokens": 0, "output_tokens": 0}'
    )
    _install_fake_redis(monkeypatch, client)

    cache = RedisSemanticCache("redis://localhost:6379/0", similarity_threshold=0.5)
    result = cache.get([1.0, 0.0])

    assert result is not None
    assert result.answer == "cached"
    assert cache.hit_count == 1
    assert cache.miss_count == 0


def test_redis_cache_set_and_clear_use_pipeline_and_reset_counters(monkeypatch: Any) -> None:
    client = FakeRedisClient()
    _install_fake_redis(monkeypatch, client)

    cache = RedisSemanticCache("redis://localhost:6379/0", similarity_threshold=0.9)
    cache.set([1.0, 0.0], _make_answer("stored"))
    cache.clear()

    assert len(client.executed) == 1
    assert client.deleted == [(cache._emb_key, cache._ans_key, cache._ts_key)]
    assert cache.hit_count == 0
    assert cache.miss_count == 0


def test_redis_cache_evicts_stale_entries(monkeypatch: Any) -> None:
    client = FakeRedisClient()
    old_ts = (datetime.now(UTC) - timedelta(seconds=30)).isoformat().encode()
    new_ts = (datetime.now(UTC) - timedelta(seconds=5)).isoformat().encode()
    client._hgetall_result = {
        b"stale": old_ts,
        b"fresh": new_ts,
    }
    _install_fake_redis(monkeypatch, client)

    cache = RedisSemanticCache("redis://localhost:6379/0", similarity_threshold=0.9, ttl_seconds=10)
    cache._evict_stale()

    assert len(client.executed) == 1
    assert client.executed[0][0][0] == "hdel"
    assert client.executed[0][0][1][1] == (b"stale",)
