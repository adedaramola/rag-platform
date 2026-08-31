"""Maps RAG_PLATFORM_CACHE_BACKEND to a concrete SemanticCacheProtocol implementation."""

from __future__ import annotations

from rag.config import Settings
from rag.exceptions import ConfigurationError
from rag.interfaces.cache import SemanticCacheProtocol


def get_cache(settings: Settings) -> SemanticCacheProtocol | None:
    """Return the configured cache backend, or None when caching is disabled.

    'none'   → returns None  (default — no caching overhead)
    'memory' → InMemorySemanticCache (zero infrastructure, lost on restart)
    'redis'  → RedisSemanticCache    (persistent, requires RAG_PLATFORM_REDIS_URL)
    """
    if settings.cache_backend == "none":
        return None

    if settings.cache_backend == "memory":
        from rag.cache.memory import InMemorySemanticCache  # noqa: PLC0415

        return InMemorySemanticCache(
            max_size=settings.cache_max_size,
            similarity_threshold=settings.cache_similarity_threshold,
        )

    if settings.cache_backend == "redis":
        from rag.cache.redis_cache import RedisSemanticCache  # noqa: PLC0415

        if settings.redis_url is None:
            raise ConfigurationError(
                "RAG_PLATFORM_REDIS_URL must be set when RAG_PLATFORM_CACHE_BACKEND=redis"
            )
        return RedisSemanticCache(
            redis_url=settings.redis_url.get_secret_value(),
            similarity_threshold=settings.cache_similarity_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
        )

    raise ConfigurationError(f"Unknown cache backend: {settings.cache_backend!r}")
