"""Unit tests for semantic cache factory selection and validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from rag.cache.factory import get_cache
from rag.cache.memory import InMemorySemanticCache
from rag.config import Settings
from rag.exceptions import ConfigurationError


def _settings(**kwargs: Any) -> Settings:
    return Settings(
        anthropic_api_key=SecretStr("test"),
        openai_api_key=SecretStr("test"),
        **kwargs,
    )


def test_get_cache_none_returns_none() -> None:
    """Default cache backend 'none' returns no cache instance."""
    settings = _settings(cache_backend="none")
    assert get_cache(settings) is None


def test_get_cache_memory_returns_in_memory_cache() -> None:
    """Memory backend returns an InMemorySemanticCache with configured settings."""
    settings = _settings(
        cache_backend="memory",
        cache_max_size=42,
        cache_similarity_threshold=0.88,
    )
    cache = get_cache(settings)
    assert isinstance(cache, InMemorySemanticCache)
    assert cache._max_size == 42
    assert cache._threshold == pytest.approx(0.88)


def test_get_cache_redis_missing_url_raises_configuration_error() -> None:
    """Redis backend requires STRATUM_REDIS_URL."""
    settings = _settings(cache_backend="redis")
    object.__setattr__(settings, "redis_url", None)

    with (
        patch("rag.cache.redis_cache.RedisSemanticCache"),
        pytest.raises(ConfigurationError, match="STRATUM_REDIS_URL"),
    ):
        get_cache(settings)


def test_get_cache_redis_returns_redis_cache() -> None:
    """Redis backend forwards the configured URL and thresholds to the constructor."""
    settings = _settings(
        cache_backend="redis",
        redis_url=SecretStr("redis://localhost:6379/0"),
        cache_similarity_threshold=0.91,
        cache_ttl_seconds=600,
    )

    with patch("rag.cache.redis_cache.RedisSemanticCache", autospec=True) as mock_cache_cls:
        sentinel = object()
        mock_cache_cls.return_value = sentinel

        cache = get_cache(settings)

    assert cache is sentinel
    mock_cache_cls.assert_called_once_with(
        redis_url="redis://localhost:6379/0",
        similarity_threshold=0.91,
        ttl_seconds=600,
    )


def test_get_cache_unknown_backend_raises_configuration_error() -> None:
    """Unexpected cache backend values raise ConfigurationError."""
    settings = _settings()
    object.__setattr__(settings, "cache_backend", "mystery")

    with pytest.raises(ConfigurationError, match="Unknown cache backend"):
        get_cache(settings)
