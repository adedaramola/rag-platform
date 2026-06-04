"""Single Pydantic Settings object. Every tunable value lives here — no exceptions."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all Stratum components.

    All values can be overridden via environment variables prefixed with STRATUM_
    or via a .env file. Example: STRATUM_STORE_BACKEND=weaviate
    """

    # Store backend
    store_backend: Literal["chroma", "weaviate"] = "chroma"

    # Chroma settings (default backend — no Docker required)
    chroma_persist_dir: Path = Path(".chroma")
    chroma_collection_name: str = "stratum"

    # Weaviate settings (production backend)
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080

    # Embedding settings
    # "openai" is the default — no GPU, no 2GB model download
    # Set embed_backend="local" to use BAAI/bge-large-en-v1.5 via sentence-transformers
    embed_backend: Literal["openai", "local"] = "openai"
    embed_model_openai: str = "text-embedding-3-small"
    embed_model_local: str = "BAAI/bge-large-en-v1.5"
    embed_batch_size: int = 32
    embed_dimensions: int = 1536  # text-embedding-3-small; set to 1024 for local BGE

    # Retrieval settings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_dense: int = 15
    top_k_rerank: int = 5

    # Chunking settings
    parent_token_size: int = 1500
    child_token_size: int = 300
    overlap_sentences: int = 2

    # LLM settings
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 250

    # Eval settings
    eval_judge_backend: Literal["ollama", "openai"] = "ollama"
    eval_judge_model: str = "llama3.1:8b"  # used when backend=ollama
    eval_judge_openai_model: str = "gpt-4o-mini"  # used when backend=openai
    eval_ollama_base_url: str = "http://localhost:11434"
    eval_golden_path: Path = Path("data/golden/qa_pairs.jsonl")
    eval_report_path: Path = Path("reports/deepeval_report.json")
    # warn_only=True: threshold violations log as warnings rather than failing the test.
    # Set to False only after running eval >=3 times on a stable pipeline and
    # establishing empirical baselines. See docs/evaluation.md.
    eval_warn_only: bool = True

    # Observability (Langfuse) — tracing is disabled when keys are absent
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Semantic cache
    # 'none'   — disabled (default, zero overhead)
    # 'memory' — in-memory LRU, lost on restart, no infrastructure required
    # 'redis'  — Redis-backed, persistent; requires STRATUM_REDIS_URL
    cache_backend: Literal["none", "memory", "redis"] = "none"
    cache_similarity_threshold: float = 0.95  # cosine similarity required for a hit
    cache_max_size: int = 1000  # memory backend only: max entries before LRU eviction
    cache_ttl_seconds: int | None = None  # redis backend only: None = no expiry
    redis_url: SecretStr | None = None  # e.g. redis://user:pass@host:6379

    # Secrets
    anthropic_api_key: SecretStr
    openai_api_key: SecretStr | None = None  # required when embed_backend="openai"
    # When set, all /query and /metrics requests must include X-API-Key: <value>.
    # Leave unset (default) in local dev — auth is skipped entirely when None.
    api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_prefix="STRATUM_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings singleton. Reads .env on first call."""
    return Settings()  # type: ignore[call-arg]
