"""Embedder implementations.

Two backends mirror the store design:
  OpenAIEmbedder — default. No GPU, no large download, ~1s cold start.
                   Requires STRATUM_OPENAI_API_KEY.
  BGEEmbedder    — opt-in via STRATUM_EMBED_BACKEND=local.
                   Requires: pip install 'stratum[local-embed]'

Both implement EmbedderProtocol identically. Swap via Settings — not via code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from rag.config import Settings
from rag.exceptions import ConfigurationError, EmbeddingError
from rag.interfaces.embedder import EmbedderProtocol

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class OpenAIEmbedder:
    """Embed text using the OpenAI embeddings API.

    Implements EmbedderProtocol. Default backend — no GPU required.
    """

    def __init__(self, model: str, api_key: str, batch_size: int = 32) -> None:
        self.model = model
        self._batch_size = batch_size
        try:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError("openai package is required. Run: pip install openai") from exc

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches. One API call per batch."""
        results: list[list[float]] = []
        try:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                response = self._client.embeddings.create(model=self.model, input=batch)
                results.extend(item.embedding for item in response.data)
        except Exception as exc:
            raise EmbeddingError(model=self.model) from exc
        return results


class BGEEmbedder:
    """Embed text using a local BGE model via sentence-transformers.

    Implements EmbedderProtocol. Opt-in via STRATUM_EMBED_BACKEND=local.
    Lazy-loads the model on first call to avoid startup overhead.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with normalised vectors."""
        self._load_model()
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [v.tolist() for v in vectors]
        except Exception as exc:
            raise EmbeddingError(model=self.model_name) from exc

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first call."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Install local embedding support: pip install 'stratum[local-embed]'"
            ) from exc
        logger.info("bge_model_loading", model=self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info("bge_model_ready", model=self.model_name)


def get_embedder(settings: Settings) -> EmbedderProtocol:
    """Factory: return the configured embedder backend."""
    if settings.embed_backend == "openai":
        if settings.openai_api_key is None:
            raise ConfigurationError(
                "STRATUM_OPENAI_API_KEY is required when embed_backend='openai'"
            )
        return OpenAIEmbedder(
            model=settings.embed_model_openai,
            api_key=settings.openai_api_key.get_secret_value(),
            batch_size=settings.embed_batch_size,
        )
    # embed_backend == "local"
    return BGEEmbedder(
        model_name=settings.embed_model_local,
        batch_size=settings.embed_batch_size,
    )
