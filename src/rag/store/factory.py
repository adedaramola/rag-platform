"""Store factory mapping RAG_PLATFORM_STORE_BACKEND to a concrete implementation.

All downstream code works against DocumentStoreProtocol — never against a concrete class.
Swapping backends requires one env var change, not a code change.
"""

from rag.config import Settings
from rag.exceptions import StoreError
from rag.interfaces.store import DocumentStoreProtocol


def get_store(settings: Settings) -> DocumentStoreProtocol:
    """Return the configured document store backend."""
    if settings.store_backend == "chroma":
        from rag.store.chroma import ChromaStore  # noqa: PLC0415

        return ChromaStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            dimensions=settings.embed_dimensions,
        )
    if settings.store_backend == "weaviate":
        from rag.store.weaviate import WeaviateStore  # noqa: PLC0415

        return WeaviateStore(host=settings.weaviate_host, port=settings.weaviate_port)
    raise StoreError(
        f"Unknown store backend: {settings.store_backend!r}. "
        "Set RAG_PLATFORM_STORE_BACKEND to 'chroma' or 'weaviate'.",
        context={"store_backend": settings.store_backend},
    )
