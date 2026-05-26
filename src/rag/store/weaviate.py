"""Weaviate store. Production backend — requires Docker.

Activate with STRATUM_STORE_BACKEND=weaviate.

Schema uses three collections:
  - DocumentChunk:       child chunks only, HNSW vector index, cosine distance
  - DocumentChunkParent: parent chunks only, no vector index (fetch-by-ID only)
  - StratumCorpus:       singleton storing BM25 corpus as a JSON blob

Parents are never ANN-searched — they live in a collection with no vector index.
BM25 corpus is stored in Weaviate so that vector state and BM25 state share
the same lifecycle. A cold restart restores both from a single source.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from rag.exceptions import SchemaError, StoreConnectionError, StoreError
from rag.interfaces.store import Chunk

logger = structlog.get_logger(__name__)

_CHILD_COLLECTION = "DocumentChunk"
_PARENT_COLLECTION = "DocumentChunkParent"
_CORPUS_COLLECTION = "StratumCorpus"
_CORPUS_SINGLETON_ID = "00000000-0000-0000-0000-000000000001"


class WeaviateStore:
    """Weaviate-backed document store implementing DocumentStoreProtocol."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        try:
            import weaviate  # noqa: PLC0415

            self._client = weaviate.connect_to_local(host=host, port=port)
            self._ensure_schema()
            logger.info("weaviate_connected", host=host, port=port)
        except Exception as exc:
            raise StoreConnectionError(host=host, port=port) from exc

    def _ensure_schema(self) -> None:
        """Create collections if they don't already exist (idempotent)."""
        try:
            import weaviate.classes.config as wc  # noqa: PLC0415

            existing = {c.name for c in self._client.collections.list_all().values()}

            if _CHILD_COLLECTION not in existing:
                self._client.collections.create(
                    name=_CHILD_COLLECTION,
                    properties=[
                        wc.Property(name="text", data_type=wc.DataType.TEXT),
                        wc.Property(name="source", data_type=wc.DataType.TEXT),
                        wc.Property(name="page", data_type=wc.DataType.INT),
                        wc.Property(name="chunk_index", data_type=wc.DataType.INT),
                        wc.Property(name="parent_id", data_type=wc.DataType.TEXT),
                    ],
                    vector_index_config=wc.Configure.VectorIndex.hnsw(
                        distance_metric=wc.VectorDistances.COSINE,
                        ef_construction=128,
                        max_connections=32,
                    ),
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                )

            if _PARENT_COLLECTION not in existing:
                self._client.collections.create(
                    name=_PARENT_COLLECTION,
                    properties=[
                        wc.Property(name="text", data_type=wc.DataType.TEXT),
                        wc.Property(name="source", data_type=wc.DataType.TEXT),
                        wc.Property(name="page", data_type=wc.DataType.INT),
                        wc.Property(name="chunk_index", data_type=wc.DataType.INT),
                    ],
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                )

            if _CORPUS_COLLECTION not in existing:
                self._client.collections.create(
                    name=_CORPUS_COLLECTION,
                    properties=[
                        wc.Property(name="corpus_json", data_type=wc.DataType.TEXT),
                        wc.Property(name="updated_at", data_type=wc.DataType.TEXT),
                    ],
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                )
        except Exception as exc:
            raise SchemaError(expected="DocumentChunk schema", actual=str(exc)) from exc

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Upsert parent chunks (no vectors) and child chunks (with vectors)."""
        if not chunks:
            return
        try:
            parents = [c for c in chunks if c.is_parent()]
            children = [c for c in chunks if not c.is_parent()]

            if parents:
                parent_col = self._client.collections.get(_PARENT_COLLECTION)
                with parent_col.batch.dynamic() as batch:
                    for chunk in parents:
                        batch.add_object(
                            properties={
                                "text": chunk.text,
                                "source": str(chunk.metadata.get("source", "")),
                                "page": int(chunk.metadata.get("page", 0) or 0),
                                "chunk_index": int(chunk.metadata.get("chunk_index", 0) or 0),
                            },
                            uuid=chunk.id,
                        )

            if children:
                child_embeddings = embeddings or []
                if len(child_embeddings) != len(children):
                    raise StoreError(
                        "Embeddings count must match child chunk count",
                        context={
                            "embeddings": len(child_embeddings),
                            "children": len(children),
                        },
                    )
                child_col = self._client.collections.get(_CHILD_COLLECTION)
                with child_col.batch.dynamic() as batch:
                    for chunk, vector in zip(children, child_embeddings, strict=False):
                        batch.add_object(
                            properties={
                                "text": chunk.text,
                                "source": str(chunk.metadata.get("source", "")),
                                "page": int(chunk.metadata.get("page", 0) or 0),
                                "chunk_index": int(chunk.metadata.get("chunk_index", 0) or 0),
                                "parent_id": chunk.parent_id or "",
                            },
                            uuid=chunk.id,
                            vector=vector,
                        )
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(f"Weaviate upsert failed: {exc}") from exc

    def semantic_search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """ANN search over child chunks only."""
        try:
            col = self._client.collections.get(_CHILD_COLLECTION)
            results = col.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=["distance"],
            )
            hits: list[dict[str, Any]] = []
            for obj in results.objects:
                hit: dict[str, Any] = {
                    "id": str(obj.uuid),
                    "text": obj.properties.get("text", ""),
                    "source": obj.properties.get("source", ""),
                    "page": obj.properties.get("page"),
                    "parent_id": obj.properties.get("parent_id", ""),
                    "distance": obj.metadata.distance if obj.metadata else 1.0,
                }
                hits.append(hit)
            return hits
        except Exception as exc:
            raise StoreError(f"Weaviate semantic search failed: {exc}") from exc

    def fetch_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch parent chunks by UUID list in a single batch query."""
        if not parent_ids:
            return []
        try:
            import weaviate.classes.query as wq  # noqa: PLC0415

            col = self._client.collections.get(_PARENT_COLLECTION)
            results = col.query.fetch_objects(
                filters=wq.Filter.by_id().contains_any(parent_ids),
                limit=len(parent_ids),
            )
            return [
                {
                    "id": str(obj.uuid),
                    "text": obj.properties.get("text", ""),
                    "source": obj.properties.get("source", ""),
                    "page": obj.properties.get("page"),
                }
                for obj in results.objects
            ]
        except Exception as exc:
            raise StoreError(f"Weaviate fetch_parents failed: {exc}") from exc

    def store_bm25_corpus(self, corpus: list[dict[str, Any]]) -> None:
        """Upsert BM25 corpus as a singleton JSON blob in StratumCorpus collection."""
        try:
            import datetime  # noqa: PLC0415

            col = self._client.collections.get(_CORPUS_COLLECTION)
            corpus_json = json.dumps(corpus, ensure_ascii=False)
            updated_at = datetime.datetime.now(datetime.UTC).isoformat()
            props = {"corpus_json": corpus_json, "updated_at": updated_at}

            existing = col.query.fetch_object_by_id(uuid=_CORPUS_SINGLETON_ID)
            if existing is None:
                col.data.insert(properties=props, uuid=_CORPUS_SINGLETON_ID)
                logger.info("weaviate_bm25_corpus_inserted", count=len(corpus))
            else:
                col.data.replace(uuid=_CORPUS_SINGLETON_ID, properties=props)
                logger.info("weaviate_bm25_corpus_stored", count=len(corpus))
        except Exception as exc:
            raise StoreError(f"Failed to store BM25 corpus in Weaviate: {exc}") from exc

    def load_bm25_corpus(self) -> list[dict[str, Any]]:
        """Load BM25 corpus from the StratumCorpus singleton. Returns [] if absent."""
        try:
            col = self._client.collections.get(_CORPUS_COLLECTION)
            obj = col.query.fetch_object_by_id(uuid=_CORPUS_SINGLETON_ID)
            if obj is None:
                return []
            corpus_json: str = str(obj.properties.get("corpus_json", "[]"))
            data: list[dict[str, Any]] = json.loads(corpus_json)
            logger.info("weaviate_bm25_corpus_loaded", count=len(data))
            return data
        except Exception as exc:
            raise StoreError(f"Failed to load BM25 corpus from Weaviate: {exc}") from exc

    def __enter__(self) -> WeaviateStore:
        return self

    def __exit__(self, *args: object) -> None:
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(Exception):
            self._client.close()
