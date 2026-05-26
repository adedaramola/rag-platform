"""ChromaDB store. Default backend — zero infrastructure required.

Persists to disk at chroma_persist_dir. Suitable for development and CI.

Parent and child chunks are stored in separate collections:
  - {collection_name}: child chunks with vector index (retrieval targets)
  - {collection_name}_parents: parent chunks, no vector index (fetch-by-ID only)

This design means parents can never appear in ANN search results — there is no
vector index on the parent collection and no filter that could be accidentally omitted.

BM25 corpus is stored as a JSON sidecar at {persist_dir}/bm25_corpus.json,
co-located with the Chroma persist directory so both are always in sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from rag.exceptions import StoreConnectionError, StoreError
from rag.interfaces.store import Chunk

logger = structlog.get_logger(__name__)


class ChromaStore:
    """Chroma-backed document store implementing DocumentStoreProtocol."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        dimensions: int,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._dimensions = dimensions
        self._bm25_path = persist_dir / "bm25_corpus.json"
        self._client: Any = None
        self._children: Any = None
        self._parents: Any = None
        self._connect()

    def _connect(self) -> None:
        try:
            import chromadb  # noqa: PLC0415

            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
            self._children = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._parents = self._client.get_or_create_collection(
                name=f"{self._collection_name}_parents",
            )
            logger.info(
                "chroma_connected",
                persist_dir=str(self._persist_dir),
                collection=self._collection_name,
            )
        except Exception as exc:
            raise StoreConnectionError(host="local", port=0) from exc

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Route parent chunks to the parent collection and children to child collection."""
        if not chunks:
            return
        try:
            parents = [c for c in chunks if c.is_parent()]
            children = [c for c in chunks if not c.is_parent()]

            if parents:
                self._parents.upsert(
                    ids=[c.id for c in parents],
                    documents=[c.text for c in parents],
                    metadatas=[c.metadata for c in parents],
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
                self._children.upsert(
                    ids=[c.id for c in children],
                    documents=[c.text for c in children],
                    embeddings=child_embeddings,
                    metadatas=[c.metadata for c in children],
                )
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(f"Chroma upsert failed: {exc}") from exc

    def semantic_search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """ANN search over child chunks only."""
        try:
            results = self._children.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self._children.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            hits: list[dict[str, Any]] = []
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            for chunk_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
                hits.append({"id": chunk_id, "text": text, "distance": dist, **meta})
            return hits
        except Exception as exc:
            raise StoreError(f"Chroma semantic search failed: {exc}") from exc

    def fetch_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch parent chunks by ID list."""
        if not parent_ids:
            return []
        try:
            results = self._parents.get(
                ids=parent_ids,
                include=["documents", "metadatas"],
            )
            parents: list[dict[str, Any]] = []
            for chunk_id, text, meta in zip(
                results["ids"], results["documents"], results["metadatas"], strict=False
            ):
                parents.append({"id": chunk_id, "text": text, **meta})
            return parents
        except Exception as exc:
            raise StoreError(f"Chroma fetch_parents failed: {exc}") from exc

    def store_bm25_corpus(self, corpus: list[dict[str, Any]]) -> None:
        """Write BM25 corpus as JSON sidecar next to the Chroma persist dir."""
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._bm25_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
            logger.info("bm25_corpus_stored", path=str(self._bm25_path), count=len(corpus))
        except Exception as exc:
            raise StoreError(f"Failed to store BM25 corpus: {exc}") from exc

    def load_bm25_corpus(self) -> list[dict[str, Any]]:
        """Read BM25 corpus from JSON sidecar. Returns [] if not yet stored."""
        if not self._bm25_path.exists():
            return []
        try:
            data: list[dict[str, Any]] = json.loads(self._bm25_path.read_text(encoding="utf-8"))
            logger.info("bm25_corpus_loaded", path=str(self._bm25_path), count=len(data))
            return data
        except Exception as exc:
            raise StoreError(f"Failed to load BM25 corpus: {exc}") from exc

    def __enter__(self) -> ChromaStore:
        return self

    def __exit__(self, *args: object) -> None:
        pass  # Chroma PersistentClient auto-flushes; nothing to close explicitly


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
