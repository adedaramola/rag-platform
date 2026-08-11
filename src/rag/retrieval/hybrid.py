"""Hybrid BM25 + dense retriever with RRF fusion and cross-encoder re-ranking.

BM25 index lifecycle:
  After ingestion: retriever.build_index(corpus) → store.store_bm25_corpus(corpus)
  On startup:      corpus = store.load_bm25_corpus() → build_index(corpus)
  Vector index and BM25 corpus share the same store lifecycle — restarting the
  process restores both automatically with no manual rebuild step.

RRF reference: Cormack, Clarke, Buettcher (2009) — "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods."
"""

from __future__ import annotations

from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from rag.exceptions import RetrievalError
from rag.interfaces.embedder import EmbedderProtocol
from rag.interfaces.retriever import RetrievedChunk
from rag.interfaces.store import DocumentStoreProtocol

logger = structlog.get_logger(__name__)

# Standard constant from Cormack et al. (2009).
# K=60 dampens the impact of top-ranked results, making fusion robust
# to cases where one retriever is confidently wrong.
RRF_K: int = 60


class HybridRetriever:
    """BM25 + dense hybrid retriever with RRF fusion and cross-encoder re-ranking.

    Implements RetrieverProtocol. All dependencies injected as Protocols.
    """

    def __init__(
        self,
        store: DocumentStoreProtocol,
        embedder: EmbedderProtocol,
        reranker_model: str,
        top_k_dense: int = 20,
        top_k_rerank: int = 5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k_dense = top_k_dense
        self._top_k_rerank = top_k_rerank
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict[str, Any]] = []

        # Load cross-encoder once at init — not per query
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self._reranker = CrossEncoder(reranker_model)
        logger.info("cross_encoder_loaded", model=reranker_model)

        # Restore BM25 index from the store on startup
        self.build_index(store.load_bm25_corpus())

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build or rebuild the BM25Okapi index from a list of chunk dicts."""
        self._corpus = chunks
        if not chunks:
            self._bm25 = None
            return
        tokenised = [c.get("text", "").lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("bm25_index_built", num_docs=len(chunks))

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Run the full hybrid retrieval pipeline.

        Steps:
          1. Dense:           ANN search via the vector store
          2. Sparse:          BM25 top-20 from in-memory index
          3. RRF fusion:      reciprocal rank fusion of both result lists
          4. Parent expansion: swap child text for parent context, deduplicate
          5. Cross-encoder:   re-rank fused candidates by query relevance
        """
        log = logger.bind(query=query[:80])

        # Step 1 — dense retrieval
        try:
            query_vector = self._embedder.embed(query)
            dense_hits = self._store.semantic_search(query_vector, top_k=self._top_k_dense)
            log.debug("dense_hits", count=len(dense_hits))
        except Exception as exc:
            raise RetrievalError(query=query, step="dense") from exc

        # Step 2 — sparse BM25 retrieval
        try:
            sparse_hits = self._bm25_search(query, top_k=20)
            log.debug("sparse_hits", count=len(sparse_hits))
        except Exception as exc:
            raise RetrievalError(query=query, step="sparse") from exc

        # Step 3 — RRF fusion
        fused = _rrf_fuse(
            [str(h["id"]) for h in dense_hits],
            [str(h["id"]) for h in sparse_hits],
        )
        # Merge metadata from both hit lists into a single lookup
        id_to_meta: dict[str, dict[str, Any]] = {}
        for hit in dense_hits + sparse_hits:
            id_to_meta.setdefault(hit["id"], hit)

        # Step 4 — parent expansion (swap child text for parent context)
        try:
            candidates = self._expand_to_parents(fused, id_to_meta)
        except Exception as exc:
            raise RetrievalError(query=query, step="parent_expansion") from exc

        # Step 5 — cross-encoder re-ranking
        try:
            reranked = self._rerank(query, candidates)
        except Exception as exc:
            raise RetrievalError(query=query, step="rerank") from exc

        log.info("retrieval_complete", returned=len(reranked))
        return reranked[: self._top_k_rerank]

    def _bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return top-k BM25 hits. Returns [] gracefully if index is empty."""
        if self._bm25 is None or not self._corpus:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self._corpus[i], "bm25_score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    def _expand_to_parents(
        self,
        fused_ids: list[str],
        id_to_meta: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve each fused child id to its parent passage.

        Deduplicates on parent_id so the same context is never sent twice.
        The child chunk text is preserved for retrieval/eval, while the parent
        passage is carried separately for generation context.
        """
        parent_ids: list[str] = []
        child_to_parent: dict[str, str] = {}
        seen_parents: set[str] = set()

        for chunk_id in fused_ids:
            meta = id_to_meta.get(chunk_id, {})
            pid = str(meta.get("parent_id", ""))
            if pid and pid not in seen_parents:
                parent_ids.append(pid)
                seen_parents.add(pid)
            child_to_parent[chunk_id] = pid

        parents = self._store.fetch_parents(parent_ids)
        pid_to_parent: dict[str, dict[str, Any]] = {str(p["id"]): p for p in parents}

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk_id in fused_ids:
            meta = id_to_meta.get(chunk_id, {})
            pid = child_to_parent.get(chunk_id, "")
            parent = pid_to_parent.get(pid)
            if parent and pid not in seen:
                seen.add(pid)
                candidates.append(
                    {
                        **meta,
                        "id": pid,
                        "parent_text": parent.get("text", ""),
                        "source": parent.get("source", meta.get("source", "")),
                        "page": parent.get("page", meta.get("page")),
                    }
                )
            elif chunk_id not in seen:
                seen.add(chunk_id)
                candidates.append(meta)
        return candidates

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[RetrievedChunk]:
        """Score candidates with the cross-encoder and return sorted RetrievedChunks."""
        if not candidates:
            return []
        pairs = [(query, c.get("parent_text") or c.get("text", "")) for c in candidates]
        scores: list[float] = self._reranker.predict(pairs).tolist()

        ranked = sorted(zip(candidates, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(
                id=str(c.get("id", "")),
                text=str(c.get("text", "")),
                source=str(c.get("source", "")),
                page=int(c["page"]) if c.get("page") is not None else None,
                score=score,
                parent_text=str(c["parent_text"]) if c.get("parent_text") is not None else None,
            )
            for c, score in ranked
        ]


def _rrf_fuse(dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    """Reciprocal Rank Fusion of two ranked ID lists.

    score[id] += 1 / (RRF_K + rank)  for each list the id appears in.
    Returns IDs sorted by descending fused score. Deduplicates automatically.
    """
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, chunk_id in enumerate(sparse_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda k: scores[k], reverse=True)
