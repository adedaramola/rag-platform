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

import time
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from rag.exceptions import RetrievalError
from rag.interfaces.embedder import EmbedderProtocol
from rag.interfaces.retriever import CorpusStats, RetrievalDiagnostics, RetrievedChunk
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
        return self.retrieve_with_diagnostics(query).hybrid_post_rerank

    def retrieve_with_diagnostics(self, query: str) -> RetrievalDiagnostics:
        """Run retrieval once and retain every ranked stage and stage latency."""
        log = logger.bind(query=query[:80])
        started = time.perf_counter()
        timings: dict[str, float] = {}

        # Step 1 — dense retrieval
        try:
            step_started = time.perf_counter()
            query_vector = self._embedder.embed(query)
            timings["embedding"] = _elapsed_ms(step_started)
            step_started = time.perf_counter()
            dense_hits = self._store.semantic_search(query_vector, top_k=self._top_k_dense)
            timings["dense_search"] = _elapsed_ms(step_started)
            log.debug("dense_hits", count=len(dense_hits))
        except Exception as exc:
            raise RetrievalError(query=query, step="dense") from exc

        # Step 2 — sparse BM25 retrieval
        try:
            step_started = time.perf_counter()
            sparse_hits = self._bm25_search(query, top_k=20)
            timings["bm25_search"] = _elapsed_ms(step_started)
            log.debug("sparse_hits", count=len(sparse_hits))
        except Exception as exc:
            raise RetrievalError(query=query, step="sparse") from exc

        # Step 3 — RRF fusion
        step_started = time.perf_counter()
        fused_scores = _rrf_score(
            [str(h["id"]) for h in dense_hits],
            [str(h["id"]) for h in sparse_hits],
        )
        fused = list(fused_scores)
        # Merge metadata from both hit lists into a single lookup
        id_to_meta: dict[str, dict[str, Any]] = {}
        for hit in dense_hits + sparse_hits:
            id_to_meta.setdefault(str(hit["id"]), hit)
        for chunk_id, score in fused_scores.items():
            id_to_meta[chunk_id]["rrf_score"] = score
        timings["fusion"] = _elapsed_ms(step_started)

        # Step 4 — parent expansion (swap child text for parent context)
        try:
            step_started = time.perf_counter()
            candidates = self._expand_to_parents(fused, id_to_meta)
            timings["parent_expansion"] = _elapsed_ms(step_started)
        except Exception as exc:
            raise RetrievalError(query=query, step="parent_expansion") from exc

        # Step 5 — cross-encoder re-ranking
        try:
            step_started = time.perf_counter()
            reranked = self._rerank(query, candidates)
            timings["reranking"] = _elapsed_ms(step_started)
        except Exception as exc:
            raise RetrievalError(query=query, step="rerank") from exc

        post_rerank = reranked[: self._top_k_rerank]
        timings["retrieval_total"] = _elapsed_ms(started)
        dense_ranked = _raw_hits_to_chunks(dense_hits, "distance", invert_score=True)
        bm25_ranked = _raw_hits_to_chunks(sparse_hits, "bm25_score")
        pre_rerank = _raw_hits_to_chunks(candidates, "rrf_score")
        log.info(
            "retrieval_complete",
            returned=len(post_rerank),
            latency_ms=round(timings["retrieval_total"], 2),
        )
        return RetrievalDiagnostics(
            dense=dense_ranked,
            bm25=bm25_ranked,
            hybrid_pre_rerank=pre_rerank,
            hybrid_post_rerank=post_rerank,
            timings_ms=timings,
        )

    def corpus_stats(self) -> CorpusStats:
        """Measure indexed child scale and common metadata/data-quality defects."""
        ids = [str(chunk.get("id", "")) for chunk in self._corpus]
        texts = [str(chunk.get("text", "")).strip() for chunk in self._corpus]
        sources = {
            Path(str(chunk.get("source", ""))).name for chunk in self._corpus if chunk.get("source")
        }
        pages = {
            (Path(str(chunk.get("source", ""))).name, chunk.get("page"))
            for chunk in self._corpus
            if chunk.get("source") and chunk.get("page") is not None
        }
        nonempty_texts = [text for text in texts if text]
        return CorpusStats(
            chunks=len(self._corpus),
            unique_chunk_ids=len(set(ids)),
            documents=len(sources),
            source_names=sorted(sources),
            pages=len(pages),
            duplicate_text_chunks=len(nonempty_texts) - len(set(nonempty_texts)),
            empty_text_chunks=sum(not text for text in texts),
            missing_source_chunks=sum(not chunk.get("source") for chunk in self._corpus),
            missing_page_chunks=sum(chunk.get("page") is None for chunk in self._corpus),
            missing_parent_id_chunks=sum(not chunk.get("parent_id") for chunk in self._corpus),
        )

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
    return list(_rrf_score(dense_ids, sparse_ids))


def _rrf_score(dense_ids: list[str], sparse_ids: list[str]) -> dict[str, float]:
    """Return RRF scores in descending rank order."""
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, chunk_id in enumerate(sparse_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _raw_hits_to_chunks(
    hits: list[dict[str, Any]],
    score_field: str,
    *,
    invert_score: bool = False,
) -> list[RetrievedChunk]:
    """Convert store/candidate dictionaries to the public ranked chunk type."""
    chunks: list[RetrievedChunk] = []
    for hit in hits:
        raw_score = float(hit.get(score_field, 0.0))
        score = 1.0 - raw_score if invert_score else raw_score
        chunks.append(
            RetrievedChunk(
                id=str(hit.get("id", "")),
                text=str(hit.get("text", "")),
                source=str(hit.get("source", "")),
                page=int(hit["page"]) if hit.get("page") is not None else None,
                score=score,
                parent_text=(
                    str(hit["parent_text"]) if hit.get("parent_text") is not None else None
                ),
            )
        )
    return chunks


def _elapsed_ms(started: float) -> float:
    """Convert a perf-counter interval to milliseconds."""
    return (time.perf_counter() - started) * 1000.0
