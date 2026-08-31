"""RAGPipeline and build_pipeline() factory.

build_pipeline() is the only place in the codebase where concrete classes are instantiated.
All other code works against Protocols — swapping implementations means changing Settings,
not editing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from rag.config import Settings
from rag.interfaces.cache import SemanticCacheProtocol
from rag.interfaces.embedder import EmbedderProtocol
from rag.interfaces.generator import CitedAnswer, GeneratorProtocol
from rag.interfaces.retriever import RetrievedChunk, RetrieverProtocol

logger = structlog.get_logger(__name__)


def _default_tracer() -> Any:
    """Return a no-op tracer. Used as a safe default for RAGPipeline.tracer."""
    from rag.tracing import _NoOpTracer  # noqa: PLC0415

    return _NoOpTracer()


@dataclass
class RAGPipeline:
    """Assembled query pipeline. Holds a retriever, generator, and optional cache."""

    retriever: RetrieverProtocol
    generator: GeneratorProtocol
    embedder: EmbedderProtocol
    tracer: Any = field(default_factory=_default_tracer)
    cache: SemanticCacheProtocol | None = None

    def query(self, question: str) -> CitedAnswer:
        """Run the full RAG pipeline: retrieve context then generate a cited answer.

        When a semantic cache is configured, embeds the query and checks the cache
        first. A cache hit skips retrieval and generation entirely. A miss runs the
        full pipeline and stores the result for future hits.

        Wraps retrieval and generation in Langfuse spans when tracing is enabled.
        """
        if self.cache is not None:
            query_embedding = self.embedder.embed(question)
            cached = self.cache.get(query_embedding)
            if cached is not None:
                logger.info("semantic_cache_hit", query_length=len(question))
                return cached
        else:
            query_embedding = None

        with self.tracer.trace("query", metadata={"query_length": len(question)}) as span:
            with span.span("retrieval") as s:
                chunks = self.retriever.retrieve(question)
                s.update(output={"chunks": len(chunks)})
            with span.span("generation") as s:
                answer = self.generator.generate(question, chunks)
                s.update(output={"citations": len(answer.citations)})
        self.tracer.flush()

        if self.cache is not None and query_embedding is not None:
            self.cache.set(query_embedding, answer)

        return answer

    def search(
        self,
        query: str,
        *,
        trace_id: str | None = None,
        workflow_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return retrieval evidence without invoking answer generation or semantic caching."""
        metadata: dict[str, str | int] = {"query_length": len(query)}
        if trace_id is not None:
            metadata["trace_id"] = trace_id
        if workflow_id is not None:
            metadata["workflow_id"] = workflow_id
        with (
            self.tracer.trace("search", metadata=metadata) as span,
            span.span("retrieval") as retrieval_span,
        ):
            chunks = self.retriever.retrieve(query)
            retrieval_span.update(output={"chunks": len(chunks)})
        self.tracer.flush()
        return chunks

    def pipeline_fn(self, question: str) -> dict[str, Any]:
        """DeepEval-compatible interface: returns actual_output and retrieval_context keys."""
        result = self.query(question)
        return {
            "actual_output": result.answer,
            "retrieval_context": [c.text for c in result.raw_context],
        }


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Factory function. The only place concrete implementations are instantiated.

    All downstream code works against Protocols — swap implementations by
    changing Settings env vars, not by editing call sites.
    """
    from rag.cache.factory import get_cache  # noqa: PLC0415
    from rag.generation.generator import CitationGroundedGenerator  # noqa: PLC0415
    from rag.ingestion.embedder import get_embedder  # noqa: PLC0415
    from rag.retrieval.hybrid import HybridRetriever  # noqa: PLC0415
    from rag.store.factory import get_store  # noqa: PLC0415
    from rag.tracing import get_tracer  # noqa: PLC0415

    store = get_store(settings)
    embedder = get_embedder(settings)
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        reranker_model=settings.reranker_model,
        top_k_dense=settings.top_k_dense,
        top_k_rerank=settings.top_k_rerank,
    )
    generator = CitationGroundedGenerator(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key.get_secret_value(),
        max_tokens=settings.llm_max_tokens,
    )
    tracer = get_tracer(settings)
    cache = get_cache(settings)
    if tracer.enabled:
        logger.info("tracing_enabled")
    if cache is not None:
        logger.info("semantic_cache_enabled", backend=settings.cache_backend)
    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        embedder=embedder,
        tracer=tracer,
        cache=cache,
    )
