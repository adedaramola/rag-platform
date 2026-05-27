"""Unit tests for RAGPipeline query and pipeline_fn interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from rag.config import Settings
from rag.interfaces.generator import CitationRef, CitedAnswer
from rag.interfaces.retriever import RetrievedChunk
from rag.pipeline import RAGPipeline


def _make_chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(id=str(i), text=f"chunk {i}", source="doc.pdf", page=i, score=0.9)
        for i in range(1, n + 1)
    ]


def test_pipeline_query_returns_cited_answer(mock_pipeline: RAGPipeline) -> None:
    """query() returns a CitedAnswer with a non-empty answer."""
    from tests.conftest import MockStore

    store = mock_pipeline.retriever._store
    assert isinstance(store, MockStore)

    # Seed the store with chunks so retrieval returns results
    from rag.interfaces.store import Chunk

    embedder = mock_pipeline.retriever._embedder
    parent = Chunk(
        id="p1",
        text="parent passage",
        metadata={"source": "doc.pdf", "page": 0},
        parent_id=None,
        token_count=50,
    )
    child = Chunk(
        id="c1",
        text="child chunk text about RAG",
        metadata={"source": "doc.pdf", "page": 0},
        parent_id="p1",
        token_count=20,
    )
    vec = embedder.embed(child.text)
    store.upsert_chunks([parent])
    store.upsert_chunks([child], embeddings=[vec])
    store.store_bm25_corpus([{"id": "c1", "text": child.text, "source": "doc.pdf", "page": 0}])
    mock_pipeline.retriever.build_index(store.load_bm25_corpus())

    result = mock_pipeline.query("What is RAG?")
    assert result.answer
    assert len(result.citations) >= 1
    assert len(result.raw_context) >= 1


def test_pipeline_fn_returns_deepeval_compatible_dict(mock_pipeline: RAGPipeline) -> None:
    """pipeline_fn() returns a dict with 'actual_output' and 'retrieval_context' keys."""
    from rag.interfaces.store import Chunk

    store = mock_pipeline.retriever._store
    embedder = mock_pipeline.retriever._embedder

    parent = Chunk(
        id="p2",
        text="parent text",
        metadata={"source": "s.pdf", "page": 1},
        parent_id=None,
        token_count=50,
    )
    child = Chunk(
        id="c2",
        text="child text about retrieval",
        metadata={"source": "s.pdf", "page": 1},
        parent_id="p2",
        token_count=20,
    )
    vec = embedder.embed(child.text)
    store.upsert_chunks([parent])
    store.upsert_chunks([child], embeddings=[vec])
    store.store_bm25_corpus([{"id": "c2", "text": child.text, "source": "s.pdf", "page": 1}])
    mock_pipeline.retriever.build_index(store.load_bm25_corpus())

    out = mock_pipeline.pipeline_fn("What is retrieval?")
    assert "actual_output" in out
    assert "retrieval_context" in out
    assert isinstance(out["retrieval_context"], list)


def test_pipeline_query_cache_hit_skips_retrieval_and_generation() -> None:
    """A semantic cache hit should return immediately without downstream calls."""
    cached_answer = CitedAnswer(
        answer="Cached answer [src 1]",
        citations=[CitationRef(index=1, source="cached.pdf", page=1)],
        raw_context=[],
    )

    mock_cache = MagicMock()
    mock_cache.get.return_value = cached_answer
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_tracer = MagicMock()

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        embedder=mock_embedder,
        tracer=mock_tracer,
        cache=mock_cache,
    )

    result = pipeline.query("cached question")

    assert result is cached_answer
    mock_embedder.embed.assert_called_once_with("cached question")
    mock_cache.get.assert_called_once_with([0.1, 0.2, 0.3])
    mock_retriever.retrieve.assert_not_called()
    mock_generator.generate.assert_not_called()
    mock_cache.set.assert_not_called()
    mock_tracer.trace.assert_not_called()
    mock_tracer.flush.assert_not_called()


def test_pipeline_query_cache_miss_stores_result() -> None:
    """A cache miss should run the pipeline and persist the generated answer."""
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.4, 0.5, 0.6]
    chunks = _make_chunks(2)
    answer = CitedAnswer(
        answer="Fresh answer [src 1]",
        citations=[CitationRef(index=1, source="doc.pdf", page=1)],
        raw_context=chunks,
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = chunks
    mock_generator = MagicMock()
    mock_generator.generate.return_value = answer

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        embedder=mock_embedder,
        cache=mock_cache,
    )

    result = pipeline.query("fresh question")

    assert result is answer
    mock_cache.get.assert_called_once_with([0.4, 0.5, 0.6])
    mock_cache.set.assert_called_once_with([0.4, 0.5, 0.6], answer)


def test_build_pipeline_wires_concrete_dependencies() -> None:
    """build_pipeline should assemble and return a fully wired RAGPipeline."""
    settings = Settings(
        anthropic_api_key=SecretStr("anthropic-test"),
        openai_api_key=SecretStr("openai-test"),
        cache_backend="memory",
    )
    mock_store = object()
    mock_embedder = object()
    mock_retriever = object()
    mock_generator = object()
    mock_tracer = MagicMock()
    mock_tracer.enabled = True
    mock_cache = object()

    with (
        patch("rag.store.factory.get_store", return_value=mock_store) as get_store,
        patch("rag.ingestion.embedder.get_embedder", return_value=mock_embedder) as get_embedder,
        patch("rag.retrieval.hybrid.HybridRetriever", return_value=mock_retriever) as retriever_cls,
        patch(
            "rag.generation.generator.CitationGroundedGenerator",
            return_value=mock_generator,
        ) as generator_cls,
        patch("rag.tracing.get_tracer", return_value=mock_tracer) as get_tracer,
        patch("rag.cache.factory.get_cache", return_value=mock_cache) as get_cache,
    ):
        pipeline = RAGPipeline.__module__
        del pipeline
        from rag.pipeline import build_pipeline

        built = build_pipeline(settings)

    assert isinstance(built, RAGPipeline)
    assert built.retriever is mock_retriever
    assert built.generator is mock_generator
    assert built.embedder is mock_embedder
    assert built.tracer is mock_tracer
    assert built.cache is mock_cache
    get_store.assert_called_once_with(settings)
    get_embedder.assert_called_once_with(settings)
    retriever_cls.assert_called_once_with(
        store=mock_store,
        embedder=mock_embedder,
        reranker_model=settings.reranker_model,
        top_k_dense=settings.top_k_dense,
        top_k_rerank=settings.top_k_rerank,
    )
    generator_cls.assert_called_once_with(
        model=settings.llm_model,
        api_key="anthropic-test",
    )
    get_tracer.assert_called_once_with(settings)
    get_cache.assert_called_once_with(settings)
