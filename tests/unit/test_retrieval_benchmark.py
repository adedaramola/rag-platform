"""Unit tests for deterministic retrieval benchmark calculations."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.evaluation.retrieval_benchmark import RetrievalBenchmarkRunner
from rag.interfaces.retriever import CorpusStats, RetrievalDiagnostics, RetrievedChunk


def _hit(chunk_id: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=f"text for {chunk_id}",
        source=source,
        page=1,
        score=1.0,
    )


class _DiagnosticRetriever:
    """Small deterministic retriever that exposes deliberately different rankings."""

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self.retrieve_with_diagnostics(query).hybrid_post_rerank

    def retrieve_with_diagnostics(self, query: str) -> RetrievalDiagnostics:
        relevant = _hit("relevant", "a.pdf")
        irrelevant = _hit("irrelevant", "b.pdf")
        return RetrievalDiagnostics(
            dense=[irrelevant, relevant],
            bm25=[relevant, irrelevant],
            hybrid_pre_rerank=[irrelevant, relevant],
            hybrid_post_rerank=[relevant, irrelevant],
            timings_ms={"retrieval_total": 10.0, "reranking": 2.0},
        )

    def corpus_stats(self) -> CorpusStats:
        return CorpusStats(
            chunks=2,
            unique_chunk_ids=2,
            documents=2,
            source_names=["a.pdf", "b.pdf"],
            pages=2,
            duplicate_text_chunks=0,
            empty_text_chunks=0,
            missing_source_chunks=0,
            missing_page_chunks=0,
            missing_parent_id_chunks=0,
        )

    def build_index(self, chunks: list[dict[str, object]]) -> None:
        return None


def test_benchmark_compares_ranked_stages_and_excludes_no_answer_rows(tmp_path: Path) -> None:
    """Ranking metrics use positives while no-answer behavior remains separate."""
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"question":"positive","ground_truth":"a","expected_sources":["a.pdf"]}\n'
        '{"question":"negative","ground_truth":"none","expected_sources":[]}\n',
        encoding="utf-8",
    )

    report = RetrievalBenchmarkRunner(_DiagnosticRetriever(), ks=(1, 2)).run(golden)

    assert report["dataset"] == {
        "query_count": 2,
        "positive_query_count": 1,
        "no_answer_query_count": 1,
    }
    assert report["methods"]["dense"]["recall_at_1"] == 0.0
    assert report["methods"]["dense"]["mrr"] == pytest.approx(0.5)
    assert report["methods"]["bm25"]["precision_at_1"] == 1.0
    assert report["methods"]["hybrid_pre_rerank"]["ndcg_at_1"] == 0.0
    assert report["methods"]["hybrid_post_rerank"]["ndcg_at_1"] == 1.0
    assert report["methods"]["dense"]["no_answer_nonempty_rate"] == 1.0
    assert report["latency_ms"]["reranking"]["p95"] == 2.0


def test_benchmark_flags_missing_expected_corpus_source(tmp_path: Path) -> None:
    """Golden/corpus source mismatches are surfaced as data-quality defects."""
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"question":"q","ground_truth":"a","expected_sources":["missing.pdf"]}\n',
        encoding="utf-8",
    )

    report = RetrievalBenchmarkRunner(_DiagnosticRetriever()).run(golden)

    assert report["data_quality"]["missing_expected_sources"] == ["missing.pdf"]
    assert report["data_quality"]["questions_with_missing_expected_sources"] == 1
    assert report["data_quality"]["source_only_positive_labels"] == 1
    assert report["data_quality"]["passage_labeled_positive_queries"] == 0


def test_write_report_round_trips_json(tmp_path: Path) -> None:
    """The writer creates parent directories and emits valid JSON."""
    output = tmp_path / "nested" / "report.json"
    RetrievalBenchmarkRunner.write_report({"ok": True}, output)
    assert output.read_text(encoding="utf-8") == '{\n  "ok": true\n}'
