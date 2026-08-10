# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Hierarchical ingestion pipeline with parent/child chunking strategy (ADR-001)
- Dual store backend: Chroma (default, zero-infra) and Weaviate (production) (ADR-002)
- Separate parent/child collections — no zero-vector pollution (ADR-003)
- BM25 corpus persisted via store interface — no split state on restart (ADR-004)
- Dropped `unstructured` dependency in favour of pypdf + BeautifulSoup (ADR-005)
- Hybrid retrieval: BM25 + semantic search with RRF fusion (K=60)
- Cross-encoder re-ranking via `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Citation-grounded generation enforced at prompt level (ADR-006)
- DeepEval evaluation harness with `warn_only` mode and empirical baseline guidance (ADR-007)
- Weekly scheduled DeepEval CI gate — not triggered on every PR
- Protocol-based interfaces with full dependency injection throughout
- `structlog` structured logging throughout
- `mypy` strict mode, `ruff` linting, pre-commit hooks
- CI `paths-ignore` — doc-only changes no longer trigger lint/typecheck/test jobs
- CI badges (CI status, DeepEval gate) added to README

### Changed
- Weaviate service image bumped to `1.27.0` (minimum version required by `weaviate-client>=4.5.0`)
- DeepEval CI gate switched from local Ollama judge to OpenAI `gpt-4o-mini`; Ollama removed
  as a CI service container (CPU inference too slow for 58+ questions within 60-min timeout)
- Fixed `config/eval_thresholds.yaml` key names: `context_precision` → `contextual_precision`,
  `context_recall` → `contextual_recall` to match DeepEval metric keys (two metrics were
  silently skipped during threshold checks)

### Performance
- `DeepEvalRunner._evaluate()`: all `(test_case, metric)` pairs now run concurrently via
  `ThreadPoolExecutor` with fresh metric instances per task — reduces 58-question eval from
  60+ minutes (timeout) to ~11 minutes
- `rag-platform-ingest`: document loading parallelised with `ThreadPoolExecutor`; all child chunks
  across all documents are embedded in a single `embed_batch` call instead of one call per
  document
- `WeaviateStore.fetch_parents()`: replaced per-ID round-trip loop with a single batch
  `Filter.by_id().contains_any()` query
- `WeaviateStore.store_bm25_corpus()`: replaced exception-based control flow with an
  explicit existence check before insert/replace
