# RAG Platform

[![CI](https://github.com/adedaramola/rag-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/adedaramola/rag-platform/actions/workflows/ci.yml)
[![DeepEval](https://github.com/adedaramola/rag-platform/actions/workflows/eval.yml/badge.svg)](https://github.com/adedaramola/rag-platform/actions/workflows/eval.yml)

RAG Platform searches a document collection and returns information with source
references. It supports both a standalone question-answering application and a
retrieval service for other applications.

I built it to give applications a reusable document search service, with source
references and separate checks for retrieval quality and generated answers.

In [OpsDesk](https://github.com/adedaramola/ops-platform), this service searches only
approved documents. It returns excerpts and references to the background Agent; the
[model gateway](https://github.com/adedaramola/multi-llm-platform) prepares the final
draft, which remains subject to staff review.

## Engineering decisions

- **Combine keyword and meaning-based search.** The service can find exact terms as well as related wording, then rerank the results.
- **Retain the surrounding context.** It searches smaller passages and returns their broader document context where needed.
- **Return traceable sources.** Results include document and page references when available; the OpsDesk endpoint checks an approved-source list.
- **Measure search and answer quality separately.** Finding the right document does not establish that a generated answer is correct.

## Measured results and limits

The August 15, 2026 AWS benchmark used two PDFs and 58 queries, including 55 answerable
queries. After reranking, the expected document was the first result for all 55
answerable queries. The labels identify documents, so this result does not establish
passage-level relevance or answer accuracy.

The same benchmark exposed useful limitations: all three no-answer probes still
returned results, and CPU reranking accounted for most retrieval time. A separate
repeat-query test measured a warm median response of about 245 ms against about 9 seconds
for cold requests. That comparison measures cache reuse on repeated questions.

The September 2026 OpsDesk integration used a different embedding configuration and
approved source set. The August benchmark is a dated measurement, not a score for every
deployment. See [benchmark methods and results](docs/benchmarking.md),
[answer evaluation](docs/evaluation.md), and the
[connected deployment record](https://github.com/adedaramola/ops-platform/blob/main/VALIDATION.md).

For local setup and deployment, start with [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## How search works

```text
Documents → Extract and split text → Store passages and search indexes
                                             |
Question → Keyword + meaning-based search → Rerank → Retrieve context
                                                      |
                                ┌─────────────────────┴───────────────────┐
                                ▼                                         ▼
                      /v1/search: excerpts                    /query: generated answer
                      and source references                   with source references
```

---

## Stack

| Layer | Local default | Alternative / deployment option | Purpose |
|---|---|---|---|
| Embedding | OpenAI text-embedding-3-small | BAAI/bge-large-en-v1.5 | Zero-infra default; BGE opt-in via `[local-embed]` |
| Vector store | Chroma (in-process) | Weaviate 1.27.0 | Local development or a separate vector service |
| Retrieval fusion | BM25 + ANN → RRF (K=60) | Same | Parameter-free, robust to miscalibrated retrievers |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Same | Reorders candidates; measured latency depends on hardware |
| LLM | claude-sonnet-4-6 | Same | Citation-enforcing prompt, structured output |
| Tracing | Langfuse (opt-in no-op) | Langfuse cloud | Set `RAG_PLATFORM_LANGFUSE_SECRET_KEY` to enable; system runs identically without it |
| API | FastAPI + uvicorn | Same | `/query` for answers, `/v1/search` for retrieval, `/health` for liveness |
| UI | Streamlit | Same | Chat interface with citation rendering |
| Eval | DeepEval + Ollama judge | Same | Pytest-native, zero API cost for local judge |

---

## Local setup

Follow [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md), RAG-01 through RAG-06, to
install the service, configure embeddings, ingest an approved document, and check the
API. RAG-07 covers evaluation and RAG-08 through RAG-10 cover AWS deployment.

Once configured, `make api` starts the API on port 8000 and `make ui` starts the
optional chat interface on port 8501. The examples below describe the application API.

### Query via Python

```python
from rag.pipeline import build_pipeline
from rag.config import get_settings

pipeline = build_pipeline(get_settings())
result = pipeline.query("What is the purpose of multi-head attention?")
print(result.answer)
for c in result.citations:
    print(f"  [{c.index}] {c.source} p.{c.page}")
```

### Query via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the purpose of multi-head attention?"}'
```

### Agent retrieval API

OpsDesk uses the versioned retrieval-only endpoint and performs final generation through its
separate multi-LLM gateway. Configure a service credential plus an explicit source allowlist:

```dotenv
RAG_PLATFORM_API_KEY=replace-with-a-scoped-service-key
RAG_PLATFORM_APPROVED_SOURCE_IDS=["vpn-runbook"]
```

An authenticated request returns bounded evidence rather than a generated answer:

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_PLATFORM_API_KEY" \
  -d '{"query":"VPN connection issue","source_ids":["vpn-runbook"],"max_chunks":5}'
```

Each response contains a request-local `query_id` and chunks with `citation_id`, `source_id`,
optional `page`, and a bounded approved excerpt. Requests for sources outside the server allowlist
are rejected. Raw queries and excerpts are excluded from application logs and traces.

The AWS Terraform stack terminates TLS at `rag_domain_name`, generates the scoped service
credential, stores it in Secrets Manager, and exposes only its ARN for the OpsDesk Agent role. The
same stack injects `approved_source_ids` into the API runtime; the raw credential is never an
output.

---

## Optional Backends

### Local embeddings (BGE — no API key required)

```bash
pip install -e ".[local-embed]"

# .env
RAG_PLATFORM_EMBED_BACKEND=local
```

Requires ~2GB RAM. Uses `BAAI/bge-large-en-v1.5` (1024-dim).

### Chroma development-only security boundary

The default Chroma backend uses an in-process `PersistentClient` for local development and CI. Do
not expose Chroma's HTTP server or use its multi-tenant authorization in this project: the current
latest package has unresolved 2026 authorization and code-injection advisories with no patched
PyPI release reported by the dependency audit. Production uses the private Weaviate backend. Keep
Chroma bound to the local process and re-evaluate the advisory status before changing that boundary.

### Weaviate (production vector store)

```bash
make docker-up   # starts Weaviate on localhost:8080

# .env
RAG_PLATFORM_STORE_BACKEND=weaviate
```

Re-ingest after switching backends — vectors are not portable between Chroma and Weaviate.

### OpenAI eval judge

```bash
# .env
RAG_PLATFORM_EVAL_JUDGE_BACKEND=openai
RAG_PLATFORM_OPENAI_API_KEY=sk-proj-...
```

Default judge is local Ollama (`llama3.1:8b`) at zero API cost.

### Langfuse tracing (opt-in)

```bash
pip install -e ".[observability]"

# .env
RAG_PLATFORM_LANGFUSE_PUBLIC_KEY=pk-lf-...
RAG_PLATFORM_LANGFUSE_SECRET_KEY=sk-lf-...
# Optional — defaults to Langfuse cloud
RAG_PLATFORM_LANGFUSE_HOST=https://cloud.langfuse.com
```

When keys are absent the system runs identically — all trace calls are no-ops.
Instruments retrieval (dense hits, RRF scores, rerank), generation (prompt tokens,
latency), and end-to-end query spans.

---

## AWS deployment

This repository owns its AWS deployment: an API service on EC2, a separate private
Weaviate service, document storage in S3, and a load balancer. RAG runs independently
of the OpsDesk EKS cluster. The connected deployment uses HTTPS and a scoped service
credential stored in Secrets Manager.

Use [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for setup and verification, and
[terraform/README.md](terraform/README.md) for infrastructure details. For the complete
four-service deployment order, start with the
[OpsDesk portfolio guide](https://github.com/adedaramola/ops-platform/blob/main/PORTFOLIO_IMPLEMENTATION_GUIDE.md).

## Development Workflow

| Command | Description |
|---|---|
| `make install` | Install all dev dependencies + pre-commit hooks |
| `make lint` | ruff check + format check |
| `make format` | ruff format + autofix |
| `make typecheck` | mypy strict mode |
| `make test-unit` | Unit tests (no network, no API keys) |
| `make test-integration` | Integration tests (Chroma: no Docker; Weaviate: skipped without Docker) |
| `make eval` | DeepEval gate (requires Ollama + golden dataset) |
| `make benchmark` | Dense/BM25/hybrid/rerank quality and latency report |
| `make load-test URL=...` | Deployed concurrency, E2E latency, and cache report |
| `make ingest SOURCE=path` | Ingest a document or directory |
| `make api` | Start FastAPI on port 8000 |
| `make ui` | Start Streamlit UI on port 8501 |
| `make docker-up` | Start Weaviate service container |
| `make ci` | Full CI: lint + typecheck + unit tests |
| `make clean` | Remove build artefacts and caches |

Metric definitions, relevance-label caveats, and the current measured baseline are in
[docs/benchmarking.md](docs/benchmarking.md).

---

## Project Layout

```
rag-platform/
├── src/rag/                  # All package code (PEP 517 src layout)
│   ├── interfaces/           # typing.Protocol definitions — the public contract
│   ├── ingestion/            # Loaders, chunker, embedders
│   ├── store/                # Chroma, Weaviate, factory
│   ├── retrieval/            # HybridRetriever (BM25 + dense + RRF + rerank)
│   ├── generation/           # CitationGroundedGenerator
│   ├── evaluation/           # DeepEval harness + Ollama judge adapter
│   ├── api/                  # FastAPI app (main.py)
│   ├── scripts/              # CLI entry points (rag-platform-ingest)
│   ├── config.py             # Pydantic Settings — one object rules all config
│   ├── exceptions.py         # Domain exception hierarchy
│   └── pipeline.py           # RAGPipeline + build_pipeline() factory
├── tests/
│   ├── unit/                 # No network or Docker dependencies
│   ├── integration/          # Chroma: no Docker; Weaviate: skipped without Docker
│   └── e2e/                  # DeepEval gate (weekly CI schedule)
├── terraform/                # AWS infrastructure (EC2, ALB, Weaviate, S3)
├── docs/                     # ADRs and evaluation guide
├── config/                   # eval_thresholds.yaml
├── app.py                    # Streamlit chat UI
└── data/golden/              # Golden QA pairs for DeepEval evaluation
```

---

## Evaluation

The DeepEval workflow is configured for weekly and manual runs. CI uses an OpenAI
judge and currently reports threshold failures as warnings; a successful workflow
does not necessarily mean all quality thresholds passed. Local evaluation defaults
to an Ollama judge. See [docs/evaluation.md](docs/evaluation.md) for results and interpretation.

```bash
# Pull the judge model
make ollama-pull

# Run locally
make eval
```

---

## ADR Index

| # | Decision | Status |
|---|---|---|
| [ADR-001](docs/architecture.md#adr-001-hierarchical-chunking-over-flat-chunking) | Hierarchical chunking (parent/child) | Accepted |
| [ADR-002](docs/architecture.md#adr-002-dual-store-backend-chroma-default-weaviate-production) | Dual store backend (Chroma + Weaviate) | Accepted |
| [ADR-003](docs/architecture.md#adr-003-separate-parent-and-child-collections) | Separate parent/child collections | Accepted |
| [ADR-004](docs/architecture.md#adr-004-bm25-corpus-co-located-with-vector-store) | BM25 corpus co-located with vector store | Accepted |
| [ADR-005](docs/architecture.md#adr-005-drop-unstructured-as-a-dependency) | Drop `unstructured` dependency | Accepted |
| [ADR-006](docs/architecture.md#adr-006-citation-grounding-enforced-at-generation-time) | Citation grounding enforced at generation time | Accepted |
| [ADR-007](docs/architecture.md#adr-007-deepeval-over-ragas-with-a-local-ollama-judge) | DeepEval over RAGAS with local Ollama judge | Accepted |

---

## License

MIT
