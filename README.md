# RAG Platform

[![CI](https://github.com/adedaramola/rag-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/adedaramola/rag-platform/actions/workflows/ci.yml)
[![DeepEval](https://github.com/adedaramola/rag-platform/actions/workflows/eval.yml/badge.svg)](https://github.com/adedaramola/rag-platform/actions/workflows/eval.yml)

A production-grade domain-specific RAG engine — ask questions against your documents and receive precise, citation-grounded answers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAGPipeline                              │
│                                                                  │
│  Query ──► HybridRetriever ──────────────────────────────────►  │
│               │                                                  │
│               ├── Dense:  EmbedderProtocol → DocumentStore ANN  │
│               ├── Sparse: BM25Okapi (in-store corpus)           │
│               ├── Fuse:   RRF (K=60, Cormack et al. 2009)       │
│               ├── Expand: child → parent context resolution     │
│               └── Rerank: CrossEncoder ms-marco-MiniLM          │
│                                                                  │
│  Chunks ──► CitationGroundedGenerator ──────────────────────►   │
│               │                                                  │
│               ├── Context block: [src N] (source p.PAGE)        │
│               ├── Claude API call with citation-enforcing prompt │
│               └── Parse + validate [src N] markers              │
│                                                                  │
│  ◄──────────────────────────── CitedAnswer (answer + citations) │
└─────────────────────────────────────────────────────────────────┘

Document ingestion:
  Source (PDF / URL / dir) → Loader → HierarchicalChunker
  → BGEEmbedder / OpenAIEmbedder → DocumentStore (Chroma | Weaviate)
```

---

## Stack

| Layer | Default | Production | Rationale |
|---|---|---|---|
| Embedding | OpenAI text-embedding-3-small | BAAI/bge-large-en-v1.5 | Zero-infra default; BGE opt-in via `[local-embed]` |
| Vector store | Chroma (in-process) | Weaviate 1.27.0 | Zero Docker for dev; HNSW tuning + gRPC for prod |
| Retrieval fusion | BM25 + ANN → RRF (K=60) | Same | Parameter-free, robust to miscalibrated retrievers |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Same | Strong precision at low latency (~50ms CPU) |
| LLM | claude-sonnet-4-6 | Same | Citation-enforcing prompt, structured output |
| Tracing | Langfuse (opt-in no-op) | Langfuse cloud | Set `RAG_PLATFORM_LANGFUSE_SECRET_KEY` to enable; system runs identically without it |
| API | FastAPI + uvicorn | Same | `/query` endpoint, `/health` liveness probe |
| UI | Streamlit | Same | Chat interface with citation rendering |
| Eval | DeepEval + Ollama judge | Same | Pytest-native, zero API cost for local judge |

---

## Local Quickstart

### Prerequisites
- Python 3.11+
- An Anthropic API key

### 1. Clone and install

```bash
git clone https://github.com/adedaramola/rag-platform.git
cd rag-platform
pip install -e ".[dev]"
```

### 2. Configure

```bash
# .env (create in the rag-platform/ directory)
RAG_PLATFORM_ANTHROPIC_API_KEY=sk-ant-...

# Optional — defaults to OpenAI embeddings if omitted
RAG_PLATFORM_OPENAI_API_KEY=sk-proj-...
```

### 3. Ingest documents

```bash
# Single PDF
rag-platform-ingest --source /path/to/document.pdf

# Directory of PDFs
rag-platform-ingest --source /path/to/docs/

# Web page
rag-platform-ingest --source https://example.com/page
```

### 4. Start the API

```bash
make api
# → FastAPI running at http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### 5. Start the UI

```bash
make ui
# → Streamlit running at http://localhost:8501
```

### 6. Query via Python

```python
from rag.pipeline import build_pipeline
from rag.config import get_settings

pipeline = build_pipeline(get_settings())
result = pipeline.query("What is the purpose of multi-head attention?")
print(result.answer)
for c in result.citations:
    print(f"  [{c.index}] {c.source} p.{c.page}")
```

### 7. Query via API

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

## AWS Deployment

The `terraform/` directory provisions a production-ready deployment on AWS.

### Infrastructure

```
Internet
   │
   ▼
ALB (rag-platform-prod-alb)
   ├── :80   → FastAPI  (port 8000)
   └── :8501 → Streamlit UI (port 8501)
   │
   ▼
EC2 t3.medium (rag-platform-prod-api)
   ├── rag-platform-api.service  (uvicorn, 2 workers)
   └── rag-platform-ui.service   (streamlit)
   │
   ▼ (private network)
EC2 t3.medium (rag-platform-prod-weaviate)
   └── Weaviate 1.27.0 (Docker, 20GB EBS data volume)

S3 (rag-platform-prod-docs-*)
   └── Raw document storage
```

### Requirements

- AWS CLI configured (`aws configure`)
- Terraform >= 1.5
- An SSH key pair created in AWS EC2

### Deploy

```bash
cd terraform/

# Copy and fill in secrets
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — add API keys, key pair name, GitHub token

terraform init
terraform apply
```

Terraform outputs the live URLs when complete:

```
api_endpoint = "http://<alb-dns>/query"
api_docs_url = "http://<alb-dns>/docs"
ui_url       = "http://<alb-dns>:8501"
```

### Ingest documents on AWS

```bash
# 1. Upload PDF to S3
aws s3 cp my-document.pdf s3://$(terraform output -raw documents_bucket_name)/raw/

# 2. SSH into the API instance
ssh -i ~/.ssh/<key>.pem ec2-user@$(terraform output -raw api_instance_public_ip)

# 3. Download and ingest
sudo aws s3 cp s3://<bucket>/raw/my-document.pdf /opt/rag-platform/data/raw/my-document.pdf
sudo chown rag-platform:rag-platform /opt/rag-platform/data/raw/my-document.pdf
sudo -u rag-platform bash -c 'cd /opt/rag-platform && /opt/rag-platform/.venv/bin/rag-platform-ingest \
  --source /opt/rag-platform/data/raw/my-document.pdf'

# 4. Restart services to pick up the new BM25 corpus
sudo systemctl restart rag-platform-api rag-platform-ui
```

### Tear down

```bash
terraform destroy
```

---

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
│   ├── unit/                 # No network, no Docker — fast (94 tests)
│   ├── integration/          # Chroma: no Docker; Weaviate: skipped without Docker
│   └── e2e/                  # DeepEval gate (weekly CI schedule)
├── terraform/                # AWS infrastructure (EC2, ALB, Weaviate, S3)
├── docs/                     # ADRs and evaluation guide
├── config/                   # eval_thresholds.yaml
├── app.py                    # Streamlit chat UI
└── data/golden/              # Golden QA pairs for DeepEval evaluation
```

---

## Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Foundation: config, exceptions, interfaces | ✅ Complete |
| **Phase 2** | Stores, ingestion, retrieval, generation, unit tests | ✅ Complete |
| **Phase 3** | FastAPI, Streamlit UI, AWS deployment, DeepEval gate | ✅ Complete |
| **Phase 4** | Golden dataset curation, baseline establishment, eval gate activation | 🔄 In progress |
| **Phase 5** | Learned fusion weights, streaming responses, multi-tenancy | 📋 Planned |

---

## Evaluation

DeepEval runs **weekly** (Monday 06:00 UTC) and on manual trigger — not on every PR.
Uses a local Ollama judge by default (zero API cost). See [docs/evaluation.md](docs/evaluation.md).

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
| [ADR-001](docs/architecture.md#adr-001) | Hierarchical chunking (parent/child) | Accepted |
| [ADR-002](docs/architecture.md#adr-002) | Dual store backend (Chroma + Weaviate) | Accepted |
| [ADR-003](docs/architecture.md#adr-003) | Separate parent/child collections | Accepted |
| [ADR-004](docs/architecture.md#adr-004) | BM25 corpus co-located with vector store | Accepted |
| [ADR-005](docs/architecture.md#adr-005) | Drop `unstructured` dependency | Accepted |
| [ADR-006](docs/architecture.md#adr-006) | Citation grounding enforced at generation time | Accepted |
| [ADR-007](docs/architecture.md#adr-007) | DeepEval over RAGAS with local Ollama judge | Accepted |

---

## License

MIT
