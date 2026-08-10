# Architecture Decision Records

This document records the key architectural decisions made in RAG Platform, following the
[ADR format](https://github.com/joelparkerhenderson/architecture-decision-record).

---

## ADR-001: Hierarchical Chunking Over Flat Chunking

**Status:** Accepted

**Context:**
Flat chunking forces a precision/context tradeoff. Small chunks (e.g. 300 tokens) retrieve
with high precision — the retrieved passage is tightly relevant to the query — but lose
surrounding context, leaving the LLM without enough information to answer fully. Large chunks
(e.g. 1500 tokens) preserve context but dilute relevance scores, causing the retriever to
return passages that are broadly related but not precisely targeted.

Neither alone is sufficient for production accuracy.

**Decision:**
Use a parent/child chunking strategy:
- **Child chunks** (~300 tokens) are the retrieval units. Their small size ensures the ANN
  search returns precisely relevant passages.
- **Parent chunks** (~1500 tokens) are passed to the LLM as context. Each child carries a
  `parent_id` reference resolved at query time.

The retriever fetches children, resolves their `parent_id`s, deduplicates on `parent_id`,
and passes the parent passages to the generator.

**Consequences:**
- ✅ Retrieval precision without context loss
- ✅ Re-ranking operates on child-level granularity (more candidates, better signal)
- ❌ Doubles storage footprint (both parent and child collections required)
- ❌ Adds a parent resolution step in the retrieval path

---

## ADR-002: Dual Store Backend (Chroma Default, Weaviate Production)

**Status:** Accepted

**Context:**
Requiring Docker just to run the test suite or develop locally raises the contribution
barrier significantly. A developer who clones the repo should be able to run the full
pipeline in under five minutes without installing Docker.

Chroma runs in-process with zero setup and is a credible production choice at moderate
scale (<10M vectors). Weaviate is the right choice at enterprise scale: it provides HNSW
index tuning, gRPC batch support, multi-tenancy, and a mature operational model.

**Decision:**
`RAG_PLATFORM_STORE_BACKEND=chroma|weaviate` selects the backend via environment variable.
A factory function (`store/factory.py`) maps the setting to a concrete implementation.
All downstream code works against `DocumentStoreProtocol` — the backend is invisible to
the retriever, generator, and pipeline.

**Consequences:**
- ✅ Zero-infrastructure dev experience: `pip install -e ".[dev]"` is sufficient
- ✅ CI runs without Docker service containers
- ✅ Backend swap requires one env var change, not a code change
- ❌ Two store implementations to maintain
- ❌ Feature parity must be enforced via the Protocol test suite

---

## ADR-003: Separate Parent and Child Collections

**Status:** Accepted

**Context:**
An earlier design stored parent chunks with zero vectors in the same collection as child
chunks, relying on an `is_parent=False` filter in every ANN query to exclude them. This
creates a silent correctness bug: any query that accidentally omits the filter returns
parent passages as top results. The bug is easy to introduce (a missing filter argument)
and hard to detect (results look plausible but are wrong).

**Decision:**
Store parent chunks in a dedicated collection with no vector index
(`DocumentChunkParent` in Weaviate, `{collection_name}_parents` in Chroma local dev).
Parents are fetch-by-ID only and cannot appear in ANN results by construction — there is
no vector index to search and no filter that could be omitted.

**Consequences:**
- ✅ The filter-omission bug is architecturally impossible
- ✅ No zero-vector pollution in the child collection
- ✅ Parent fetch is a simple ID lookup, not a filtered ANN search
- ❌ Two collections to manage per store implementation
- ❌ Slightly more complex schema setup

---

## ADR-004: BM25 Corpus Co-located with Vector Store

**Status:** Accepted

**Context:**
An in-memory-only BM25 index creates split state: the vector index survives a process
restart but the BM25 index does not, requiring a full corpus rebuild on every cold start.
A local JSON file decouples BM25 state from vector state, creating a synchronisation
obligation (the file can go out of sync with the vector index if ingestion fails midway).

**Decision:**
`DocumentStoreProtocol` includes `store_bm25_corpus` and `load_bm25_corpus` methods.
Both Chroma (JSON sidecar at `{persist_dir}/bm25_corpus.json`, local dev) and Weaviate
(singleton `WeaviateCorpus` collection, production) persist the corpus alongside their
vector data. `HybridRetriever` calls `load_bm25_corpus()` in `__init__`, restoring both
retrieval indexes automatically on startup with no manual rebuild step.

**Consequences:**
- ✅ Single source of truth for all retrieval state
- ✅ Restarts are transparent — no `rag-platform-ingest --rebuild-bm25` step needed
- ✅ Vector index and BM25 corpus always in sync (shared write path)
- ❌ Adds two methods to the store interface and both implementations
- ❌ Large corpora incur a JSON serialisation cost on every ingest

---

## ADR-005: Drop `unstructured` as a Dependency

**Status:** Accepted

**Context:**
`unstructured` has a history of breaking API changes between minor versions and pulls in
heavy system-level dependencies (libmagic, poppler, tesseract) that complicate CI setup
and increase contributor friction. It is also overkill for the initial scope: PDFs and
web pages.

**Decision:**
Use `pypdf` for PDFs and `httpx + BeautifulSoup` for web content. No `unstructured`.
If complex document formats (docx, pptx, scanned images) are needed in the future,
add `unstructured` as a pinned optional extra at that time.

**Consequences:**
- ✅ Stable, lightweight dependency surface
- ✅ Fast CI installs (no system library dependencies)
- ✅ No surprise API breaks between minor versions
- ❌ No OCR support for scanned PDFs
- ❌ No Office format (docx, pptx) support out of the box

---

## ADR-006: Citation Grounding Enforced at Generation Time

**Status:** Accepted

**Context:**
An earlier approach considered post-hoc citation filtering: generate the answer freely,
then strip any claims that don't have citations. The problem is that this approach
allows hallucinated content to exist in the final answer with no visible signal — a claim
with no citation would simply be stripped silently. The system would appear to work
correctly while producing ungrounded responses.

**Decision:**
The system prompt requires `[src N]` on every factual claim. `CitationGroundedGenerator`
raises `CitationGroundingError` if the response contains zero citation markers. This makes
uncited claims a detectable, raiseable error rather than silent data loss.

**Consequences:**
- ✅ Hallucinations produce a detectable, raiseable error
- ✅ Citation coverage is measurable and auditable
- ✅ The grounding contract is explicit and version-controlled (in `SYSTEM_PROMPT`)
- ❌ Adds a prompt-compliance failure mode requiring prompt iteration
- ❌ Slightly constrains answer style toward shorter, more structured responses

---

## ADR-007: DeepEval Over RAGAS With a Local Ollama Judge

**Status:** Accepted

**Context:**
RAGAS pinning is fragile — the API has broken on minor version bumps historically,
requiring explicit pins (`ragas==0.1.21`) that fall behind security fixes. Additionally,
RAGAS uses a hosted LLM judge for faithfulness and answer relevancy metrics. At 200 golden
questions × 4 metrics × weekly runs, hosted-model costs accumulate quickly for a portfolio
project. RAGAS also runs as a separate runner, not as pytest assertions, which means
failures don't integrate naturally with CI tooling.

**Decision:**
Use DeepEval with a local Ollama judge (`llama3.1:8b`) as the default for local runs.
DeepEval is pytest-native (metrics are assertions, failures include diagnostic reasoning),
has a stable API across minor versions, and supports pluggable judge backends.
`RAG_PLATFORM_EVAL_JUDGE_BACKEND=openai` switches to `gpt-4o-mini` for higher-fidelity scoring;
this is the judge used in the weekly CI gate (Ollama on CPU runners is too slow for 58+
questions within the 60-minute job timeout).

Four metrics: `FaithfulnessMetric`, `AnswerRelevancyMetric`,
`ContextualPrecisionMetric`, `ContextualRecallMetric`. Thresholds start in warn-only mode
(`RAG_PLATFORM_EVAL_WARN_ONLY=true`) until empirical baselines are established.

**Consequences:**
- ✅ Pytest-native — failures are first-class CI failures with diagnostic output
- ✅ Stable API — no fragile version pinning required
- ✅ Self-explaining failures — each metric reports why it failed
- ✅ Local runs are zero API cost (Ollama judge)
- ❌ CI eval uses OpenAI `gpt-4o-mini` — costs ~$0.02–0.10 per weekly run
- ❌ Local Ollama judge agreement with GPT-4o-mini is ~85%, not 100%
