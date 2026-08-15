# Benchmarking Guide

RAG Platform reports classical retrieval quality, LLM-judged answer quality, and
production behavior separately. Combining them into one score would hide whether a
regression came from retrieval, generation, infrastructure, or the semantic cache.

## Retrieval benchmark

```bash
make benchmark
```

The JSON report compares dense ANN, BM25, hybrid RRF before reranking, and hybrid
after cross-encoder reranking. For each method it reports Precision@K, Recall@K,
F1@K, MRR, NDCG@K, and no-answer non-empty retrieval rate. It also reports mean,
P50, P95, and maximum latency for embedding, ANN, BM25, fusion, parent expansion,
reranking, and total retrieval.

Golden rows may contain relevance at three levels, in descending preference:

```json
{"question":"...", "relevant_chunk_ids":["uuid"]}
{"question":"...", "expected_sources":["doc.pdf"], "expected_pages":[4]}
{"question":"...", "expected_sources":["doc.pdf"]}
```

The report records which level was used. Source-only labels answer “did we retrieve
the correct document?” but cannot support claims about passage-level relevance.
No-answer rows are excluded from positive-query ranking metrics and reported through
`no_answer_nonempty_rate` instead.

## Answer quality

```bash
make eval
```

DeepEval measures faithfulness, answer relevancy, contextual precision, and
contextual recall with the configured LLM judge. See [evaluation.md](evaluation.md).

## Deployed load and cache benchmark

```bash
make load-test URL=http://your-api-host
```

The load test sends one cold request per selected query, then an exact-repeat
concurrent workload. It reports end-to-end P50/P95/P99, throughput, status/error
counts, cold-to-warm latency speedup, and `/metrics` counter deltas for semantic-cache
hits and misses. Use one API worker with the in-memory cache, or Redis for multi-worker
tests, so all requests observe a coherent cache and metrics store.

## 2026-08-15 local baseline

The synchronized Chroma baseline used 58 queries (55 positive, 3 no-answer), two PDFs,
732 non-empty pages, 732 parent chunks, and 2,508 child chunks. All positive labels
were source-level.

| Method | P@1 | P@5 | R@1 | R@5 | MRR | NDCG@5 | F1@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.909 | 0.800 | 0.909 | 1.000 | 0.945 | 0.959 | 0.889 |
| Dense | 0.964 | 0.971 | 0.964 | 0.982 | 0.975 | 0.975 | 0.976 |
| Hybrid, pre-rerank | 0.982 | 0.953 | 0.982 | 1.000 | 0.991 | 0.993 | 0.976 |
| Hybrid, post-rerank | 1.000 | 0.985 | 1.000 | 1.000 | 1.000 | 1.000 | 0.993 |

| Local retrieval stage | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| Embedding (OpenAI) | 200.953 | 166.242 | 325.707 |
| Dense ANN | 4.311 | 4.075 | 5.558 |
| BM25 | 7.230 | 7.445 | 10.072 |
| Reranking | 128.379 | 94.131 | 272.037 |
| Total retrieval | 344.347 | 276.535 | 611.775 |

Before rebuilding, the BM25 sidecar had 2,462 chunks while Chroma had zero vectors,
the expected Transformer paper was absent, and all 55 positive queries scored zero.
The clean rebuild reduced missing expected sources from one to zero, missing parent IDs
from 2,462 to zero, and restored 2,508 synchronized vector and lexical records.
