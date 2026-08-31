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

## 2026-08-15 AWS baseline

The deployed benchmark used a `t3.medium` API node, a separate `t3.medium` Weaviate
node, OpenAI `text-embedding-3-small`, and the CPU cross-encoder reranker. Ingestion
produced 732 parent chunks and 2,507 child chunks from the same two PDFs and 732
non-empty pages. This is one child fewer than the local Chroma build and the exact
cause has not yet been isolated; no empty, duplicate, source-less, page-less, or
parent-less child chunks were detected in the AWS corpus.

The evaluation set contained 58 queries: 55 answerable queries and three no-answer
probes. All 55 positive labels identify the expected source document only, so these
numbers measure document/source retrieval, not passage-level relevance. Every method
returned results for all three no-answer probes (`no_answer_nonempty_rate=1.0`), which
shows that abstention still needs an explicit relevance threshold.

| AWS method | P@1 | P@5 | R@1 | R@5 | MRR | NDCG@5 | F1@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.891 | 0.782 | 0.891 | 1.000 | 0.932 | 0.949 | 0.878 |
| Dense | 0.945 | 0.935 | 0.945 | 0.964 | 0.955 | 0.957 | 0.949 |
| Hybrid, pre-rerank | 0.964 | 0.931 | 0.964 | 1.000 | 0.982 | 0.987 | 0.964 |
| Hybrid, post-rerank | 1.000 | 0.978 | 1.000 | 1.000 | 1.000 | 1.000 | 0.989 |

| AWS retrieval stage | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| Embedding (OpenAI) | 187.757 | 164.155 | 310.344 |
| Dense ANN (Weaviate) | 5.011 | 4.523 | 5.922 |
| BM25 | 14.183 | 13.202 | 25.308 |
| Parent expansion | 4.293 | 3.874 | 6.260 |
| Reranking (CPU) | 5,858.661 | 5,632.296 | 8,042.230 |
| Total retrieval | 6,070.306 | 5,830.300 | 8,183.350 |

Reranking accounts for about 96.5% of mean retrieval time on this instance. The
quality gain from pre- to post-rerank is measurable (P@1 0.964 to 1.000), but the
current CPU deployment pays roughly 5.86 seconds mean latency for it. This is the
first infrastructure optimization target.

### End-to-end load and cache results

The API ran one worker with the in-memory semantic cache so cache counters and entries
were process-consistent. Five distinct cold requests were followed by 50 exact-repeat
requests at concurrency five.

| Workload | Requests | Concurrency | Success | P50 ms | P95 ms | P99 ms | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold, sequential | 5 | 1 | 100% | 9,007.742 | 13,227.665 | 13,705.836 | n/a |
| Warm exact repeats | 50 | 5 | 100% | 244.810 | 749.238 | 1,134.525 | 12.56 req/s |
| Cold, distinct-query burst | 5 | 5 | 100% | 28,839.066 | 32,632.454 | 33,253.556 | 0.15 req/s |

The exact-repeat cache produced 50 hits and five misses, a 90.91% hit rate across the
55-request run. Warm P50 was 36.795 times faster than cold P50. Citation-presence
coverage was 100% for those 55 successful responses; this confirms the generation
guard emitted citations, but it is not a semantic faithfulness score.

The latest documented LLM-judge run remains the 2026-04-27 local DeepEval baseline:
faithfulness 0.708, answer relevancy 0.806, contextual precision 0.699, and contextual
recall 0.923 across 58 questions with `gpt-4o-mini` as judge. It predates this AWS
deployment and should not be presented as an AWS generation result. See
[evaluation.md](evaluation.md) for thresholds and interpretation.
