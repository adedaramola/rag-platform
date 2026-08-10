# Evaluation Guide

RAG Platform uses [DeepEval](https://docs.confident-ai.com/) as its evaluation framework.
This guide explains what the metrics mean, how to build a golden dataset, and
how to set defensible thresholds.

**Why DeepEval over RAGAS:**
- Pytest-native — metrics are assertions, not a separate runner
- Self-explaining failures — each metric reports its diagnostic reasoning
- Pluggable judge — local Ollama (zero API cost) for local runs; CI uses OpenAI `gpt-4o-mini`
- Stable API across minor versions (RAGAS has broken on minor bumps historically)

---

## 1. What DeepEval Measures

### Faithfulness
Measures whether every claim in the generated answer is supported by the provided
context passages. Scored by an LLM judge that checks each claim against the context.

**Low score means:** The generator is hallucinating — producing claims not grounded in
the retrieved passages. Fix by: tightening the system prompt, reducing `llm_max_tokens`,
or improving retrieval precision so the LLM has less incentive to fill gaps.

### Answer Relevancy
Measures how directly the answer addresses the question. A high-scoring answer is
focused and on-topic; a low-scoring answer drifts or adds irrelevant hedging.

**Low score means:** The generator is producing verbose, tangential answers. Fix by:
adjusting the system prompt to emphasise conciseness, or reducing `top_k_rerank` to
limit context noise.

### Contextual Precision
Measures what fraction of the retrieved context passages are actually relevant to the
question. High precision means your retriever is not including noise.

**Low score means:** The retriever is returning too many irrelevant passages. Fix by:
reducing `top_k_dense`, increasing `top_k_rerank` selectivity, or tuning the
cross-encoder reranker model.

### Contextual Recall
Measures what fraction of the information needed to answer the question is present in
the retrieved context. High recall means you're not missing relevant passages.

**Low score means:** The retriever is missing relevant chunks. Fix by: increasing
`top_k_dense`, re-examining chunking strategy (child chunks may be too small), or
expanding the document corpus.

---

## 2. Curating a Golden Dataset

### Format
JSONL file at `data/golden/qa_pairs.jsonl`. One JSON object per line:

```json
{"question": "What is the retention policy for audit logs?", "ground_truth": "Audit logs are retained for 7 years per SOC 2 requirements.", "expected_sources": ["policy.pdf"]}
```

Fields:
- `question`: a realistic question a user would ask
- `ground_truth`: the correct, complete answer (used by contextual_recall metric)
- `expected_sources`: list of source filenames that should be retrieved (optional, for debugging)

### Dataset Size
- **Minimum viable:** 50 pairs — enough for rough signal but high variance
- **Recommended:** 150–200 pairs — stable signal, defensible thresholds
- **Coverage guidance:** include edge cases, ambiguous queries, multi-hop questions,
  and questions where the answer is explicitly *not* in the corpus

### Semi-automated Curation
1. Use the LLM to generate candidate Q&A pairs from your documents: prompt Claude with
   a document passage and ask it to generate 3 questions with ground-truth answers.
2. Human-review each pair — discard anything the model got wrong.
3. Deliberately add hard negative questions (no answer in corpus) to test refusal behaviour.

---

## 3. Running Evaluation Locally

```bash
make eval
```

This runs `pytest tests/e2e/test_eval.py -v --tb=short`.

Prerequisites:
- `RAG_PLATFORM_ANTHROPIC_API_KEY` set in `.env`
- Golden dataset at `data/golden/qa_pairs.jsonl`
- Documents ingested: `make ingest SOURCE=data/golden/docs/`

**Local Ollama judge (default — zero API cost):**
- Ollama running locally with the judge model pulled: `make ollama-pull`

**OpenAI judge (used in CI):**
- `RAG_PLATFORM_OPENAI_API_KEY` set in `.env`
- `RAG_PLATFORM_EVAL_JUDGE_BACKEND=openai` set in `.env` or passed on the command line

Output: `reports/deepeval_report.json` with scores, failures, judge info, and timestamp.

### Switching judge backends

**Local Ollama (default — zero API cost):**
```bash
# Default — no env vars needed if Ollama is running
make eval
```

**OpenAI (higher fidelity):**
```bash
RAG_PLATFORM_EVAL_JUDGE_BACKEND=openai make eval
```
Requires `RAG_PLATFORM_OPENAI_API_KEY` in `.env`. Uses `gpt-4o-mini` by default
(`RAG_PLATFORM_EVAL_JUDGE_OPENAI_MODEL` to override). Costs ~$0.02–0.10 per run at
100 golden questions.

---

## 4. Interpreting Results

| Failing metric        | Most likely cause                          | Where to look                |
|-----------------------|--------------------------------------------|------------------------------|
| faithfulness          | Generator hallucinating                    | `SYSTEM_PROMPT`, `llm_model` |
| answer_relevancy      | Answers too verbose or off-topic           | System prompt, `llm_max_tokens` |
| contextual_precision  | Retriever returning irrelevant passages    | `top_k_dense`, reranker model |
| contextual_recall     | Retriever missing relevant passages        | `top_k_dense`, chunk sizes   |

If faithfulness and contextual_recall both fail simultaneously, the chunking strategy is
likely the root cause — child chunks may be too small to carry sufficient signal.

---

## 5. Setting Baselines and Thresholds

**Do not set `RAG_PLATFORM_EVAL_WARN_ONLY=false` until you have empirical baselines.**

### Establishing baselines
1. Ingest your full document corpus.
2. Run `make eval` three times on a stable, unchanged pipeline.
3. Record the scores from each run.
4. For each metric, compute `mean - 0.05` (the 0.05 absorbs LLM judge variance of ±0.03–0.05).
5. Set those values in `config/eval_thresholds.yaml`.
6. Document your baseline scores in this file (append a table below).

### Activating the hard gate
Once baselines are set:
```
RAG_PLATFORM_EVAL_WARN_ONLY=false
```

Now threshold violations fail CI. `eval.yml` runs weekly and on manual trigger.

### Adjusting thresholds over time
- If a metric consistently exceeds its threshold by >0.1, raise the threshold.
- If flakiness causes false failures (score variance > ±0.05), lower the threshold by 0.03.
- Document every threshold change with the date and reason.

---

## Baseline Score Log

| Date | faithfulness | answer_relevancy | contextual_precision | contextual_recall | Judge | Notes |
|------|-------------|-----------------|----------------------|-------------------|-------|-------|
| 2026-04-27 | 0.708 | 0.806 | 0.699 | 0.923 | gpt-4o-mini | First observed run — 58 questions, attention_is_all_you_need.pdf, warn_only=true |

**Observations from first run:**
- `faithfulness` (0.708) is 0.142 below the 0.850 starting threshold — generator is producing
  claims not fully grounded in retrieved context. Priority fix: tighten the system prompt.
- `contextual_precision` (0.699) is 0.051 below the 0.750 starting threshold — retriever
  is returning some irrelevant passages ahead of relevant ones. Priority fix: reranker tuning.
- `answer_relevancy` (0.806) and `contextual_recall` (0.923) are comfortably above threshold.

**Next steps before flipping `RAG_PLATFORM_EVAL_WARN_ONLY=false`:**
1. Run eval ≥2 more times to confirm score stability (LLM judge variance ±0.03–0.05).
2. If scores are stable, lower thresholds to `mean - 0.05` or improve the pipeline first.
3. Document updated thresholds here before activating the hard gate.
