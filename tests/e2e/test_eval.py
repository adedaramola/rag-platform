"""DeepEval evaluation gate. Runs on weekly schedule and manual trigger — not every PR.

See docs/evaluation.md for guidance on curating golden datasets and setting
empirical baselines before setting RAG_PLATFORM_EVAL_WARN_ONLY=false.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from rag.config import get_settings
from rag.evaluation.deepeval_runner import DeepEvalRunner, build_judge
from rag.pipeline import RAGPipeline, build_pipeline


@pytest.fixture(scope="session")
def rag_pipeline() -> RAGPipeline:
    """Build a real RAGPipeline for e2e evaluation.

    Skipped automatically when:
      - RAG_PLATFORM_ANTHROPIC_API_KEY is not set
      - eval_judge_backend=ollama and Ollama is not reachable
    """
    if not os.environ.get("RAG_PLATFORM_ANTHROPIC_API_KEY"):
        pytest.skip("RAG_PLATFORM_ANTHROPIC_API_KEY not set — skipping e2e eval")

    settings = get_settings()

    if settings.eval_judge_backend == "ollama":
        try:
            import ollama  # noqa: PLC0415

            ollama.Client(host=settings.eval_ollama_base_url).list()
        except Exception:
            pytest.skip(
                f"Ollama not reachable at {settings.eval_ollama_base_url} — "
                "run `ollama serve` or set RAG_PLATFORM_EVAL_JUDGE_BACKEND=openai"
            )

    return build_pipeline(settings)


@pytest.mark.e2e
def test_deepeval_thresholds(rag_pipeline: RAGPipeline) -> None:
    """DeepEval gate: fail (or warn) if any metric falls below its threshold."""
    import structlog  # noqa: PLC0415

    settings = get_settings()

    thresholds_path = Path("config/eval_thresholds.yaml")
    if not thresholds_path.exists():
        pytest.skip(f"Eval thresholds file not found: {thresholds_path}")

    thresholds: dict[str, float] = yaml.safe_load(thresholds_path.read_text())

    golden_path = settings.eval_golden_path
    if not golden_path.exists():
        pytest.skip(f"Golden dataset not found: {golden_path}. See docs/evaluation.md.")

    judge = build_judge(settings)
    runner = DeepEvalRunner(
        pipeline_fn=rag_pipeline.pipeline_fn,
        thresholds=thresholds,
        judge=judge,
        warn_only=settings.eval_warn_only,
    )

    result = runner.run(golden_path=golden_path)

    Path("reports").mkdir(exist_ok=True)
    runner.write_report(result, settings.eval_report_path)

    if not result.thresholds_met:
        if settings.eval_warn_only:
            structlog.get_logger().warning(
                "deepeval_threshold_warnings",
                failures=result.failures,
                scores=result.scores,
            )
        else:
            failure_lines = "\n".join(f"  · {f}" for f in result.failures)
            pytest.fail(f"DeepEval gate failed:\n{failure_lines}\n\nScores: {result.scores}")
