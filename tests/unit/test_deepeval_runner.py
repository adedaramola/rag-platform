"""Unit tests for DeepEvalRunner, build_judge, and EvalResult."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from rag.config import Settings
from rag.evaluation.deepeval_runner import (
    DeepEvalRunner,
    EvalResult,
    _load_golden,
    build_judge,
)


def _settings(**kwargs: object) -> Settings:
    return Settings(  # type: ignore[call-arg]
        anthropic_api_key=SecretStr("test"),
        openai_api_key=SecretStr("test-openai"),
        **kwargs,
    )


def _write_golden(path: Path, pairs: list[dict]) -> None:
    with path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


# ---------------------------------------------------------------------------
# build_judge
# ---------------------------------------------------------------------------


def test_build_judge_ollama_returns_model_with_correct_name() -> None:
    s = _settings(eval_judge_backend="ollama", eval_judge_model="llama3.1:8b")
    judge = build_judge(s)
    assert judge is not None
    assert "llama3.1" in judge.get_model_name()


def test_build_judge_openai_missing_key_raises() -> None:
    s = _settings(eval_judge_backend="openai")
    object.__setattr__(s, "openai_api_key", None)
    with pytest.raises(ValueError, match="RAG_PLATFORM_OPENAI_API_KEY"):
        build_judge(s)


def test_build_judge_unknown_backend_raises() -> None:
    s = _settings()
    object.__setattr__(s, "eval_judge_backend", "unknown")
    with pytest.raises(ValueError, match="unknown"):
        build_judge(s)


# ---------------------------------------------------------------------------
# _load_golden
# ---------------------------------------------------------------------------


def test_load_golden_parses_jsonl() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "golden.jsonl"
        _write_golden(
            path,
            [
                {"question": "Q1", "ground_truth": "A1"},
                {"question": "Q2", "ground_truth": "A2"},
            ],
        )
        pairs = _load_golden(path)

    assert len(pairs) == 2
    assert pairs[0]["question"] == "Q1"
    assert pairs[1]["ground_truth"] == "A2"


def test_load_golden_skips_blank_lines() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "golden.jsonl"
        path.write_text('{"question": "Q1"}\n\n{"question": "Q2"}\n')
        pairs = _load_golden(path)
    assert len(pairs) == 2


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


def test_eval_result_defaults() -> None:
    r = EvalResult(scores={"faithfulness": 0.9})
    assert r.passed is True
    assert r.failures == []
    assert r.report_path is None


def test_eval_result_with_failures() -> None:
    r = EvalResult(scores={"faithfulness": 0.7}, failures=["metric X failed"], passed=False)
    assert not r.passed
    assert len(r.failures) == 1


# ---------------------------------------------------------------------------
# DeepEvalRunner._check_thresholds
# ---------------------------------------------------------------------------


def test_check_thresholds_passes_when_above() -> None:
    runner = DeepEvalRunner(
        pipeline_fn=lambda q: {"actual_output": "", "retrieval_context": []},
        thresholds={"faithfulness": 0.80},
        warn_only=True,
    )
    failures = runner._check_thresholds({"faithfulness": 0.90})
    assert failures == []


def test_check_thresholds_fails_when_below() -> None:
    runner = DeepEvalRunner(
        pipeline_fn=lambda q: {"actual_output": "", "retrieval_context": []},
        thresholds={"faithfulness": 0.80},
        warn_only=True,
    )
    failures = runner._check_thresholds({"faithfulness": 0.70})
    assert len(failures) == 1
    assert "faithfulness" in failures[0]


def test_check_thresholds_skips_missing_metric() -> None:
    runner = DeepEvalRunner(
        pipeline_fn=lambda q: {"actual_output": "", "retrieval_context": []},
        thresholds={"faithfulness": 0.80, "answer_relevancy": 0.75},
        warn_only=True,
    )
    # Only faithfulness score provided
    failures = runner._check_thresholds({"faithfulness": 0.90})
    assert failures == []


# ---------------------------------------------------------------------------
# DeepEvalRunner.write_report
# ---------------------------------------------------------------------------


def test_write_report_creates_json_file() -> None:
    runner = DeepEvalRunner(
        pipeline_fn=lambda q: {"actual_output": "", "retrieval_context": []},
        thresholds={"faithfulness": 0.80},
        warn_only=True,
    )
    result = EvalResult(scores={"faithfulness": 0.85}, passed=True)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "reports" / "report.json"
        runner.write_report(result, out)
        data = json.loads(out.read_text())

    assert data["passed"] is True
    assert data["scores"]["faithfulness"] == 0.85
    assert "timestamp" in data
    assert "deepeval_version" in data


def test_write_report_includes_judge_metadata() -> None:
    mock_judge = MagicMock()
    mock_judge.get_model_name.return_value = "llama3.1:8b"
    runner = DeepEvalRunner(
        pipeline_fn=lambda q: {"actual_output": "", "retrieval_context": []},
        judge=mock_judge,
        warn_only=True,
    )
    result = EvalResult(scores={}, passed=True)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.json"
        runner.write_report(result, out)
        data = json.loads(out.read_text())

    assert data["judge_model"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# DeepEvalRunner._build_test_cases
# ---------------------------------------------------------------------------


def test_build_test_cases_returns_one_per_pair() -> None:
    calls: list[str] = []

    def mock_fn(q: str) -> dict:
        calls.append(q)
        return {"actual_output": f"answer to {q}", "retrieval_context": ["ctx"]}

    runner = DeepEvalRunner(pipeline_fn=mock_fn, warn_only=True)
    pairs = [
        {"question": "Q1", "ground_truth": "A1"},
        {"question": "Q2", "ground_truth": "A2"},
    ]
    test_cases = runner._build_test_cases(pairs)

    assert len(test_cases) == 2
    assert calls == ["Q1", "Q2"]


def test_build_test_cases_handles_pipeline_failure() -> None:
    def failing_fn(q: str) -> dict:
        raise RuntimeError("pipeline down")

    runner = DeepEvalRunner(pipeline_fn=failing_fn, warn_only=True)
    pairs = [{"question": "Q1", "ground_truth": "A1"}]
    test_cases = runner._build_test_cases(pairs)

    # Should not raise — failed case gets empty output
    assert len(test_cases) == 1
    assert test_cases[0].actual_output == ""


# ---------------------------------------------------------------------------
# DeepEvalRunner.run — integration-style with mocked metrics
# ---------------------------------------------------------------------------


def test_run_returns_eval_result() -> None:
    """run() loads golden data, calls pipeline, returns EvalResult."""

    def mock_fn(q: str) -> dict:
        return {"actual_output": "answer", "retrieval_context": ["context"]}

    runner = DeepEvalRunner(
        pipeline_fn=mock_fn,
        thresholds={"faithfulness": 0.80},
        warn_only=True,
    )

    with tempfile.TemporaryDirectory() as d:
        golden = Path(d) / "golden.jsonl"
        _write_golden(golden, [{"question": "Q1", "ground_truth": "A1"}])

        # Patch the metrics so no LLM calls happen
        with patch(
            "rag.evaluation.deepeval_runner.DeepEvalRunner._evaluate",
            return_value={
                "faithfulness": 0.9,
                "answer_relevancy": 0.85,
                "contextual_precision": 0.80,
                "contextual_recall": 0.75,
            },
        ):
            result = runner.run(golden)

    assert isinstance(result, EvalResult)
    assert result.passed is True
    assert result.scores["faithfulness"] == 0.9


def test_run_marks_failed_when_below_threshold() -> None:
    def mock_fn(q: str) -> dict:
        return {"actual_output": "answer", "retrieval_context": ["context"]}

    runner = DeepEvalRunner(
        pipeline_fn=mock_fn,
        thresholds={"faithfulness": 0.90},
        warn_only=True,
    )

    with tempfile.TemporaryDirectory() as d:
        golden = Path(d) / "golden.jsonl"
        _write_golden(golden, [{"question": "Q1", "ground_truth": "A1"}])

        with patch(
            "rag.evaluation.deepeval_runner.DeepEvalRunner._evaluate",
            return_value={"faithfulness": 0.70},
        ):
            result = runner.run(golden)

    assert result.passed is False
    assert len(result.failures) == 1
