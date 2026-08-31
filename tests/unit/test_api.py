"""Unit tests for the authenticated retrieval-only HTTP contract."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import SecretStr
from pytest import MonkeyPatch

from rag.api import main as api
from rag.config import Settings
from rag.interfaces.retriever import RetrievedChunk


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=SecretStr("test-anthropic"),
        openai_api_key=SecretStr("test-openai"),
        api_key=SecretStr("agent-only-key"),
        approved_source_ids=["vpn-runbook"],
        search_max_excerpt_chars=100,
    )


def test_agent_search_returns_only_approved_bounded_evidence(monkeypatch: MonkeyPatch) -> None:
    pipeline = MagicMock()
    pipeline.search.return_value = [
        RetrievedChunk(
            id="approved",
            text="short child",
            source="/approved/VPN Runbook.pdf",
            page=4,
            score=0.9,
            parent_text=(
                "Restart the VPN client after refreshing its device certificate. "
                "Then verify the connection from a trusted network and contact support."
            ),
        ),
        RetrievedChunk(
            id="private",
            text="must not leave",
            source="private-notes.pdf",
            page=1,
            score=0.8,
        ),
    ]
    monkeypatch.setattr(api, "_pipeline", pipeline)
    monkeypatch.setattr(api, "get_settings", _settings)

    response = TestClient(api.app).post(
        "/v1/search",
        headers={
            "X-API-Key": "agent-only-key",
            "traceparent": "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01",
            "X-Workflow-ID": "12345678-1234-1234-1234-123456789abc",
        },
        json={"query": "VPN issue", "source_ids": ["vpn-runbook"], "max_chunks": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_id"].startswith("rag-")
    assert len(payload["chunks"]) == 1
    assert payload["chunks"][0]["citation_id"] == "C1"
    assert payload["chunks"][0]["source_id"] == "vpn-runbook"
    assert payload["chunks"][0]["page"] == 4
    assert len(payload["chunks"][0]["text"]) == 100
    assert payload["chunks"][0]["text"].startswith("Restart the VPN client")
    pipeline.search.assert_called_once_with(
        "VPN issue",
        trace_id="1234567890abcdef1234567890abcdef",
        workflow_id="12345678-1234-1234-1234-123456789abc",
    )


def test_agent_search_requires_configured_auth_and_approved_sources(
    monkeypatch: MonkeyPatch,
) -> None:
    pipeline = MagicMock()
    monkeypatch.setattr(api, "_pipeline", pipeline)
    monkeypatch.setattr(api, "get_settings", _settings)
    client = TestClient(api.app)

    missing_key = client.post("/v1/search", json={"query": "VPN issue"})
    unknown_source = client.post(
        "/v1/search",
        headers={"X-API-Key": "agent-only-key"},
        json={"query": "VPN issue", "source_ids": ["private-notes"]},
    )

    assert missing_key.status_code == 403
    assert unknown_source.status_code == 403
    pipeline.search.assert_not_called()


def test_source_id_is_stable_for_paths_and_urls() -> None:
    assert api._source_id("/docs/VPN Runbook.pdf") == "vpn-runbook"
    assert api._source_id("https://docs.example.com/runbooks/MFA Setup.html") == "mfa-setup"
