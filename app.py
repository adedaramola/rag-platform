"""RAG Platform RAG — Streamlit chat interface.

Connects to the FastAPI backend (rag.api.main).
The backend URL defaults to localhost:8000 but can be overridden:

    RAG_PLATFORM_API_URL=http://<alb-host> streamlit run app.py

Start the backend first (local):
    uvicorn rag.api.main:app --reload

Then run this app:
    streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_PLATFORM_API_URL", "http://localhost:8000").rstrip("/")
# When RAG_PLATFORM_API_KEY is set, the UI forwards it on every backend call.
# Leave unset for local dev (mirrors the server-side opt-in behaviour).
_API_KEY = os.environ.get("RAG_PLATFORM_API_KEY")
_AUTH_HEADERS: dict[str, str] = {"X-API-Key": _API_KEY} if _API_KEY else {}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Platform RAG",
    page_icon="📚",
    layout="centered",
)

st.title("📚 RAG Platform RAG")
st.caption("Citation-grounded document Q&A powered by hybrid retrieval.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_ok" not in st.session_state:
    st.session_state.api_ok = False


# ---------------------------------------------------------------------------
# Backend health check (once per session)
# ---------------------------------------------------------------------------


def check_backend() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return bool(r.status_code == 200)
    except Exception:
        return False


if not st.session_state.api_ok:
    st.session_state.api_ok = check_backend()

if not st.session_state.api_ok:
    st.error(
        "**Backend not reachable.** Start it with:\n```\nuvicorn rag.api.main:app --reload\n```",
        icon="🔌",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — document info, metrics, controls
# ---------------------------------------------------------------------------


def fetch_metrics() -> dict[str, Any] | None:
    """Fetch /metrics from the backend. Returns None on any failure."""
    try:
        r = httpx.get(f"{API_URL}/metrics", headers=_AUTH_HEADERS, timeout=3.0)
        r.raise_for_status()
        result: dict[str, Any] = r.json()
        return result
    except Exception:
        return None


with st.sidebar:
    st.header("RAG Platform")
    st.markdown(
        "Ask questions about your ingested documents. "
        "Every answer includes citations so you can verify the source."
    )
    st.divider()

    # Operational metrics
    st.markdown("**System metrics**")
    m = fetch_metrics()
    if m is None:
        st.caption("Metrics unavailable")
    else:
        total = m.get("total_queries", 0)
        p95 = m.get("p95_latency_ms")
        cost = m.get("avg_cost_usd")
        coverage = m.get("citation_coverage")

        col1, col2 = st.columns(2)
        col1.metric("Queries", total)
        col2.metric(
            "P95 latency",
            f"{p95:.0f} ms" if p95 is not None else "—",
            help="Computed after 20+ queries",
        )

        col3, col4 = st.columns(2)
        col3.metric(
            "Avg cost",
            f"${cost:.4f}" if cost is not None else "—",
            help="USD per query (input + output tokens)",
        )
        col4.metric(
            "Citation rate",
            f"{coverage * 100:.0f}%" if coverage is not None else "—",
            help="Fraction of queries returning ≥1 citation",
        )

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- Hybrid retrieval (BM25 + dense)\n"
        "- Cross-encoder re-ranking\n"
        "- Citation-grounded generation\n"
        "- ChromaDB vector store"
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s)"):
                for c in msg["citations"]:
                    page_label = f" · p.{c['page']}" if c.get("page") is not None else ""
                    st.markdown(f"**[{c['index']}]** `{c['source']}`{page_label}")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if question := st.chat_input("Ask a question about your documents…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call the backend
    with st.chat_message("assistant"), st.spinner("Retrieving and generating…"):
        try:
            response = httpx.post(
                f"{API_URL}/query",
                json={"question": question},
                headers=_AUTH_HEADERS,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            answer = data["answer"]
            citations = data.get("citations", [])
            context_chunks = data.get("context_chunks", 0)

            st.markdown(answer)

            if citations:
                label = f"📎 {len(citations)} source(s) · {context_chunks} chunks retrieved"
                with st.expander(label):
                    for c in citations:
                        page_label = f" · p.{c['page']}" if c.get("page") is not None else ""
                        st.markdown(f"**[{c['index']}]** `{c['source']}`{page_label}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
            )

        except httpx.HTTPStatusError as exc:
            error_msg = f"API error {exc.response.status_code}: {exc.response.text}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ {error_msg}", "citations": []}
            )
        except Exception as exc:
            error_msg = f"Could not reach backend: {exc}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ {error_msg}", "citations": []}
            )
