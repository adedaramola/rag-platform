"""RAG Platform — domain-specific RAG engine with hybrid retrieval and citation-grounded generation.

Public API:
    build_pipeline  — assemble a production RAGPipeline from Settings
    RAGPipeline     — the assembled query pipeline
    get_settings    — return the cached Settings singleton
"""

from rag.config import Settings, get_settings
from rag.pipeline import RAGPipeline, build_pipeline

__all__ = [
    "RAGPipeline",
    "Settings",
    "build_pipeline",
    "get_settings",
]
