"""Public re-exports for the interfaces package."""

from rag.interfaces.cache import SemanticCacheProtocol
from rag.interfaces.embedder import EmbedderProtocol
from rag.interfaces.generator import CitationRef, CitedAnswer, GeneratorProtocol
from rag.interfaces.retriever import RetrievedChunk, RetrieverProtocol
from rag.interfaces.store import Chunk, DocumentStoreProtocol

__all__ = [
    "Chunk",
    "CitationRef",
    "CitedAnswer",
    "DocumentStoreProtocol",
    "EmbedderProtocol",
    "GeneratorProtocol",
    "RetrievedChunk",
    "RetrieverProtocol",
    "SemanticCacheProtocol",
]
