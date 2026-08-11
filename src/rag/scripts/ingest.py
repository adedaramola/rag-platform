"""CLI entrypoint for document ingestion.

Zero business logic — orchestration only. All logic lives in the rag package.
Entry point registered in pyproject.toml as: rag-platform-ingest = "rag.scripts.ingest:main"
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import structlog

from rag.config import get_settings
from rag.ingestion.chunker import HierarchicalChunker
from rag.ingestion.embedder import get_embedder
from rag.ingestion.loaders import Document, DocxLoader, PDFLoader, TextLoader, WebLoader
from rag.interfaces.store import Chunk
from rag.store.factory import get_store

logger = structlog.get_logger(__name__)


_EXTENSION_LOADERS: dict[str, Any] = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": DocxLoader,
}


def _resolve_sources(source: str) -> list[tuple[Any, Any]]:
    """Return a list of (loader, path_or_url) pairs from the given source argument.

    Accepts:
      - A URL (http:// or https://) — fetched as a web page
      - A single file: .pdf, .txt, .md, .docx
      - A directory — recursively globs all supported file types
    """
    path = Path(source)

    if source.startswith(("http://", "https://")):
        return [(WebLoader(), source)]

    if path.is_file():
        loader_cls = _EXTENSION_LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            logger.error("unsupported_file_type", source=source, suffix=path.suffix)
            sys.exit(1)
        return [(loader_cls(), path)]

    if path.is_dir():
        pairs: list[tuple[Any, Any]] = []
        for ext, loader_cls in _EXTENSION_LOADERS.items():
            for file in sorted(path.rglob(f"*{ext}")):
                pairs.append((loader_cls(), file))
        if not pairs:
            logger.warning(
                "no_supported_files_found",
                directory=source,
                supported=list(_EXTENSION_LOADERS.keys()),
            )
        return pairs

    logger.error("unknown_source_type", source=source)
    sys.exit(1)


def main() -> None:
    """CLI entry point for rag-platform-ingest."""
    parser = argparse.ArgumentParser(
        description=(
            "RAG Platform document ingestion — loads, chunks, embeds, and indexes documents."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="PDF file path, URL (http/https), or directory containing PDFs",
    )
    parser.add_argument(
        "--env",
        default="dev",
        choices=["dev", "ci", "prod"],
        help="Environment label (logged for observability; does not alter Settings)",
    )
    args = parser.parse_args()

    settings = get_settings()
    log = logger.bind(source=args.source, env=args.env, store_backend=settings.store_backend)

    store = get_store(settings)
    embedder = get_embedder(settings)
    chunker = HierarchicalChunker(settings)

    sources = _resolve_sources(args.source)
    if not sources:
        log.error("no_sources_resolved")
        sys.exit(1)

    all_child_chunks: list[dict[str, Any]] = []
    parent_count = 0
    child_count = 0
    doc_count = 0

    # Phase 1: load documents in parallel (I/O-bound — PDF reads, HTTP fetches)
    def _load(loader: Any, path_or_url: Any) -> list[Document]:
        return cast(list[Document], loader.load(path_or_url))

    loaded: list[tuple[Any, list[Document]]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_source = {
            pool.submit(_load, loader, path_or_url): path_or_url for loader, path_or_url in sources
        }
        for future in as_completed(future_to_source):
            path_or_url = future_to_source[future]
            try:
                loaded.append((path_or_url, future.result()))
            except Exception as exc:
                log.error("document_load_failed", path=str(path_or_url), error=str(exc))

    # Phase 2: chunk all documents and collect parents/children
    all_parents: list[Chunk] = []
    all_children: list[Chunk] = []
    for _path_or_url, documents in loaded:
        for doc in documents:
            chunks = list(chunker.chunk_document(doc.text, doc.metadata))
            all_parents.extend(c for c in chunks if c.is_parent())
            all_children.extend(c for c in chunks if not c.is_parent())

    if not all_children:
        log.warning("no_chunks_produced", sources=len(sources))
        return

    # Phase 3: single embed_batch call across all documents
    child_embeddings = embedder.embed_batch([c.text for c in all_children])

    # Phase 4: upsert to store
    store.upsert_chunks(all_parents, embeddings=None)
    store.upsert_chunks(all_children, embeddings=child_embeddings)

    all_child_chunks = [{"id": c.id, "text": c.text, **c.metadata} for c in all_children]
    parent_count = len(all_parents)
    child_count = len(all_children)
    doc_count = sum(len(docs) for _, docs in loaded)

    # Merge with the existing corpus so re-ingesting doc B does not erase doc A's
    # BM25 entries — matches upsert_chunks accumulate semantics for symmetric hybrid retrieval.
    corpus_dict = {doc["id"]: doc for doc in store.load_bm25_corpus()}
    for chunk in all_child_chunks:
        corpus_dict[chunk["id"]] = chunk
    store.store_bm25_corpus(list(corpus_dict.values()))

    log.info(
        "ingestion_complete",
        documents=doc_count,
        parents=parent_count,
        children=child_count,
        sources=len(sources),
    )
