"""Domain exception hierarchy. Never raise bare ValueError or RuntimeError from this package."""


class RAGError(Exception):
    """Base exception for all RAG Platform errors. Stores typed context alongside the message."""

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = context or {}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestionError(RAGError):
    """Raised when document ingestion fails."""


class DocumentLoadError(IngestionError):
    """Raised when a document cannot be loaded from a given source."""

    def __init__(self, source: str, message: str = "") -> None:
        self.source = source
        super().__init__(
            message or f"Failed to load document from: {source}",
            context={"source": source},
        )


class ChunkingError(IngestionError):
    """Raised when chunking a document fails."""

    def __init__(self, document_id: str, message: str = "") -> None:
        self.document_id = document_id
        super().__init__(
            message or f"Failed to chunk document: {document_id}",
            context={"document_id": document_id},
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalError(RAGError):
    """Raised when retrieval fails at a specific pipeline step."""

    def __init__(self, query: str, step: str, message: str = "") -> None:
        self.query = query
        self.step = step
        super().__init__(
            message or f"Retrieval failed at step '{step}' for query: {query!r}",
            context={"query": query, "step": step},
        )


class EmbeddingError(RetrievalError):
    """Raised when the embedder fails to produce vectors."""

    def __init__(self, model: str, query: str = "", message: str = "") -> None:
        self.model = model
        super().__init__(
            query=query,
            step="embedding",
            message=message or f"Embedding failed with model: {model}",
        )
        self.context["model"] = model


class IndexError(RetrievalError):
    """Raised when the BM25 or vector index operation fails."""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class GenerationError(RAGError):
    """Raised when the LLM generation step fails."""


class CitationGroundingError(GenerationError):
    """Raised when a generated answer contains no citation markers."""

    def __init__(self, answer: str, reason: str) -> None:
        self.answer = answer
        self.reason = reason
        super().__init__(
            f"Citation grounding failed: {reason}",
            context={"answer": answer, "reason": reason},
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class StoreError(RAGError):
    """Raised when a document store operation fails."""


class ConnectionError(StoreError):  # noqa: A001
    """Raised when the store backend cannot be reached."""

    def __init__(self, host: str, port: int, message: str = "") -> None:
        self.host = host
        self.port = port
        super().__init__(
            message or f"Cannot connect to store at {host}:{port}",
            context={"host": host, "port": port},
        )


class SchemaError(StoreError):
    """Raised when the store schema does not match expectations."""

    def __init__(self, expected: str, actual: str, message: str = "") -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            message or f"Schema mismatch — expected: {expected!r}, got: {actual!r}",
            context={"expected": expected, "actual": actual},
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationError(RAGError):
    """Raised when the RAGAS evaluation harness fails."""


class ThresholdViolationError(EvaluationError):
    """Raised when a RAGAS metric falls below its required threshold."""

    def __init__(self, metric: str, actual: float, required: float) -> None:
        self.metric = metric
        self.actual = actual
        self.required = required
        super().__init__(
            str(self),
            context={"metric": metric, "actual": actual, "required": required},
        )

    def __str__(self) -> str:
        return (
            f"Metric '{self.metric}' scored {self.actual:.3f}, "
            f"below required threshold {self.required:.3f}"
        )
