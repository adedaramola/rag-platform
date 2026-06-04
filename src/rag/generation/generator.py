"""Citation-grounded answer generator.

Citation grounding is enforced at generation time via prompt instruction and a
single repair pass. A draft answer is generated first, then revised to remove
unsupported claims and tighten citations before being returned to the caller.
See docs/architecture.md ADR-006.
"""

from __future__ import annotations

import re

import structlog

from rag.exceptions import CitationGroundingError, GenerationError
from rag.interfaces.generator import CitationRef, CitedAnswer
from rag.interfaces.retriever import RetrievedChunk

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a precise document assistant.\n"
    "Answer the user's question using ONLY the numbered source passages provided.\n"
    "Rules:\n"
    "- Cite every factual claim with [src N] immediately after it.\n"
    "- If multiple sources support a claim, use [src N, src M].\n"
    "- Do not include any information not present in the provided sources.\n"
    "- If sources are insufficient, say so explicitly — do not speculate.\n"
    "- Answer the question directly in 1 short paragraph, with no preamble.\n"
    "- Prefer the smallest complete answer that fully addresses the question."
)

GROUNDING_REPAIR_PROMPT = (
    "You are revising a draft answer for strict source grounding.\n"
    "Rewrite the draft so every factual claim is directly supported by the numbered sources.\n"
    "Rules:\n"
    "- Remove unsupported claims instead of weakening them.\n"
    "- Keep the answer short and direct.\n"
    "- Cite every factual claim with [src N] immediately after it.\n"
    "- If the sources are insufficient, say so explicitly and cite the closest relevant source.\n"
    "- Return only the final revised answer."
)

_CITATION_RE = re.compile(r"\[src\s+(\d+)\]")


class CitationGroundedGenerator:
    """Generate citation-grounded answers using Claude.

    Implements GeneratorProtocol. api_key injected — never reads env directly.
    """

    def __init__(self, model: str, api_key: str, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        try:
            import anthropic  # noqa: PLC0415

            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError("anthropic package is required. Run: pip install anthropic") from exc

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> CitedAnswer:
        """Generate an answer grounded in the provided chunks.

        Runs one repair pass to improve grounding. Raises CitationGroundingError
        if neither the draft nor the revised answer contain valid [src N] markers.
        """
        if not chunks:
            raise GenerationError("Cannot generate: no context chunks provided.")

        context_block = self._build_context_block(chunks)
        draft_prompt = f"{context_block}\n\nQuestion: {query}"

        log = logger.bind(model=self._model, query=query[:80], num_chunks=len(chunks))
        log.debug("generation_start")

        draft_answer, draft_input_tokens, draft_output_tokens = self._generate_text(
            system_prompt=SYSTEM_PROMPT,
            user_message=draft_prompt,
        )
        draft_citations = self._extract_citations(draft_answer, chunks)

        revised_prompt = f"{context_block}\n\nQuestion: {query}\n\nDraft answer:\n{draft_answer}"
        revised_answer, revised_input_tokens, revised_output_tokens = self._generate_text(
            system_prompt=GROUNDING_REPAIR_PROMPT,
            user_message=revised_prompt,
        )
        revised_citations = self._extract_citations(revised_answer, chunks)

        if revised_citations:
            final_answer = revised_answer
            citations = revised_citations
        elif draft_citations:
            final_answer = draft_answer
            citations = draft_citations
            log.warning(
                "repair_pass_lost_citations",
                hint="Falling back to cited draft answer after uncited repair output",
            )
        else:
            raise CitationGroundingError(
                answer=revised_answer or draft_answer,
                reason="model returned answer without valid [src N] citations after repair",
            )

        input_tokens = draft_input_tokens + revised_input_tokens
        output_tokens = draft_output_tokens + revised_output_tokens

        log.info(
            "generation_complete",
            citations=len(citations),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            repaired=bool(revised_citations),
        )
        return CitedAnswer(
            answer=final_answer,
            citations=citations,
            raw_context=chunks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _generate_text(self, system_prompt: str, user_message: str) -> tuple[str, int, int]:
        """Call the Anthropic API once and return text plus token counts."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            first_block = response.content[0]
            if not hasattr(first_block, "text"):
                raise GenerationError("Unexpected response block type from Claude API")
            answer = str(first_block.text)
        except Exception as exc:
            raise GenerationError(f"Claude API call failed: {exc}") from exc

        input_tokens = int(getattr(response.usage, "input_tokens", 0))
        output_tokens = int(getattr(response.usage, "output_tokens", 0))
        return answer, input_tokens, output_tokens

    @staticmethod
    def _build_context_block(chunks: list[RetrievedChunk]) -> str:
        """Format chunks as numbered source passages for the prompt."""
        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            page_label = f" p.{chunk.page}" if chunk.page is not None else ""
            lines.append(f"[src {i}] ({chunk.source}{page_label})\n{chunk.text}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[CitationRef]:
        """Parse [src N] markers from the answer into CitationRef objects.

        - Deduplicates: [src 1] ... [src 1] → one CitationRef
        - Silently drops out-of-range indices
        """
        seen: set[int] = set()
        refs: list[CitationRef] = []
        for match in _CITATION_RE.finditer(answer):
            idx = int(match.group(1))
            if idx in seen or idx < 1 or idx > len(chunks):
                continue
            seen.add(idx)
            chunk = chunks[idx - 1]
            refs.append(
                CitationRef(
                    index=idx,
                    source=chunk.source,
                    page=chunk.page,
                )
            )
        return refs
