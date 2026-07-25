"""Grounded chat orchestration (plan §7): hybrid retrieval -> rerank ->
cited answer, with an explicit refusal path when nothing retrieved supports
the question.

Determinism boundary (BRIEF rule 2, same spirit as the delta engine):
retrieval, reranking, citation-extraction and the "nothing retrieved"
refusal are all pure/deterministic. The LLM is only ever asked to
(a) write prose grounded in the passages it's handed and (b) say so in a
fixed, parseable format when it can't -- never asked to decide relevance or
alignment, which stays entirely in chat/index.py and chat/rerank.py.
"""

import re
from typing import Protocol

from pydantic import BaseModel, Field

from src.canonical.model import ElementType
from src.chat.index import ChatIndex, RetrievedChunk
from src.chat.llm import LLMClient
from src.chat.rerank import CrossEncoderReranker, Reranker, rerank_chunks
from src.config import settings
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

_NOT_GROUNDED_PREFIX = "NOT_GROUNDED:"

_SYSTEM_PROMPT = f"""You are answering questions about engineering drawings (P&IDs) using \
ONLY the numbered context passages provided below the question. Each passage is prefixed \
with its citation id in square brackets, like [pid:adapter:00042].

Rules:
1. Answer using ONLY information present in the provided context. Never use outside \
knowledge about engineering, P&IDs, or the specific equipment.
2. Cite the source of every factual claim by including its bracketed id inline, e.g. \
"The setpoint is 260 bar (g) [26-KA-901_B:pdf_native:00042]."
3. A passage like "16. SOME TEXT HERE." is the full content of note 16 -- treat the \
leading number as the note's identifier and answer directly from that text. A short \
passage that is ONLY a bare reference (e.g. just the words "NOTE 16" with nothing else) \
is a cross-reference to that note elsewhere on the drawing, not its content -- if a fuller \
numbered passage with the same number is also present, use that one and ignore the bare \
reference.
4. If the provided context does not contain enough information to answer the question, \
respond with EXACTLY: "{_NOT_GROUNDED_PREFIX} <one short sentence explaining what's \
missing>" and nothing else. Do not guess or use outside knowledge to fill the gap.
5. Be concise."""

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


class LLM(Protocol):
    def complete(
        self, system: str, user: str, max_tokens: int = 1024, purpose: str = ""
    ) -> str: ...


class Citation(BaseModel):
    element_id: str
    pid: str
    revision_label: str | None
    type: ElementType
    page: int
    text: str


class ChatAnswer(BaseModel):
    query: str
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool
    refusal_reason: str | None = None
    retrieved_count: int
    reranked_count: int
    invalid_citation_ids: list[str] = Field(default_factory=list)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        rev = f", {c.revision_label}" if c.revision_label else ""
        lines.append(f"[{c.element_id}] (page {c.page}{rev}): {c.text}")
    return "\n".join(lines)


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_citations(answer_text: str, valid_ids: set[str]) -> tuple[list[str], list[str]]:
    """Splits every bracketed [id] the LLM emitted into ones that match a
    chunk actually handed to it (trustworthy) vs. not (a hallucinated or
    malformed citation -- surfaced, never silently trusted or dropped)."""
    found = _CITATION_RE.findall(answer_text)
    valid = _dedupe([f for f in found if f in valid_ids])
    invalid = _dedupe([f for f in found if f not in valid_ids])
    return valid, invalid


_NO_GROUNDING_TEXT = "I don't have grounding for that in the submitted documents."


def answer_question(
    index: ChatIndex,
    query: str,
    llm: LLM | None = None,
    reranker: Reranker | None = None,
) -> ChatAnswer:
    with tracing.span("chat.answer", query_len=len(query)) as sp:
        retrieved = index.hybrid_search(query, top_k=settings.retrieval_top_k)
        sp["retrieved_count"] = len(retrieved)

        if not retrieved:
            sp["refused"] = True
            return ChatAnswer(
                query=query,
                answer_text=_NO_GROUNDING_TEXT,
                refused=True,
                refusal_reason="no relevant content retrieved",
                retrieved_count=0,
                reranked_count=0,
            )

        # Exact-source chunks are deterministic, certain matches (the query
        # named a literal tag or note number) -- always kept, never left to
        # compete with the cross-encoder's judgment. Found live (2026-07-25,
        # after fixing chat/index.py's note-number lookup): several "what
        # does note N say" questions still refused, because the now-
        # correctly-retrieved note could still be reranked out of the
        # rerank_top_k=5 slots sent to the LLM -- the same crowding-out
        # mechanism already documented in chat/index.py for the HH:250/
        # PIT-9062 case, just also reachable for exact hits, which by
        # construction should never lose that competition.
        exact = [c for c in retrieved if c.source == "exact"]
        vector_only = [c for c in retrieved if c.source != "exact"]
        remaining_budget = max(settings.rerank_top_k - len(exact), 0)
        reranked = exact + rerank_chunks(
            query, vector_only, reranker or CrossEncoderReranker(), top_k=remaining_budget
        )
        sp["reranked_count"] = len(reranked)

        context = _build_context(reranked)
        client = llm or LLMClient()
        raw = client.complete(
            system=_SYSTEM_PROMPT,
            user=f"{context}\n\nQuestion: {query}",
            max_tokens=512,
            purpose="chat_answer",
        ).strip()

        if raw.startswith(_NOT_GROUNDED_PREFIX):
            reason = raw[len(_NOT_GROUNDED_PREFIX) :].strip() or "insufficient grounded context"
            sp["refused"] = True
            return ChatAnswer(
                query=query,
                answer_text=raw,
                refused=True,
                refusal_reason=reason,
                retrieved_count=len(retrieved),
                reranked_count=len(reranked),
            )

        valid_ids = {c.element_id for c in reranked}
        valid_cited, invalid_cited = _extract_citations(raw, valid_ids)
        by_id = {c.element_id: c for c in reranked}
        citations = [
            Citation(
                element_id=cid,
                pid=by_id[cid].pid,
                revision_label=by_id[cid].revision_label,
                type=by_id[cid].type,
                page=by_id[cid].page,
                text=by_id[cid].text,
            )
            for cid in valid_cited
        ]
        if invalid_cited:
            log.warning(
                "LLM cited id(s) not present in retrieved context",
                extra={"extra_fields": {"invalid_citation_ids": invalid_cited}},
            )
        sp["refused"] = False
        sp["citations"] = len(citations)
        sp["invalid_citations"] = len(invalid_cited)
        return ChatAnswer(
            query=query,
            answer_text=raw,
            citations=citations,
            refused=False,
            retrieved_count=len(retrieved),
            reranked_count=len(reranked),
            invalid_citation_ids=invalid_cited,
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from eval.schema import load_manifest
    from src.ingest.dwg import DwgAdapter
    from src.ingest.pdf_native import PdfNativeAdapter
    from src.ingest.pdf_scanned import PdfScannedAdapter

    _ADAPTERS = {
        "pdf_native": PdfNativeAdapter,
        "pdf_scanned": PdfScannedAdapter,
        "dwg": DwgAdapter,
    }

    parser = argparse.ArgumentParser(description="Grounded chat REPL over a sample pair.")
    parser.add_argument("manifest", type=Path, help="path to a pair's manifest.json")
    args = parser.parse_args()

    if not LLMClient().is_configured:
        raise SystemExit(
            "No LLM configured (set ANTHROPIC_API_KEY or LLM_PROVIDER=openai_compatible "
            "with LLM_API_KEY in .env) -- chat needs a live model to answer."
        )

    manifest = load_manifest(args.manifest)
    doc_a = _ADAPTERS[manifest.doc_a.format]().ingest(
        Path(manifest.doc_a.path),
        pid=manifest.doc_a.pid,
        revision_label=manifest.doc_a.revision_label,
    )
    doc_b = _ADAPTERS[manifest.doc_b.format]().ingest(
        Path(manifest.doc_b.path),
        pid=manifest.doc_b.pid,
        revision_label=manifest.doc_b.revision_label,
    )

    chat_index = ChatIndex(collection_name=f"chat_{manifest.pair_id}")
    n_a = chat_index.index_document(doc_a)
    n_b = chat_index.index_document(doc_b)
    print(f"Indexed {n_a} elements from {doc_a.pid} and {n_b} from {doc_b.pid}.")
    print("Ask a question (Ctrl-C to quit):")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        result = answer_question(chat_index, query)
        print(result.answer_text)
        if result.citations:
            print("Citations:", ", ".join(c.element_id for c in result.citations))
        if result.invalid_citation_ids:
            unknown = ", ".join(result.invalid_citation_ids)
            print(f"(warning: model cited unknown id(s): {unknown})")
