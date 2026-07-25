"""Retrieval index over a chat session's CanonicalDocument(s) and its
DeltaReport (plan §7): hybrid retrieval = a deterministic exact tag/text
lookup, unioned with chromadb vector search over the same embedding model
the delta engine uses (SentenceTransformerEmbedder, plan §2 -- one embedding
model, one place it's loaded, config-driven via EMBEDDING_MODEL).
index_document() indexes each revision's static content ("what is X");
index_delta() separately indexes the *relationship* between two revisions
("what changed about X") -- a raw element from either document can never
answer the second question on its own, since neither side of a change knows
about the other.

Exact lookup exists because P&ID tags ("26-KA-901", "PSV-204") are short,
high-signal strings that a user will often quote verbatim -- semantic
similarity is the wrong tool for "find this exact string", the same reason
the delta engine's tier 1 (src/delta/align.py::element_key) exact-keys tags
instead of embedding them.

Known limitation, measured on real Pair 1 data (2026-07-24): a bare value
annotation like "HH: 250" carries no textual link to the instrument tag it
belongs to ("PIT-9062" sits ~0.05 normalized-bbox-units away but is a
separate Element with no cross-reference in `attributes`). Asking "what is
the HH trip limit for PIT-9062?" retrieves both the exact tag match
"PIT-9062" *and* the value "HH: 250" inside the top-20 hybrid candidates
(vector score 0.43) -- but the generic cross-encoder reranker (not P&ID
domain-tuned) ranks the literal tag-name match higher than the
value-bearing fragment, and with rerank_top_k=5 the value can be crowded
out of the context actually sent to the LLM, producing a false
NOT_GROUNDED refusal even though the answer exists in the corpus. Indexing
both revisions into one collection compounds this: the unchanged tag
"PIT-9062" appears twice (once per revision) and can occupy two of the five
reranked slots on its own. No fix attempted here -- spatial-context
enrichment at index time was considered and rejected for this phase (real
risk of mis-attributing a value to the wrong nearby tag on dense layouts,
e.g. "TIT-9064" sits only ~0.006 farther from "HH: 245" than "PIT-9062"
does); this is exactly the kind of retrieval-quality gap plan §10's
recall@k/MRR eval and candid failure table are meant to surface and
quantify properly, rather than patching ad hoc.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import chromadb
from pydantic import BaseModel

from src.canonical.model import CanonicalDocument, ElementType
from src.config import settings
from src.delta.align import Embedder, SentenceTransformerEmbedder
from src.delta.engine import Delta, DeltaReport
from src.observability import tracing

_TAG_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/]{1,}")

# "note 37" / "note number 16" / "note #8" -- deliberately separate from
# _extract_tag_tokens: a bare note number ("37") is too short to survive that
# filter (len >= 3), and even if it did, it would never equal a whole note's
# full sentence text via the substring-token match exact_lookup does for
# tags. Notes need to be matched by their note_number *attribute* instead.
# Found live (2026-07-25): a real eval run against Pair 1's 15-question QA
# set showed 6 of 11 answerable questions getting a false NOT_GROUNDED
# refusal -- 5 of those 6 were exactly "what does note N say?" phrasings
# that fell through to vector search alone and lost the reranking race
# (see src/chat/index.py's own docstring on the similar, deliberately-NOT-
# fixed HH:250/PIT-9062 case for why that one variance is expected; this one
# has a clean deterministic fix since note_number is already extracted at
# ingest time by classify_block_lines).
_NOTE_REFERENCE_RE = re.compile(r"\bnote\s*(?:number\s*)?#?\s*(\d{1,3})\b", re.IGNORECASE)


def _extract_tag_tokens(query: str) -> list[str]:
    """P&ID tags always carry a digit (instrument/line/valve numbers) -- this
    filters out ordinary English words in the question so 'What is the SP for
    26-KA-901?' probes for '26-KA-901' but not 'What'/'the'/'SP'."""
    return [
        tok
        for tok in _TAG_TOKEN_RE.findall(query)
        if len(tok) >= 3 and any(c.isdigit() for c in tok)
    ]


def _extract_note_numbers(query: str) -> set[str]:
    return {m.group(1) for m in _NOTE_REFERENCE_RE.finditer(query)}


def _delta_chunk_id(d: Delta) -> str:
    return f"delta:{d.change_type}:{d.old_element_id or '_'}:{d.new_element_id or '_'}"


def _delta_page(d: Delta) -> int:
    """new_bbox is preferred (added/modified both have one); removed only
    has old_bbox. Both are None only for the geometry-move case index_delta
    already filters out before this is ever called."""
    bbox = d.new_bbox or d.old_bbox
    return bbox.page if bbox else 0


def _delta_chunk_text(d: Delta, pid_a: str, pid_b: str) -> str:
    """Phrases a Delta as a standalone sentence a semantic search can match
    against a "what changed" style question -- unlike a raw element's text,
    which only supports "what is X" questions since it describes one static
    revision, not a relationship between two."""
    if d.change_type == "modified":
        return (
            f"CHANGED ({d.element_type}) from {pid_a} to {pid_b}: '{d.old_text}' -> '{d.new_text}'"
        )
    if d.change_type == "added":
        return f"ADDED in {pid_b}, not present in {pid_a} ({d.element_type}): '{d.new_text}'"
    return f"REMOVED from {pid_a}, not present in {pid_b} ({d.element_type}): '{d.old_text}'"


class RetrievedChunk(BaseModel):
    element_id: str
    pid: str
    revision_label: str | None
    type: ElementType
    page: int
    text: str
    score: float
    source: Literal["exact", "vector"]


def _chunk_from_row(
    doc_id: str,
    text: str,
    meta: Mapping[str, Any],
    score: float,
    source: Literal["exact", "vector"],
) -> RetrievedChunk:
    revision_label = meta.get("revision_label")
    return RetrievedChunk(
        element_id=doc_id,
        pid=str(meta["pid"]),
        revision_label=str(revision_label) if revision_label else None,
        type=cast(ElementType, meta["type"]),
        page=int(meta["page"]),
        text=text,
        score=score,
        source=source,
    )


class ChatIndex:
    """Wraps one chromadb collection: the grounding corpus for one chat
    session (typically both sides of a submitted pair, indexed via two
    `index_document` calls so questions can span either revision, plus one
    `index_delta` call so questions can span the diff between them)."""

    def __init__(
        self,
        collection_name: str,
        persist_dir: Path | None = None,
        embedder: Embedder | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self._embedder = embedder or SentenceTransformerEmbedder()
        self._client = client or chromadb.PersistentClient(
            path=str(persist_dir or settings.chroma_persist_dir)
        )
        # Embeddings are L2-normalized (Embedder contract) -- cosine space
        # makes chroma's distance directly convertible to a similarity score.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def index_document(self, doc: CanonicalDocument) -> int:
        with tracing.span("chat.index.document", pid=doc.pid) as sp:
            eligible = [el for el in doc.elements if el.text.strip()]
            if not eligible:
                sp["indexed"] = 0
                return 0
            texts = [el.text.strip() for el in eligible]
            embeddings = self._embedder.embed(texts)
            ids = [el.id for el in eligible]
            metadatas: list[Mapping[str, Any]] = [
                {
                    "pid": doc.pid,
                    "revision_label": doc.revision_label or "",
                    "type": el.type,
                    "page": el.bbox.page,
                    "note_number": el.attributes.get("note_number", ""),
                }
                for el in eligible
            ]
            self._collection.upsert(
                ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas
            )
            sp["indexed"] = len(ids)
            return len(ids)

    def index_delta(self, report: DeltaReport) -> int:
        """Indexes each delta entry as its own retrievable chunk, phrased as
        a change description (see _delta_chunk_text) -- closes the gap where
        chat could answer "what is X" (grounded in one revision's static
        content via index_document) but not "what changed about X" (which
        needs the relationship between both revisions that only the delta
        report itself carries). Vector-search-only by design: a delta's text
        is a full descriptive sentence, never a bare tag, so it doesn't
        participate in exact_lookup's tag-token equality match -- semantic
        search is the right tool here, the same reasoning this module's own
        docstring gives for why exact_lookup exists for tags in the first
        place, just pointed the other direction.
        """
        with tracing.span("chat.index.delta", pid_a=report.pid_a, pid_b=report.pid_b) as sp:
            eligible = [
                d for d in report.deltas if (d.old_text or "").strip() or (d.new_text or "").strip()
            ]
            if not eligible:
                sp["indexed"] = 0
                return 0
            texts = [_delta_chunk_text(d, report.pid_a, report.pid_b) for d in eligible]
            embeddings = self._embedder.embed(texts)
            ids = [_delta_chunk_id(d) for d in eligible]
            combined_pid = f"{report.pid_a}→{report.pid_b}"
            metadatas: list[Mapping[str, Any]] = [
                {
                    "pid": combined_pid,
                    "revision_label": "",
                    "type": d.element_type,
                    "page": _delta_page(d),
                    "note_number": "",
                }
                for d in eligible
            ]
            self._collection.upsert(
                ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas
            )
            sp["indexed"] = len(ids)
            return len(ids)

    def exact_lookup(self, query: str) -> list[RetrievedChunk]:
        with tracing.span("chat.retrieve.exact") as sp:
            tokens = {t.lower() for t in _extract_tag_tokens(query)}
            note_numbers = _extract_note_numbers(query)
            if not tokens and not note_numbers:
                sp["matched"] = 0
                return []
            got = self._collection.get(include=["documents", "metadatas"])
            results = [
                _chunk_from_row(doc_id, text, meta, score=1.0, source="exact")
                for doc_id, text, meta in zip(
                    got["ids"], got["documents"] or [], got["metadatas"] or [], strict=True
                )
                if text.strip().lower() in tokens
                or str(meta.get("note_number") or "") in note_numbers
            ]
            sp["matched"] = len(results)
            return results

    def vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        with tracing.span("chat.retrieve.vector", top_k=top_k) as sp:
            if self._collection.count() == 0:
                sp["matched"] = 0
                return []
            query_embedding = self._embedder.embed([query])[0].tolist()
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks = [
                _chunk_from_row(doc_id, text, meta, score=max(0.0, 1.0 - dist), source="vector")
                for doc_id, text, meta, dist in zip(
                    result["ids"][0],
                    result["documents"][0],  # type: ignore[index]
                    result["metadatas"][0],  # type: ignore[index]
                    result["distances"][0],  # type: ignore[index]
                    strict=True,
                )
            ]
            sp["matched"] = len(chunks)
            return chunks

    def hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        with tracing.span("chat.retrieve.hybrid", top_k=top_k) as sp:
            exact = self.exact_lookup(query)
            vector = self.vector_search(query, top_k=top_k)
            seen_ids = {c.element_id for c in exact}
            merged = exact + [c for c in vector if c.element_id not in seen_ids]
            sp["exact_matched"] = len(exact)
            sp["vector_matched"] = len(vector)
            sp["merged"] = len(merged)
            return merged
