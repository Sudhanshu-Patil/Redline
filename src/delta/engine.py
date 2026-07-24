"""Delta engine orchestration: runs the three alignment tiers in order, then
classifies matched pairs as modified/unchanged and unmatched elements as
added/removed, with a per-delta confidence score. Entry point: compute_delta().

Determinism (BRIEF rule 2): every function this module calls (align.*, the
classification below) is pure Python + numpy over a local, weights-frozen
embedding model — never the LLMClient. See
tests/test_delta_engine.py::test_classifier_never_invokes_llm_client for the
proof this module is required to satisfy.
"""

from typing import Literal

from pydantic import BaseModel, Field

from src.canonical.model import BBox, CanonicalDocument, Element, ElementType
from src.config import settings
from src.delta.align import (
    Embedder,
    MatchedPair,
    SentenceTransformerEmbedder,
    element_key,
    embedding_proximity_match,
    exact_key_match,
    geometry_match,
)
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

ChangeType = Literal["added", "removed", "modified"]
MatchTier = Literal["exact_key", "geometry", "embedding_proximity"]


class Delta(BaseModel):
    change_type: ChangeType
    element_type: ElementType
    old_element_id: str | None = None
    new_element_id: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    old_bbox: BBox | None = None
    new_bbox: BBox | None = None
    match_tier: MatchTier | None = None  # None for added/removed
    confidence: float = Field(ge=0.0, le=1.0)


class DeltaStats(BaseModel):
    total_a: int
    total_b: int
    matched: int
    unchanged: int
    added: int
    removed: int
    modified: int
    matched_by_tier: dict[str, int]
    alignment_rate: float  # fraction of all elements matched, any tier -- descriptive
    exact_key_rate: float  # fraction of *keyed* elements matched via tier 1 -- used for the warning


class DeltaReport(BaseModel):
    pid_a: str
    pid_b: str
    deltas: list[Delta]
    stats: DeltaStats
    warnings: list[str] = Field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _pair_confidence(pair: MatchedPair) -> float:
    """match_score is 1.0 for exact-key matches, a distance/similarity-derived
    value in [0,1] for the other two tiers (see align.py) -- one formula
    covers all tiers uniformly."""
    base = min(pair.a.extraction_confidence, pair.b.extraction_confidence)
    return round(base * pair.match_score, 4)


_GEOMETRY_UNCHANGED_EPSILON = 1e-6  # bbox_distance below this counts as "didn't move"


def _delta_from_pair(pair: MatchedPair) -> Delta | None:
    if pair.a.type == "geometry" and pair.b.type == "geometry":
        # Geometry elements carry no text (plan §3) -- text equality is
        # always trivially true (both ""), so "modified" has to mean
        # "moved" instead. Without this branch, Pair 3's moved valve
        # (GT3-MOVE) matched correctly via the geometry tier but was
        # silently classified as unchanged.
        if pair.bbox_distance <= _GEOMETRY_UNCHANGED_EPSILON:
            return None  # unchanged: didn't move
    elif _normalize(pair.a.text) == _normalize(pair.b.text):
        return None  # unchanged; not reported as a delta
    return Delta(
        change_type="modified",
        element_type=pair.b.type,
        old_element_id=pair.a.id,
        new_element_id=pair.b.id,
        old_text=pair.a.text,
        new_text=pair.b.text,
        old_bbox=pair.a.bbox,
        new_bbox=pair.b.bbox,
        match_tier=pair.tier,  # type: ignore[arg-type]
        confidence=_pair_confidence(pair),
    )


def _delta_from_removed(el: Element) -> Delta:
    return Delta(
        change_type="removed",
        element_type=el.type,
        old_element_id=el.id,
        old_text=el.text,
        old_bbox=el.bbox,
        confidence=round(el.extraction_confidence, 4),
    )


def _delta_from_added(el: Element) -> Delta:
    return Delta(
        change_type="added",
        element_type=el.type,
        new_element_id=el.id,
        new_text=el.text,
        new_bbox=el.bbox,
        confidence=round(el.extraction_confidence, 4),
    )


def compute_delta(
    doc_a: CanonicalDocument, doc_b: CanonicalDocument, embedder: Embedder | None = None
) -> DeltaReport:
    with tracing.span("delta.compute", pid_a=doc_a.pid, pid_b=doc_b.pid) as sp:
        embedder = embedder or SentenceTransformerEmbedder()

        m1, a1, b1 = exact_key_match(doc_a.elements, doc_b.elements)
        m2, a2, b2 = geometry_match(a1, b1)
        m3, a3, b3 = embedding_proximity_match(a2, b2, embedder)

        all_matched = m1 + m2 + m3
        deltas: list[Delta] = []
        unchanged = 0
        for pair in all_matched:
            delta = _delta_from_pair(pair)
            if delta is None:
                unchanged += 1
            else:
                deltas.append(delta)
        deltas.extend(_delta_from_removed(el) for el in a3)
        deltas.extend(_delta_from_added(el) for el in b3)

        # Deterministic ordering for downstream consumers (report renderer, eval diffing).
        deltas.sort(key=lambda d: (d.change_type, d.old_element_id or "", d.new_element_id or ""))

        total_a = len(doc_a.elements)
        total_b = len(doc_b.elements)
        matched_by_tier = {
            "exact_key": len(m1), "geometry": len(m2), "embedding_proximity": len(m3),
        }  # fmt: skip
        alignment_rate = len(all_matched) / max(min(total_a, total_b), 1)

        # The warning is based on the exact-key rate specifically, not the
        # overall (tier-3-inflated) alignment_rate. Two structurally similar
        # but unrelated documents (same P&ID template, different equipment --
        # plan §6's Pair 4) coincidentally place *some* element near *some*
        # other at nearly every page position, which tier 3's proximity
        # matching picks up regardless of same-type gating -- on real
        # samples this kept the overall rate high (0.74-0.86) for BOTH a
        # genuine revision and an unrelated pair, too close to threshold on.
        # The exact-key rate isn't fooled by layout coincidence (it requires
        # literal text identity): measured 0.99 on the genuine revision vs.
        # 0.24 on the negative control -- a clean, well-separated signal.
        keyed_a = sum(1 for el in doc_a.elements if element_key(el) is not None)
        keyed_b = sum(1 for el in doc_b.elements if element_key(el) is not None)
        exact_key_rate = len(m1) / max(min(keyed_a, keyed_b), 1)

        warnings: list[str] = []
        if (
            min(keyed_a, keyed_b) >= settings.low_alignment_min_elements
            and exact_key_rate < settings.low_alignment_rate_threshold
        ):
            warnings.append(
                f"Low exact-key alignment ({exact_key_rate:.1%}) between {doc_a.pid} and "
                f"{doc_b.pid} — these may not be a genuine revision pair rather than a "
                "heavily-edited one."
            )

        stats = DeltaStats(
            total_a=total_a,
            total_b=total_b,
            matched=len(all_matched),
            unchanged=unchanged,
            added=sum(1 for d in deltas if d.change_type == "added"),
            removed=sum(1 for d in deltas if d.change_type == "removed"),
            modified=sum(1 for d in deltas if d.change_type == "modified"),
            matched_by_tier=matched_by_tier,
            alignment_rate=round(alignment_rate, 4),
            exact_key_rate=round(exact_key_rate, 4),
        )
        sp["deltas"] = len(deltas)
        sp["added"] = stats.added
        sp["removed"] = stats.removed
        sp["modified"] = stats.modified
        sp["unchanged"] = stats.unchanged
        sp["alignment_rate"] = stats.alignment_rate
        sp["exact_key_rate"] = stats.exact_key_rate
        log.info(
            "delta computed",
            extra={
                "extra_fields": {
                    "pid_a": doc_a.pid,
                    "pid_b": doc_b.pid,
                    "added": stats.added,
                    "removed": stats.removed,
                    "modified": stats.modified,
                    "alignment_rate": stats.alignment_rate,
                }
            },
        )
        return DeltaReport(
            pid_a=doc_a.pid, pid_b=doc_b.pid, deltas=deltas, stats=stats, warnings=warnings
        )
