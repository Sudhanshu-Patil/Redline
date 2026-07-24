"""Alignment: matches Elements between two CanonicalDocuments across three
tiers (plan §4). Every function here is deterministic, non-LLM logic (BRIEF
rule 2) — see tests/test_delta_engine.py::test_classifier_never_invokes_llm_client
for the proof.

Tier 1 — exact key match: tag/instrument_loop/valve/line_number (by their
canonical text) and note definitions (by note_number). Highest confidence,
fully deterministic. Elements whose defining trait IS a value that changes
(setpoint, dimension) are deliberately excluded here — exact-keying
"SP = 257 bar (g)" would never match its own edited "SP = 260 bar (g)".

Tier 2 — geometry-aware match (DWG, plan §4.2): same layer + entity_type
(+ block_name when present) is the coarse key; bbox proximity disambiguates
duplicates and bounds how far a "moved" element can be before it's no longer
a candidate.

Tier 3 — embedding + bbox proximity, for everything tiers 1/2 left unmatched.
Thresholds are calibrated against real edits, not guessed (see PROVENANCE.md
and the Pair 1 sample): a pure value edit at a stable position can score
*below* the embedding-similarity threshold on text alone — "19057" vs "20500"
scores 0.10 cosine similarity, while an unrelated-but-nearby element ("776
NOTE 28") scores 0.25 against the same target. Embedding similarity alone
would pick the WRONG candidate. So proximity is treated as decisive, not a
tiebreaker, when two elements occupy nearly the same slot:

  - within `alignment_bbox_proximity_tolerance` ("tight"): position alone
    is decisive, regardless of text similarity — but candidates *within*
    this band are still ranked by actual closeness (mapped to [0.9, 1.0]),
    not collapsed to a single flat score. A dense P&ID can pack two
    different setpoints (e.g. an HH and an LL annotation) both within the
    tight radius of the same instrument bubble; collapsing them to an equal
    score turns the choice into an arbitrary id tie-break instead of picking
    the genuinely nearest one. Caught on real Pair 1 data: an early version
    matched "HH: 245" to an unrelated "LL: 120" (distance 0.017) instead of
    its true partner "HH: 250" (distance 0.003) because both were inside the
    tight band and scored identically.
  - within the looser `alignment_tier3_loose_proximity` AND embedding
    similarity clears `alignment_embedding_similarity_threshold`: accepted,
    scored by similarity but scaled below 0.9 so this band can never
    outrank a tight-band match.
  - otherwise: not a candidate pair.

All candidates across the full leftover pool are ranked together and
assigned greedily, best score first (see `_greedy_assign`) — this is what
lets an element with a perfect self-match (unchanged text, zero movement)
claim itself *before* any ambiguous candidate is considered, so a changed
value's true partner isn't stolen by a merely-nearby unrelated element that
would otherwise have already found its own exact match anyway.

Known scaling note: candidate generation is O(n*m) per tier over the
leftover pool after coarser tiers have thinned it. Fine at Pair-1 scale
(hundreds of elements); a 500-sheet set (plan §14) would want a spatial
index (KD-tree / grid) instead of the full pairwise scan.
"""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.canonical.model import Element, ElementType
from src.config import settings
from src.observability import tracing

_TIER1_TEXT_KEY_TYPES: frozenset[ElementType] = frozenset(
    {"tag", "instrument_loop", "valve", "line_number"}
)


def element_key(el: Element) -> str | None:
    """Deterministic exact-match key, or None if this element must go
    through a later tier."""
    if el.type in _TIER1_TEXT_KEY_TYPES:
        text = el.text.strip()
        return text or None
    if el.type == "note" and el.attributes.get("kind") == "definition":
        note_number = el.attributes.get("note_number")
        return f"note-def:{note_number}" if note_number else None
    return None


def _bbox_center(el: Element) -> tuple[float, float]:
    x0, y0, x1, y1 = el.bbox.normalized
    return (x0 + x1) / 2, (y0 + y1) / 2


def _center_distance(a: Element, b: Element) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return math.hypot(ax - bx, ay - by)


@dataclass(frozen=True)
class MatchedPair:
    a: Element
    b: Element
    tier: str  # "exact_key" | "geometry" | "embedding_proximity"
    match_score: float  # 0..1, tier-specific meaning
    bbox_distance: float


def _greedy_assign(
    candidates: list[tuple[float, Element, Element, float]],
) -> tuple[list[MatchedPair], set[str], set[str]]:
    """candidates: (score, a, b, bbox_distance), higher score wins.
    Deterministic tie-break by (a.id, b.id) so results never depend on
    input order or dict/set iteration order.
    """
    ranked = sorted(candidates, key=lambda c: (-c[0], c[1].id, c[2].id))
    used_a: set[str] = set()
    used_b: set[str] = set()
    pairs: list[MatchedPair] = []
    for score, a, b, dist in ranked:
        if a.id in used_a or b.id in used_b:
            continue
        pairs.append(MatchedPair(a=a, b=b, tier="", match_score=score, bbox_distance=dist))
        used_a.add(a.id)
        used_b.add(b.id)
    return pairs, used_a, used_b


def exact_key_match(
    a_elements: list[Element], b_elements: list[Element]
) -> tuple[list[MatchedPair], list[Element], list[Element]]:
    with tracing.span("delta.align.exact_key_match") as sp:
        a_by_key: dict[str, list[Element]] = {}
        for el in a_elements:
            key = element_key(el)
            if key is not None:
                a_by_key.setdefault(key, []).append(el)
        b_by_key: dict[str, list[Element]] = {}
        for el in b_elements:
            key = element_key(el)
            if key is not None:
                b_by_key.setdefault(key, []).append(el)

        matched: list[MatchedPair] = []
        matched_a_ids: set[str] = set()
        matched_b_ids: set[str] = set()
        for key in sorted(a_by_key.keys() & b_by_key.keys()):
            a_group = a_by_key[key]
            b_group = b_by_key[key]
            candidates = [
                (1.0 - _center_distance(a, b), a, b, _center_distance(a, b))
                for a in a_group
                for b in b_group
            ]
            pairs, used_a, used_b = _greedy_assign(candidates)
            matched.extend(
                MatchedPair(
                    a=p.a, b=p.b, tier="exact_key", match_score=1.0, bbox_distance=p.bbox_distance
                )
                for p in pairs
            )
            matched_a_ids |= used_a
            matched_b_ids |= used_b

        unmatched_a = [el for el in a_elements if el.id not in matched_a_ids]
        unmatched_b = [el for el in b_elements if el.id not in matched_b_ids]
        sp["matched"] = len(matched)
        return matched, unmatched_a, unmatched_b


def geometry_match(
    a_elements: list[Element], b_elements: list[Element]
) -> tuple[list[MatchedPair], list[Element], list[Element]]:
    with tracing.span("delta.align.geometry_match") as sp:
        a_geo = [el for el in a_elements if el.type == "geometry"]
        b_geo = [el for el in b_elements if el.type == "geometry"]

        def coarse_key(el: Element) -> tuple[str, str, str]:
            return (
                el.attributes.get("layer", ""),
                el.attributes.get("entity_type", ""),
                el.attributes.get("block_name", ""),
            )

        a_by_key: dict[tuple[str, str, str], list[Element]] = {}
        for el in a_geo:
            a_by_key.setdefault(coarse_key(el), []).append(el)
        b_by_key: dict[tuple[str, str, str], list[Element]] = {}
        for el in b_geo:
            b_by_key.setdefault(coarse_key(el), []).append(el)

        matched: list[MatchedPair] = []
        matched_a_ids: set[str] = set()
        matched_b_ids: set[str] = set()
        for key in sorted(a_by_key.keys() & b_by_key.keys()):
            _layer, _entity_type, block_name = key
            # A named block reference (INSERT) has strong identity beyond
            # position, so it's reasonable to let it move a lot and still be
            # "the same instance" (the generous configured tolerance). A bare
            # primitive (LINE/CIRCLE/... with no block_name) has *no*
            # identity beyond position and entity_type, both of which are
            # shared by every sibling on the same layer -- so a coarse
            # tolerance there doesn't find "the moved one", it force-pairs
            # whichever two happen to be left over, even when the honest
            # answer is "unrelated, one removed and one added elsewhere"
            # (caught on Pair 3: a removed drain stub matched an unrelated
            # added tie-in line at distance 0.16, while the genuinely moved
            # named block was only 0.09 away -- too close to separate by a
            # single threshold). Bare primitives get a much tighter radius.
            max_dist = (
                settings.geometry_match_max_bbox_distance
                if block_name
                else settings.geometry_match_unnamed_max_bbox_distance
            )
            candidates = []
            for a in a_by_key[key]:
                for b in b_by_key[key]:
                    dist = _center_distance(a, b)
                    if dist <= max_dist:
                        candidates.append((1.0 - dist, a, b, dist))
            pairs, used_a, used_b = _greedy_assign(candidates)
            matched.extend(
                MatchedPair(
                    a=p.a, b=p.b, tier="geometry",
                    match_score=p.match_score, bbox_distance=p.bbox_distance,
                )
                for p in pairs
            )
            matched_a_ids |= used_a
            matched_b_ids |= used_b

        unmatched_a = [el for el in a_elements if el.id not in matched_a_ids]
        unmatched_b = [el for el in b_elements if el.id not in matched_b_ids]
        sp["matched"] = len(matched)
        return matched, unmatched_a, unmatched_b


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized embedding vector per input text, same order."""
        ...


class SentenceTransformerEmbedder:
    """Local, deterministic embedding model (plan §2) — never the LLMClient.
    Lazily loaded once and cached at class level so repeated engine runs in
    the same process don't reload the model.
    """

    _model: object | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if SentenceTransformerEmbedder._model is None:
            from sentence_transformers import SentenceTransformer

            with tracing.span("delta.load_embedding_model", model=settings.embedding_model):
                SentenceTransformerEmbedder._model = SentenceTransformer(settings.embedding_model)
        model = SentenceTransformerEmbedder._model
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)  # type: ignore[attr-defined]
        return np.asarray(vectors)


def embedding_proximity_match(
    a_elements: list[Element], b_elements: list[Element], embedder: Embedder
) -> tuple[list[MatchedPair], list[Element], list[Element]]:
    with tracing.span(
        "delta.align.embedding_proximity_match", a=len(a_elements), b=len(b_elements)
    ) as sp:
        min_len = settings.alignment_min_embed_text_len
        a_eligible = [el for el in a_elements if len(el.text.strip()) >= min_len]
        b_eligible = [el for el in b_elements if len(el.text.strip()) >= min_len]

        if not a_eligible or not b_eligible:
            sp["matched"] = 0
            return [], a_elements, b_elements

        a_texts = [el.text.strip() for el in a_eligible]
        b_texts = [el.text.strip() for el in b_eligible]
        a_emb = embedder.embed(a_texts)
        b_emb = embedder.embed(b_texts)

        tight = settings.alignment_bbox_proximity_tolerance
        loose = settings.alignment_tier3_loose_proximity
        min_sim = settings.alignment_embedding_similarity_threshold

        candidates: list[tuple[float, Element, Element, float]] = []
        for i, a in enumerate(a_eligible):
            for j, b in enumerate(b_eligible):
                if a.type != b.type:
                    # Two structurally-similar-but-unrelated documents (same
                    # P&ID template, different equipment -- plan §6's Pair 4)
                    # will coincidentally place *some* element near *some*
                    # other element at nearly every position. Without this
                    # gate, tight proximity alone matched pairs like
                    # "NOTE 26"<->"N3601" or "DSS"<->"150#" at score ~0.99
                    # purely by layout coincidence. Real edits never change
                    # an element's classified type, so this costs nothing on
                    # genuine revisions while rejecting cross-type noise.
                    continue
                dist = _center_distance(a, b)
                if dist <= tight:
                    # Position is decisive -- but rank by closeness *within*
                    # this band too (mapped to [0.9, 1.0]), rather than a
                    # flat 1.0 for everything inside the radius. A dense P&ID
                    # can have two different setpoints (e.g. HH and LL) both
                    # sitting within the tight radius of the same instrument
                    # bubble; collapsing them to an equal score turns the
                    # choice between them into an arbitrary id tie-break
                    # instead of picking the genuinely nearest one. [0.9, 1.0]
                    # keeps this band always ranked above the loose band below
                    # (whose max is capped under 0.9), while still preserving
                    # fine-grained ordering inside it.
                    score = 0.9 + 0.1 * (1 - dist / tight) if tight > 0 else 1.0
                elif dist <= loose:
                    sim = float(np.dot(a_emb[i], b_emb[j]))
                    if sim < min_sim:
                        continue
                    score = sim * 0.89  # always < 0.9: never outranks a tight-band match
                else:
                    continue
                candidates.append((score, a, b, dist))

        pairs, used_a_ids, used_b_ids = _greedy_assign(candidates)
        matched = [
            MatchedPair(
                a=p.a, b=p.b, tier="embedding_proximity",
                match_score=p.match_score, bbox_distance=p.bbox_distance,
            )
            for p in pairs
        ]
        unmatched_a = [el for el in a_elements if el.id not in used_a_ids]
        unmatched_b = [el for el in b_elements if el.id not in used_b_ids]
        sp["matched"] = len(matched)
        return matched, unmatched_a, unmatched_b
