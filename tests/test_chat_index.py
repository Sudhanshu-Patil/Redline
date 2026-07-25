"""Tests for ChatIndex: real chromadb (EphemeralClient, in-memory, no disk
I/O) exercised for real, with a FakeEmbedder standing in for the actual
embedding model -- same split used by test_delta_align.py: the vector store
is tested for real, only the (slow, network-touching in prod) model call is
faked for determinism and speed.
"""

import uuid

import chromadb
import pytest

from src.canonical.model import BBox, CanonicalDocument
from src.chat.index import ChatIndex, _delta_chunk_text, _extract_note_numbers, _extract_tag_tokens
from src.delta.engine import Delta, DeltaReport, DeltaStats
from tests.conftest import FakeEmbedder
from tests.conftest import make_positioned_element as el


def doc(pid: str, elements, fmt="pdf_native", revision_label=None) -> CanonicalDocument:
    return CanonicalDocument(
        pid=pid, format=fmt, revision_label=revision_label, page_count=1,
        elements=elements, raw_source_path=f"{pid}.pdf",
    )  # fmt: skip


_VECTORS = {
    "26-KA-901": [1.0, 0.0, 0.0],
    "Compressor discharge pressure alarm high high": [0.0, 1.0, 0.0],
    "SP = 257 bar (g)": [0.0, 0.0, 1.0],
    "alarm setpoint question": [0.1, 0.9, 0.1],
    "26-KA-901 tag lookup": [0.9, 0.1, 0.0],
    "completely unrelated query": [0.0, 0.0, 0.0],
    "37. PSV 9066A/B SET PRESSURE REVISED TO 260 BAR(G).": [0.0, 0.0, -1.0],
}


@pytest.fixture
def sample_doc():
    return doc(
        "26-KA-901_A",
        [
            el("26-KA-901", element_type="tag"),
            el("Compressor discharge pressure alarm high high", element_type="note"),
            el("SP = 257 bar (g)", element_type="setpoint"),
            el("", element_type="geometry"),  # must be excluded (empty text)
        ],
        revision_label="Rev A",
    )


@pytest.fixture
def doc_with_note():
    return doc(
        "26-KA-901_B",
        [
            el(
                "37. PSV 9066A/B SET PRESSURE REVISED TO 260 BAR(G).",
                element_type="note",
                kind="definition",
                note_number="37",
            ),
        ],
    )


def _bbox(page: int = 0, x0: float = 0.1, y0: float = 0.1) -> BBox:
    return BBox(
        page=page, x0=x0, y0=y0, x1=x0 + 0.05, y1=y0 + 0.05, page_width=1.0, page_height=1.0
    )  # fmt: skip


def _stats() -> DeltaStats:
    return DeltaStats(
        total_a=1, total_b=1, matched=0, unchanged=0, added=1, removed=1, modified=2,
        matched_by_tier={}, alignment_rate=0.0, exact_key_rate=0.0,
    )  # fmt: skip


@pytest.fixture
def sample_report():
    return DeltaReport(
        pid_a="pair1_A",
        pid_b="pair1_B",
        deltas=[
            Delta(
                change_type="modified",
                element_type="setpoint",
                old_element_id="pair1_A:pdf_native:00042",
                new_element_id="pair1_B:pdf_native:00042",
                old_text="SP = 257 bar (g)",
                new_text="SP = 260 bar (g)",
                old_bbox=_bbox(),
                new_bbox=_bbox(),
                match_tier="exact_key",
                confidence=0.95,
            ),  # fmt: skip
            Delta(
                change_type="added",
                element_type="note",
                new_element_id="pair1_B:pdf_native:00099",
                new_text="42. NEW NOTE ADDED.",
                new_bbox=_bbox(page=1),
                confidence=1.0,
            ),  # fmt: skip
            Delta(
                change_type="removed",
                element_type="note",
                old_element_id="pair1_A:pdf_native:00088",
                old_text="41. OLD NOTE REMOVED.",
                old_bbox=_bbox(),
                confidence=1.0,
            ),  # fmt: skip
            Delta(
                # pure geometry move: no text on either side -- must be
                # excluded, there's nothing meaningful to phrase or search.
                change_type="modified",
                element_type="geometry",
                old_element_id="pair3_A:dwg:00005",
                new_element_id="pair3_B:dwg:00005",
                old_text="",
                new_text="",
                old_bbox=_bbox(),
                new_bbox=_bbox(x0=0.3, y0=0.3),
                match_tier="geometry",
                confidence=0.9,
            ),  # fmt: skip
        ],
        stats=_stats(),
    )


@pytest.fixture
def index():
    # chromadb.EphemeralClient() instances share a process-global in-memory
    # backend keyed by settings (verified empirically) -- a unique collection
    # name per test is what actually gives isolation, not a "fresh" client.
    return ChatIndex(
        collection_name=f"test_{uuid.uuid4().hex}",
        embedder=FakeEmbedder(_VECTORS),
        client=chromadb.EphemeralClient(),
    )


class TestExtractTagTokens:
    def test_finds_hyphenated_tag(self):
        assert "26-KA-901" in _extract_tag_tokens("What is the setpoint for 26-KA-901?")

    def test_ignores_plain_words(self):
        assert _extract_tag_tokens("What is the alarm setpoint?") == []

    def test_ignores_short_all_letter_tokens(self):
        assert _extract_tag_tokens("SP is high") == []

    def test_finds_alnum_tag_without_hyphen(self):
        assert "26BL9072" in _extract_tag_tokens("Tell me about 26BL9072 valve")


class TestExtractNoteNumbers:
    def test_finds_bare_note_reference(self):
        assert _extract_note_numbers("What does note 37 say?") == {"37"}

    def test_finds_single_digit_note(self):
        assert _extract_note_numbers("What does note 8 say about the flame arrester?") == {"8"}

    def test_finds_note_number_phrasing(self):
        assert _extract_note_numbers("What is note number 16 about?") == {"16"}

    def test_finds_hash_phrasing(self):
        assert _extract_note_numbers("What does note #19 describe?") == {"19"}

    def test_no_note_reference_returns_empty_set(self):
        assert _extract_note_numbers("What is the setpoint for 26-KA-901?") == set()

    def test_multiple_note_references(self):
        assert _extract_note_numbers("Compare note 8 and note 22") == {"8", "22"}


class TestIndexDocument:
    def test_indexes_all_text_bearing_elements(self, index, sample_doc):
        count = index.index_document(sample_doc)
        assert count == 3  # geometry (empty text) excluded

    def test_reindexing_is_idempotent(self, index, sample_doc):
        index.index_document(sample_doc)
        count = index.index_document(sample_doc)
        assert count == 3
        assert index._collection.count() == 3

    def test_empty_document_indexes_nothing(self, index):
        empty_doc = doc("empty_pid", [])
        assert index.index_document(empty_doc) == 0


class TestExactLookup:
    def test_finds_verbatim_tag(self, index, sample_doc):
        index.index_document(sample_doc)
        results = index.exact_lookup("What is 26-KA-901 rated for?")
        assert len(results) == 1
        assert results[0].text == "26-KA-901"
        assert results[0].source == "exact"
        assert results[0].score == 1.0

    def test_no_tag_tokens_returns_empty(self, index, sample_doc):
        index.index_document(sample_doc)
        assert index.exact_lookup("What is the alarm setpoint?") == []

    def test_no_match_for_unindexed_tag(self, index, sample_doc):
        index.index_document(sample_doc)
        assert index.exact_lookup("What about 99-ZZ-000?") == []

    def test_carries_revision_label_and_pid(self, index, sample_doc):
        index.index_document(sample_doc)
        results = index.exact_lookup("26-KA-901")
        assert results[0].pid == "26-KA-901_A"
        assert results[0].revision_label == "Rev A"

    def test_finds_note_by_number_reference(self, index, doc_with_note):
        # Real bug found via a live eval run (2026-07-25): "37" alone is
        # filtered out by _extract_tag_tokens (too short), and even if it
        # weren't, it would never equal a whole note's full-sentence text --
        # this must go through the note_number metadata path instead, not
        # the tag-token path.
        index.index_document(doc_with_note)
        results = index.exact_lookup("What does note 37 say?")
        assert len(results) == 1
        assert results[0].text == "37. PSV 9066A/B SET PRESSURE REVISED TO 260 BAR(G)."
        assert results[0].source == "exact"

    def test_note_lookup_ignores_non_matching_number(self, index, doc_with_note):
        index.index_document(doc_with_note)
        assert index.exact_lookup("What does note 99 say?") == []

    def test_elements_without_a_note_number_are_not_matched_by_note_lookup(self, index, sample_doc):
        # sample_doc's note element has no note_number attribute at all --
        # asking about "note 1" must not spuriously match it via an
        # empty-string-equals-empty-string coincidence.
        index.index_document(sample_doc)
        assert index.exact_lookup("What does note 1 say?") == []


class TestVectorSearch:
    def test_ranks_by_similarity(self, index, sample_doc):
        index.index_document(sample_doc)
        results = index.vector_search("alarm setpoint question", top_k=3)
        assert results[0].text == "Compressor discharge pressure alarm high high"
        assert results[0].source == "vector"

    def test_empty_collection_returns_empty_without_crashing(self, index):
        assert index.vector_search("anything", top_k=5) == []

    def test_respects_top_k(self, index, sample_doc):
        index.index_document(sample_doc)
        results = index.vector_search("alarm setpoint question", top_k=1)
        assert len(results) == 1


class TestHybridSearch:
    def test_merges_exact_and_vector_results(self, index, sample_doc):
        index.index_document(sample_doc)
        results = index.hybrid_search("26-KA-901 tag lookup", top_k=3)
        ids = {r.element_id for r in results}
        # exact hit for the tag, plus vector neighbors -- all present, no duplicates
        assert len(ids) == len(results)
        assert any(r.source == "exact" for r in results)

    def test_exact_hit_not_duplicated_if_also_a_vector_neighbor(self, index, sample_doc):
        index.index_document(sample_doc)
        results = index.hybrid_search("26-KA-901 tag lookup", top_k=3)
        tag_hits = [r for r in results if r.text == "26-KA-901"]
        assert len(tag_hits) == 1
        assert tag_hits[0].source == "exact"  # exact wins the dedup, not vector

    def test_no_matches_returns_empty(self, index):
        assert index.hybrid_search("completely unrelated query", top_k=3) == []


class TestIndexDelta:
    @pytest.fixture
    def index(self, sample_report):
        # Overrides the module-level `index` fixture: index_delta() embeds
        # its own synthesized sentences (never a raw element text), so the
        # strict dict-keyed FakeEmbedder needs vectors keyed by whatever
        # _delta_chunk_text() actually produces for this fixture's deltas --
        # computed here, not duplicated as hard-coded strings, so this stays
        # correct if that format ever changes.
        basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        eligible = [
            d
            for d in sample_report.deltas
            if (d.old_text or "").strip() or (d.new_text or "").strip()
        ]
        texts = {
            _delta_chunk_text(d, sample_report.pid_a, sample_report.pid_b): basis[i]
            for i, d in enumerate(eligible)
        }
        return ChatIndex(
            collection_name=f"test_{uuid.uuid4().hex}",
            embedder=FakeEmbedder(texts),
            client=chromadb.EphemeralClient(),
        )

    def test_indexes_text_bearing_deltas_only(self, index, sample_report):
        # 4 deltas in the fixture, but the pure-geometry move has no text on
        # either side -- nothing meaningful to phrase or search, must be skipped.
        count = index.index_delta(sample_report)
        assert count == 3

    def test_reindexing_is_idempotent(self, index, sample_report):
        index.index_delta(sample_report)
        count = index.index_delta(sample_report)
        assert count == 3
        assert index._collection.count() == 3

    def test_empty_deltas_indexes_nothing(self, index):
        empty_report = DeltaReport(pid_a="a", pid_b="b", deltas=[], stats=_stats())
        assert index.index_delta(empty_report) == 0

    def test_modified_chunk_names_both_pids_and_both_values(self, index, sample_report):
        index.index_delta(sample_report)
        got = index._collection.get(include=["documents"])
        modified_text = next(t for t in got["documents"] if t.startswith("CHANGED"))
        assert "SP = 257 bar (g)" in modified_text
        assert "SP = 260 bar (g)" in modified_text
        assert "pair1_A" in modified_text
        assert "pair1_B" in modified_text

    def test_added_chunk_is_labeled(self, index, sample_report):
        index.index_delta(sample_report)
        got = index._collection.get(include=["documents"])
        assert any(t.startswith("ADDED") and "NEW NOTE ADDED" in t for t in got["documents"])

    def test_removed_chunk_is_labeled(self, index, sample_report):
        index.index_delta(sample_report)
        got = index._collection.get(include=["documents"])
        assert any(t.startswith("REMOVED") and "OLD NOTE REMOVED" in t for t in got["documents"])

    def test_metadata_carries_element_type_and_page(self, index, sample_report):
        index.index_delta(sample_report)
        got = index._collection.get(include=["documents", "metadatas"])
        modified_meta = next(
            m
            for t, m in zip(got["documents"], got["metadatas"], strict=True)
            if t.startswith("CHANGED")
        )
        assert modified_meta["type"] == "setpoint"
        assert modified_meta["page"] == 0
        added_meta = next(
            m
            for t, m in zip(got["documents"], got["metadatas"], strict=True)
            if t.startswith("ADDED")
        )
        assert added_meta["page"] == 1  # from new_bbox -- added has no old_bbox

    def test_delta_chunks_retrievable_via_vector_search(self, index, sample_report):
        # Uses the real _delta_chunk_text() to compute the exact synthesized
        # sentence rather than duplicating its format as a hard-coded string,
        # so this test breaks if the two ever drift apart. `index` here is
        # this class's own fixture override (see above), which already
        # supplies a distinct basis vector per eligible delta.
        modified = sample_report.deltas[0]
        modified_text = _delta_chunk_text(modified, sample_report.pid_a, sample_report.pid_b)
        index._embedder._vectors["what changed about the setpoint"] = [0.9, 0.0, 0.1]
        index.index_delta(sample_report)
        results = index.vector_search("what changed about the setpoint", top_k=1)
        assert results[0].text == modified_text
        assert results[0].source == "vector"

    def test_delta_chunks_are_not_matched_by_exact_lookup(self, index, sample_report):
        # By design (see index_delta's docstring): a delta chunk is a full
        # sentence, never a bare tag, so it must never surface via the
        # tag-token equality path -- only vector search should ever find it.
        index.index_delta(sample_report)
        assert index.exact_lookup("What changed for pair1_A pair1_B?") == []
