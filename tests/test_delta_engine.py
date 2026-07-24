"""Engine-level tests: classification (added/removed/modified/unchanged),
confidence scoring, the low-alignment-rate warning, and the mandatory proof
that the classifier path never invokes an LLM (BRIEF rule 2).
"""

import pytest

from src.canonical.model import CanonicalDocument
from src.config import settings
from src.delta.engine import compute_delta
from tests.conftest import FakeEmbedder
from tests.conftest import make_positioned_element as el


def doc(pid: str, elements, fmt="pdf_native") -> CanonicalDocument:
    return CanonicalDocument(
        pid=pid, format=fmt, revision_label=None, page_count=1,
        elements=elements, raw_source_path=f"{pid}.pdf",
    )  # fmt: skip


class TestClassification:
    def test_unchanged_matched_pair_produces_no_delta(self):
        a = el("26BL9072")
        b = el("26BL9072")
        report = compute_delta(doc("A", [a]), doc("B", [b]), embedder=FakeEmbedder({}))
        assert report.deltas == []
        assert report.stats.unchanged == 1
        assert report.stats.matched == 1

    def test_tier1_matches_are_never_reported_as_modified(self):
        """Architectural invariant: for tier-1 types the key IS the full
        text, so a tier-1 match implies identical text by construction --
        there is no way to reach `compute_delta` with a tier-1 pair whose
        text differs. A text change on a keyed tag instead shows up as a
        failed match (different key -> separate removed + added), never a
        false "unchanged" tier-1 pair. Positioned far apart and with
        dissimilar embeddings so tier 3 doesn't rescue them either -- this
        test is specifically about tier 1's own invariant."""
        a = el("26-HA-911", element_type="tag", x0=0.05, y0=0.05, x1=0.10, y1=0.06)
        b = el("26-HA-912", element_type="tag", x0=0.85, y0=0.85, x1=0.90, y1=0.86)
        embedder = FakeEmbedder({"26-HA-911": [1.0, 0.0], "26-HA-912": [0.0, 1.0]})
        report = compute_delta(doc("A", [a]), doc("B", [b]), embedder=embedder)
        assert report.stats.matched_by_tier["exact_key"] == 0
        assert report.stats.added == 1
        assert report.stats.removed == 1
        assert report.stats.modified == 0

    def test_moved_geometry_reports_as_modified_not_unchanged(self):
        """Geometry elements carry no text (plan §3) -- 'modified' has to
        mean 'moved', not 'text changed'. Pair 3's GT3-MOVE regression: a
        matched geometry pair whose bbox differs must not be silently
        dropped as unchanged just because both sides have text=''."""
        a = el("", element_type="geometry", x0=0.40, y0=0.30, x1=0.42, y1=0.32,
               layer="PIPING", entity_type="INSERT", block_name="VALVE_GATE")
        b = el("", element_type="geometry", x0=0.44, y0=0.34, x1=0.46, y1=0.36,
               layer="PIPING", entity_type="INSERT", block_name="VALVE_GATE")
        report = compute_delta(doc("A", [a]), doc("B", [b]), embedder=FakeEmbedder({}))
        assert report.stats.matched_by_tier["geometry"] == 1
        assert report.stats.modified == 1
        assert report.stats.unchanged == 0
        d = report.deltas[0]
        assert d.change_type == "modified"
        assert d.match_tier == "geometry"

    def test_unmoved_geometry_reports_as_unchanged(self):
        a = el("", element_type="geometry", x0=0.40, y0=0.30, x1=0.42, y1=0.32,
               layer="PIPING", entity_type="LINE")
        b = el("", element_type="geometry", x0=0.40, y0=0.30, x1=0.42, y1=0.32,
               layer="PIPING", entity_type="LINE")
        report = compute_delta(doc("A", [a]), doc("B", [b]), embedder=FakeEmbedder({}))
        assert report.stats.matched_by_tier["geometry"] == 1
        assert report.stats.modified == 0
        assert report.stats.unchanged == 1
        assert report.deltas == []

    def test_setpoint_modification_via_tier3(self):
        a = el("SP = 257 bar (g)", element_type="setpoint", x0=0.10, y0=0.10, x1=0.20, y1=0.11)
        b = el("SP = 260 bar (g)", element_type="setpoint", x0=0.10, y0=0.10, x1=0.20, y1=0.11)
        embedder = FakeEmbedder({"SP = 257 bar (g)": [1.0, 0.0], "SP = 260 bar (g)": [0.95, 0.312]})
        report = compute_delta(doc("A", [a]), doc("B", [b]), embedder=embedder)
        assert len(report.deltas) == 1
        d = report.deltas[0]
        assert d.change_type == "modified"
        assert d.old_text == "SP = 257 bar (g)"
        assert d.new_text == "SP = 260 bar (g)"
        assert d.match_tier == "embedding_proximity"
        assert d.element_type == "setpoint"

    def test_unmatched_a_is_removed_unmatched_b_is_added(self):
        removed_el = el("OLD-TAG", element_type="tag", x0=0.05, y0=0.05, x1=0.10, y1=0.06)
        added_el = el("NEW-TAG", element_type="tag", x0=0.85, y0=0.85, x1=0.90, y1=0.86)
        embedder = FakeEmbedder({"OLD-TAG": [1.0, 0.0], "NEW-TAG": [0.0, 1.0]})
        report = compute_delta(doc("A", [removed_el]), doc("B", [added_el]), embedder=embedder)
        assert report.stats.added == 1
        assert report.stats.removed == 1
        assert report.stats.modified == 0
        types = {d.change_type for d in report.deltas}
        assert types == {"added", "removed"}

    def test_deterministic_ordering_across_repeated_runs(self):
        """Elements deliberately share a position -- every candidate ties on
        distance, so this exercises the id-based tie-break in the greedy
        assignment, not just "there happens to be one obvious pairing"."""
        a1, a2 = el("AAA-1", element_type="tag"), el("BBB-2", element_type="tag")
        b1, b2 = el("CCC-3", element_type="tag"), el("DDD-4", element_type="tag")
        vectors = {
            "AAA-1": [1, 0, 0, 0], "BBB-2": [0, 1, 0, 0],
            "CCC-3": [0, 0, 1, 0], "DDD-4": [0, 0, 0, 1],
        }  # fmt: skip
        d1 = compute_delta(doc("A", [a1, a2]), doc("B", [b1, b2]), embedder=FakeEmbedder(vectors))
        d2 = compute_delta(doc("A", [a1, a2]), doc("B", [b1, b2]), embedder=FakeEmbedder(vectors))
        assert [x.model_dump() for x in d1.deltas] == [x.model_dump() for x in d2.deltas]


class TestConfidence:
    def test_removed_confidence_equals_element_extraction_confidence(self):
        a = el("OLD-TAG", element_type="tag", conf=0.42)
        report = compute_delta(doc("A", [a]), doc("B", []), embedder=FakeEmbedder({}))
        assert report.deltas[0].confidence == pytest.approx(0.42)

    def test_modified_confidence_is_bounded_by_lower_extraction_confidence(self):
        a = el("SP = 257 bar (g)", element_type="setpoint", conf=0.5,
               x0=0.10, y0=0.10, x1=0.20, y1=0.11)
        b = el("SP = 260 bar (g)", element_type="setpoint", conf=1.0,
               x0=0.10, y0=0.10, x1=0.20, y1=0.11)
        embedder = FakeEmbedder({"SP = 257 bar (g)": [1.0, 0.0], "SP = 260 bar (g)": [1.0, 0.0]})
        report = compute_delta(doc("A", [a]), doc("B", [b]), embedder=embedder)
        # tight-proximity override -> match_score=1.0; confidence = min(0.5,1.0)*1.0
        assert report.deltas[0].confidence == pytest.approx(0.5)


class TestLowAlignmentWarning:
    def test_fires_on_mismatched_documents(self):
        """Simulates Pair 4: two documents with no shared keys, clustered in
        clearly separate regions of the page and with orthogonal embeddings,
        so nothing coincidentally aligns via any tier."""
        a_elements = [
            el(f"A-TAG-{i}", element_type="tag", x0=0.01 * i, y0=0.05, x1=0.01 * i + 0.01, y1=0.06)
            for i in range(25)
        ]
        b_elements = [
            el(f"B-TAG-{i}", element_type="tag", x0=0.01 * i, y0=0.95, x1=0.01 * i + 0.01, y1=0.96)
            for i in range(25)
        ]
        vectors = {f"A-TAG-{i}": [1.0, 0.0] for i in range(25)}
        vectors.update({f"B-TAG-{i}": [0.0, 1.0] for i in range(25)})
        report = compute_delta(
            doc("pair4_A", a_elements), doc("pair4_B", b_elements), embedder=FakeEmbedder(vectors)
        )
        assert report.stats.exact_key_rate < settings.low_alignment_rate_threshold
        assert any("alignment" in w.lower() for w in report.warnings)

    def test_does_not_fire_on_well_aligned_documents(self):
        shared = [el(f"TAG-{i}", element_type="tag") for i in range(25)]
        report = compute_delta(doc("A", shared), doc("B", shared), embedder=FakeEmbedder({}))
        assert report.warnings == []

    def test_does_not_fire_below_minimum_element_count(self):
        """A tiny hand-built test doc with a genuinely low alignment rate
        shouldn't trigger the warning -- too small a sample to be meaningful."""
        a_elements = [el("ONLY-TAG", element_type="tag")]
        b_elements = [el("DIFFERENT-TAG", element_type="tag")]
        embedder = FakeEmbedder({"ONLY-TAG": [1.0, 0.0], "DIFFERENT-TAG": [0.0, 1.0]})
        report = compute_delta(doc("A", a_elements), doc("B", b_elements), embedder=embedder)
        assert report.warnings == []


class TestLLMNeverCalled:
    """BRIEF rule 2: delta alignment and classification stay deterministic,
    non-LLM logic. This test proves it at runtime rather than documenting it:
    every code path an LLM client could be reached through is patched to
    raise, and a realistic multi-tier delta run must still complete clean.
    """

    def test_classifier_never_invokes_llm_client(self, monkeypatch):
        import anthropic
        import openai

        from src.chat.llm import LLMClient

        def boom(*_args, **_kwargs):
            raise AssertionError("LLM client must not be invoked during delta computation")

        monkeypatch.setattr(LLMClient, "complete", boom)
        monkeypatch.setattr(LLMClient, "read_image_text", boom)
        monkeypatch.setattr(anthropic, "Anthropic", boom)
        monkeypatch.setattr(openai, "OpenAI", boom)

        # A realistic mixed scenario exercising all three tiers plus
        # add/remove/modify/unchanged classification, using a fake embedder
        # (the embedding model itself is local/deterministic, never the LLM
        # client -- this test's job is to prove the LLM specifically is
        # never reached, not to avoid the embedder). Positions are chosen so
        # the removed/added notes are far from everything else and from
        # each other -- otherwise tier 3's tight-proximity override would
        # spuriously pair them.
        a_docs = [
            el("26BL9072", element_type="valve"),  # tier1, unchanged
            el("30. SOME NOTE", element_type="note", x0=0.05, y0=0.20, x1=0.10, y1=0.21,
               kind="definition", note_number="30"),  # tier1 key fails to find a B match -> removed
            el("", element_type="geometry", x0=0.4, y0=0.4, x1=0.42, y1=0.42,
               layer="PIPING", entity_type="LINE"),  # tier2
            el("19057", element_type="text_block", x0=0.1, y0=0.6, x1=0.15, y1=0.61),  # tier3
        ]
        b_docs = [
            el("26BL9072", element_type="valve"),
            el("", element_type="geometry", x0=0.41, y0=0.41, x1=0.43, y1=0.43,
               layer="PIPING", entity_type="LINE"),
            el("20500", element_type="text_block", x0=0.1, y0=0.6, x1=0.14, y1=0.61),
            el("31. NEW NOTE", element_type="note", x0=0.90, y0=0.80, x1=0.95, y1=0.81,
               kind="definition", note_number="31"),  # tier1 key fails to find an A match -> added
        ]
        embedder = FakeEmbedder(
            {
                "19057": [1.0, 0.0, 0.0, 0.0],
                "20500": [0.1, 0.995, 0.0, 0.0],
                "30. SOME NOTE": [0.0, 0.0, 1.0, 0.0],
                "31. NEW NOTE": [0.0, 0.0, 0.0, 1.0],
            }
        )

        report = compute_delta(doc("A", a_docs), doc("B", b_docs), embedder=embedder)

        # Completed without hitting any patched call, and produced a sane result.
        assert report.stats.total_a == 4
        assert report.stats.total_b == 4
        change_types = {d.change_type for d in report.deltas}
        assert "removed" in change_types
        assert "added" in change_types
