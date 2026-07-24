"""Integration: run the real delta engine (real adapters, real embedding
model) against the committed sample pairs and check it finds the documented
ground truth. Complements test_delta_align.py / test_delta_engine.py's fast
hand-built-fixture tests -- this is the "does it actually work on real data"
proof appropriate to Phase 5 (full P/R/F1 scorecard machinery is Phase 10).
"""

from pathlib import Path

import pytest

from eval.schema import load_ground_truth, load_manifest
from src.delta.engine import compute_delta
from src.ingest.dwg import DwgAdapter
from src.ingest.pdf_native import PdfNativeAdapter

PAIR1 = Path("data/samples/pair1")
PAIR3 = Path("data/samples/pair3")
PAIR4 = Path("data/samples/pair4")


def _has_modified(deltas, old_text=None, new_text=None, element_type=None, min_count=1):
    matches = [
        d
        for d in deltas
        if d.change_type == "modified"
        and (old_text is None or d.old_text == old_text)
        and (new_text is None or d.new_text == new_text)
        and (element_type is None or d.element_type == element_type)
    ]
    return len(matches) >= min_count


def _has_removed(deltas, old_text_contains):
    return any(
        d.change_type == "removed" and d.old_text and old_text_contains in d.old_text
        for d in deltas
    )


def _has_added(deltas, new_text_contains):
    return any(
        d.change_type == "added" and d.new_text and new_text_contains in d.new_text
        for d in deltas
    )


@pytest.fixture(scope="module")
def pair1_report():
    manifest = load_manifest(PAIR1 / "manifest.json")
    doc_a = PdfNativeAdapter().ingest(Path(manifest.doc_a.path), pid="pair1_A_it")
    doc_b = PdfNativeAdapter().ingest(Path(manifest.doc_b.path), pid="pair1_B_it")
    return compute_delta(doc_a, doc_b)


@pytest.fixture(scope="module")
def pair1_ground_truth():
    return load_ground_truth(PAIR1 / "ground_truth.json")


@pytest.fixture(scope="module")
def pair3_report():
    manifest = load_manifest(PAIR3 / "manifest.json")
    doc_a = DwgAdapter().ingest(Path(manifest.doc_a.path), pid="pair3_A_it")
    doc_b = DwgAdapter().ingest(Path(manifest.doc_b.path), pid="pair3_B_it")
    return compute_delta(doc_a, doc_b)


pytestmark_pair1 = pytest.mark.skipif(not PAIR1.exists(), reason="pair1 not present (make data)")
pytestmark_pair3 = pytest.mark.skipif(not PAIR3.exists(), reason="pair3 not present (make data)")
pytestmark_pair4 = pytest.mark.skipif(not PAIR4.exists(), reason="pair4 not present (make data)")


@pytestmark_pair1
class TestPair1GroundTruth:
    def test_report_has_exactly_six_documented_ground_truth_deltas(self, pair1_ground_truth):
        assert len(pair1_ground_truth.expected_deltas) == 6

    def test_finds_both_psv_setpoint_modifications(self, pair1_report):
        """GT-SP-9066A + GT-SP-9066B: two identical-text setpoint edits,
        disambiguated purely by position -- the engine's core duplicate-key
        handling proven on real data, not a hand-built fixture."""
        assert _has_modified(
            pair1_report.deltas, old_text="SP = 257 bar (g)", new_text="SP = 260 bar (g)",
            element_type="setpoint", min_count=2,
        )

    def test_finds_hh_setpoint_modification(self, pair1_report):
        assert _has_modified(
            pair1_report.deltas, old_text="HH: 245", new_text="HH: 250", element_type="setpoint"
        )

    def test_finds_flow_rate_modification(self, pair1_report):
        """GT-FLOW: no stable key, caught only via tier-3 proximity -- the
        text is 'modified', not shown as an unrelated remove+add pair."""
        assert any(
            d.change_type == "modified" and d.new_text == "20500" for d in pair1_report.deltas
        )

    def test_finds_note30_removed(self, pair1_report):
        assert _has_removed(pair1_report.deltas, "SAFETY CRITICAL HEAT TRACING")

    def test_finds_note37_added(self, pair1_report):
        assert _has_added(pair1_report.deltas, "PSV 9066A/B SET PRESSURE REVISED")

    def test_exact_key_rate_is_high_for_a_genuine_revision(self, pair1_report):
        assert pair1_report.stats.exact_key_rate > 0.9
        assert pair1_report.warnings == []

    def test_no_llm_client_touched_on_real_documents(self, monkeypatch):
        """The full proof (BRIEF rule 2), repeated end-to-end on real
        adapter output rather than hand-built elements."""
        import anthropic
        import openai

        from src.chat.llm import LLMClient

        def boom(*_a, **_k):
            raise AssertionError("LLM client must not be invoked during delta computation")

        monkeypatch.setattr(LLMClient, "complete", boom)
        monkeypatch.setattr(LLMClient, "read_image_text", boom)
        monkeypatch.setattr(anthropic, "Anthropic", boom)
        monkeypatch.setattr(openai, "OpenAI", boom)

        manifest = load_manifest(PAIR1 / "manifest.json")
        doc_a = PdfNativeAdapter().ingest(Path(manifest.doc_a.path), pid="pair1_A_llmcheck")
        doc_b = PdfNativeAdapter().ingest(Path(manifest.doc_b.path), pid="pair1_B_llmcheck")
        report = compute_delta(doc_a, doc_b)
        assert len(report.deltas) > 0


@pytestmark_pair3
class TestPair3GroundTruth:
    def test_setpoint_modification(self, pair3_report):
        assert _has_modified(
            pair3_report.deltas, old_text="SP = 257 bar (g)", new_text="SP = 260 bar (g)"
        )

    def test_dimension_modification_via_tier3(self, pair3_report):
        """GT3-DIM: a DWG dimension's measured value changes -- same 'value
        changed, key can't be the value itself' pattern as GT-FLOW, proven
        on the geometry-rich DWG format this time."""
        assert _has_modified(pair3_report.deltas, old_text="600", new_text="750")

    def test_moved_valve_caught_by_geometry_tier(self, pair3_report):
        """GT3-MOVE: the plan §4.2 showcase -- same layer/entity/block_name,
        bbox moved, no text key at all (geometry elements carry no text)."""
        geometry_modified = [
            d
            for d in pair3_report.deltas
            if d.change_type == "modified" and d.match_tier == "geometry"
        ]
        assert geometry_modified, "expected at least one geometry-tier modification (moved valve)"

    def test_added_valve_and_removed_drain_line(self, pair3_report):
        added_valve_label = any(
            d.change_type == "added" and d.new_text == "43BL9020" for d in pair3_report.deltas
        )
        assert added_valve_label
        assert pair3_report.stats.removed >= 1


@pytestmark_pair4
class TestPair4NegativeControl:
    """Two different compressors' P&IDs, presented as if a revision pair.
    The real acceptance criterion (plan §6): a low-alignment warning, not a
    dump of hundreds of spurious deltas."""

    def test_low_exact_key_rate_warning_fires_on_real_mismatched_documents(self):
        manifest = load_manifest(PAIR4 / "manifest.json")
        doc_a = PdfNativeAdapter().ingest(Path(manifest.doc_a.path), pid="pair4_A_it")
        doc_b = PdfNativeAdapter().ingest(Path(manifest.doc_b.path), pid="pair4_B_it")
        report = compute_delta(doc_a, doc_b)
        assert report.warnings, "expected a low-alignment warning for unrelated documents"
        assert report.stats.exact_key_rate < 0.5
