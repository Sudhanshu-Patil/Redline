"""Validate the committed sample pairs: manifests/ground truth parse against
the schema, referenced files exist, and Pair 1's B actually contains the
ground-truth edits (and nothing left over from the redactions).
"""

from pathlib import Path

import fitz
import pytest

from eval.schema import GroundTruth, load_ground_truth, load_manifest
from src.ingest.pdf_native import PdfNativeAdapter

SAMPLES = Path("data/samples")
PAIR_IDS = ["pair1", "pair2", "pair4"]


@pytest.fixture(scope="module", params=PAIR_IDS)
def pair(request):
    pair_dir = SAMPLES / request.param
    manifest = load_manifest(pair_dir / "manifest.json")
    ground_truth = load_ground_truth(Path(manifest.ground_truth_path))
    return manifest, ground_truth


def test_manifest_and_ground_truth_are_consistent(pair):
    manifest, ground_truth = pair
    assert manifest.pair_id == ground_truth.pair_id
    assert Path(manifest.doc_a.path).exists()
    assert Path(manifest.doc_b.path).exists()


def test_ground_truth_deltas_are_well_formed(pair):
    _, ground_truth = pair
    for delta in ground_truth.expected_deltas:
        if delta.change_type == "added":
            assert delta.old_text is None and delta.new_text
        elif delta.change_type == "removed":
            assert delta.old_text and delta.new_text is None
        else:
            assert delta.old_text and delta.new_text


@pytest.fixture(scope="module")
def b_texts() -> str:
    manifest = load_manifest(SAMPLES / "pair1" / "manifest.json")
    doc = PdfNativeAdapter().ingest(Path(manifest.doc_b.path), pid="test_pair1_B")
    return "\n".join(e.text for e in doc.elements)


@pytest.fixture(scope="module")
def pair1_ground_truth() -> GroundTruth:
    return load_ground_truth(SAMPLES / "pair1" / "ground_truth.json")


class TestPair1:
    def test_has_the_six_documented_edits(self, pair1_ground_truth):
        assert len(pair1_ground_truth.expected_deltas) == 6
        by_type = {d.change_type for d in pair1_ground_truth.expected_deltas}
        assert by_type == {"added", "removed", "modified"}

    def test_new_values_present_in_b(self, b_texts):
        assert b_texts.count("SP = 260 bar (g)") == 2
        assert "HH: 250" in b_texts
        assert "20500" in b_texts
        assert "37. PSV 9066A/B SET PRESSURE REVISED TO 260 BAR(G)." in b_texts

    def test_old_values_absent_in_b(self, b_texts):
        assert "SP = 257 bar (g)" not in b_texts
        assert "HH: 245" not in b_texts
        assert "19057" not in b_texts
        assert "SAFETY CRITICAL HEAT TRACING" not in b_texts
        assert "HYDRATE MITIGATION" not in b_texts  # full line removed, not just the head

    def test_neighbouring_notes_survived_redaction(self, b_texts):
        assert "POWER AT COMPRESSOR COUPLING" in b_texts  # note 28
        assert "CASE 8A" in b_texts  # note 29
        assert "LL SET POINT IS OVERRIDEN" in b_texts  # note 31

    def test_dangling_note30_callouts_remain(self, b_texts):
        """The definition was deleted; the drawing's NOTE 30 references stay
        (documented in ground truth notes as intentionally unchanged)."""
        assert "NOTE 30" in b_texts


class TestPair2:
    def test_b_is_image_only(self):
        manifest = load_manifest(SAMPLES / "pair2" / "manifest.json")
        doc = fitz.open(manifest.doc_b.path)
        assert doc.page_count == 1
        assert doc[0].get_text().strip() == ""
        assert manifest.doc_b.format == "pdf_scanned"

    def test_ground_truth_matches_pair1s(self):
        gt1 = load_ground_truth(SAMPLES / "pair1" / "ground_truth.json")
        gt2 = load_ground_truth(SAMPLES / "pair2" / "ground_truth.json")
        assert [d.gt_id for d in gt2.expected_deltas] == [d.gt_id for d in gt1.expected_deltas]


class TestPair4:
    def test_is_negative_control_with_no_expected_deltas(self):
        gt = load_ground_truth(SAMPLES / "pair4" / "ground_truth.json")
        assert gt.negative_control is True
        assert gt.expected_deltas == []

    def test_references_the_two_original_documents(self):
        manifest = load_manifest(SAMPLES / "pair4" / "manifest.json")
        assert "26-KA-901" in manifest.doc_a.path
        assert "26-KA-902" in manifest.doc_b.path
