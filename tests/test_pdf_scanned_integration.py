"""Integration: real tesseract OCR over the real Pair 2 scanned PDF.

Skipped automatically when tesseract isn't installed (CI installs it).
The vision fallback is disabled here -- integration tests must not require
network or an API key; fallback behavior is covered by unit tests.
"""

from pathlib import Path

import pytest

from src.canonical.model import CanonicalDocument
from src.ingest.pdf_scanned import PdfScannedAdapter, tesseract_available

SCANNED = Path("data/samples/pair2/B_scanned.pdf")

pytestmark = pytest.mark.skipif(
    not tesseract_available() or not SCANNED.exists(),
    reason="tesseract or sample scanned PDF not available",
)


class NoVision:
    @property
    def is_configured(self) -> bool:
        return False

    def read_image_text(self, png_bytes: bytes, context_hint: str = "") -> str:
        raise AssertionError("must not be called")


@pytest.fixture(scope="module")
def doc() -> CanonicalDocument:
    return PdfScannedAdapter(vision_client=NoVision()).ingest(SCANNED, pid="pair2_B_test")


def test_extracts_a_substantial_number_of_elements(doc):
    assert len(doc.elements) > 300


def test_confidences_are_in_unit_interval(doc):
    assert all(0.0 <= e.extraction_confidence <= 1.0 for e in doc.elements)


def test_known_strong_text_is_found(doc):
    """Large-type strings from the drawing should survive OCR verbatim."""
    joined = "\n".join(e.text for e in doc.elements)
    assert "COMPRESSOR" in joined
    assert "9066A" in joined or "PSV" in joined


def test_bboxes_are_in_pdf_point_space(doc):
    for e in doc.elements:
        assert e.bbox.unit == "pdf_points"
        assert e.bbox.page_width == pytest.approx(1191.0, abs=2.0)
        x0, y0, x1, y1 = e.bbox.normalized
        assert -0.01 <= x0 <= 1.01 and -0.01 <= y1 <= 1.01


def test_unconfigured_fallback_marks_low_confidence_elements(doc):
    low_conf = [e for e in doc.elements if e.attributes.get("ocr_fallback") == "unavailable"]
    assert low_conf, "expected at least some low-confidence regions flagged on a dense P&ID"


def test_instrument_bubbles_cluster_across_ocr_noise(doc):
    """Spatial clustering + noise trim must recover instrument loops that
    tesseract's sparse mode splits apart (the Phase 3 clustering fix)."""
    loops = [e for e in doc.elements if e.type == "instrument_loop"]
    assert len(loops) >= 10
    assert any(e.text == "PIT-9062" for e in loops), "ground-truth bubble PIT-9062 missing"


def test_edited_hh_setpoint_classified_as_setpoint(doc):
    """Pair 2 ground truth: the HH: 250 edit must come through OCR as a
    typed setpoint element, not a generic text_block."""
    hh = [
        e
        for e in doc.elements
        if e.type == "setpoint" and e.attributes.get("value") == "250"
    ]
    assert hh, "HH: 250 not classified as setpoint in scanned output"


def test_ingest_emits_trace_spans(tmp_path, monkeypatch):
    """Observability ground rule: the OCR path must emit its span tree."""
    import json

    from src.observability import tracing

    monkeypatch.setattr(tracing.settings, "traces_dir", tmp_path)
    with tracing.trace("test_scanned_ingest") as trace_id:
        PdfScannedAdapter(vision_client=NoVision()).ingest(SCANNED, pid="span_check")
    records = [
        json.loads(line)
        for line in (tmp_path / f"{trace_id}.jsonl").read_text().splitlines()
    ]
    names = {r["name"] for r in records}
    assert {"pdf_scanned.ingest", "pdf_scanned.render", "pdf_scanned.tesseract"} <= names
    ingest_span = next(r for r in records if r["name"] == "pdf_scanned.ingest")
    assert ingest_span["attributes"]["elements_extracted"] > 0


def test_cross_adapter_consistency_with_native_reading_of_same_revision(doc):
    """Pair 1 B (native) and Pair 2 B (scanned) are the SAME revision.
    A floor on exact-text agreement between the two adapters keeps the
    cross-format matching surface honest -- if this drops, Phase 5's
    exact-key alignment quietly starves.

    Floors are deliberately conservative (OCR is lossy); the point is to
    catch regressions, not to flatter the OCR.
    """
    from difflib import SequenceMatcher

    from src.ingest.pdf_native import PdfNativeAdapter

    native = PdfNativeAdapter().ingest(Path("data/samples/pair1/B.pdf"), pid="consistency_A")
    scanned_texts = [e.text for e in doc.elements]
    scanned_set = set(scanned_texts)

    # Short keyed identifiers must survive OCR *verbatim* often enough for
    # exact-key alignment to have raw material. Floors absorb tesseract
    # version variance across environments (measured locally: >=50%).
    for element_type, floor in [("valve", 0.4), ("line_number", 0.4)]:
        native_texts = {e.text for e in native.elements if e.type == element_type}
        hits = sum(1 for t in native_texts if t in scanned_set)
        rate = hits / len(native_texts)
        assert rate >= floor, (
            f"{element_type}: only {hits}/{len(native_texts)} ({rate:.0%}) of native "
            f"texts found verbatim in scanned output (floor {floor:.0%})"
        )

    # Notes are prose: a single OCR character error breaks exact equality, so
    # measure them the way the delta engine will match them -- by similarity.
    # Measured locally: 80% of notes reach ratio >=0.90; floor set well below.
    native_notes = [e.text for e in native.elements if e.type == "note"]
    fuzzy_hits = sum(
        1
        for note in native_notes
        if max(SequenceMatcher(None, note, t).ratio() for t in scanned_texts) >= 0.85
    )
    rate = fuzzy_hits / len(native_notes)
    assert rate >= 0.6, (
        f"notes: only {fuzzy_hits}/{len(native_notes)} ({rate:.0%}) reached "
        "similarity 0.85 against scanned output (floor 60%)"
    )
