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
