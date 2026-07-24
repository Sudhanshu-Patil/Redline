"""Integration test: run the real adapter against the real sample PDFs.

Complements test_pdf_native.py's pure classify_block_lines unit tests by
confirming pymupdf's actual block/line extraction feeds the classifier
correctly on real, messy P&ID text -- and pins down specific values known
to exist in the source document (grounds this as a regression guard, not
just a smoke test).
"""

from pathlib import Path

import pytest

from src.canonical.model import CanonicalDocument
from src.ingest.pdf_native import PdfNativeAdapter

SAMPLE = Path("data/samples/originals/lift_gas_compressor_26-KA-901.pdf")

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not present")


@pytest.fixture(scope="module")
def doc() -> CanonicalDocument:
    return PdfNativeAdapter().ingest(SAMPLE, pid="lift_gas_A", revision_label="as-provided")


def test_produces_a_substantial_number_of_elements(doc: CanonicalDocument):
    assert len(doc.elements) > 500


def test_all_elements_have_exact_extraction_confidence(doc: CanonicalDocument):
    assert all(e.extraction_confidence == 1.0 for e in doc.elements)


def test_element_ids_are_unique(doc: CanonicalDocument):
    ids = [e.id for e in doc.elements]
    assert len(ids) == len(set(ids))


def test_covers_a_variety_of_element_types(doc: CanonicalDocument):
    types_present = {e.type for e in doc.elements}
    assert {"note", "line_number", "valve", "tag", "setpoint", "instrument_loop", "text_block"} <= (
        types_present
    )


def test_finds_the_psv_9066_setpoint_value(doc: CanonicalDocument):
    """Ground truth: PSV 9066A/B are set at 257 bar(g) in the as-provided drawing
    (this is the exact value Pair 1's synthetic edit, plan §6, will later change)."""
    setpoints = [
        e for e in doc.elements if e.type == "setpoint" and e.attributes.get("value") == "257"
    ]
    assert setpoints, "expected to find a setpoint element with value 257 (PSV 9066A/B)"


def test_finds_pit_9062_instrument_loop_and_its_hh_setpoint(doc: CanonicalDocument):
    """Ground truth: PIT-9062 exists with an HH setpoint of 245 (Pair 1 edit target)."""
    loops = [
        e
        for e in doc.elements
        if e.type == "instrument_loop" and e.attributes.get("loop_number") == "9062"
    ]
    assert loops, "expected an instrument_loop element for loop 9062 (PIT-9062)"

    expected_attrs = {"setpoint_type": "HH", "value": "245"}
    hh_setpoints = [
        e for e in doc.elements if e.type == "setpoint" and e.attributes == expected_attrs
    ]
    assert hh_setpoints, "expected a standalone HH:245 setpoint element near PIT-9062"


def test_finds_flow_rate_note_29_value(doc: CanonicalDocument):
    """Ground truth: FLOW RATE value 19057 appears near NOTE 29 (Pair 1 edit target)."""
    assert any("19057" in e.text for e in doc.elements)


def test_bbox_normalized_stays_within_unit_square(doc: CanonicalDocument):
    for e in doc.elements:
        x0, y0, x1, y1 = e.bbox.normalized
        assert -0.01 <= x0 <= 1.01
        assert -0.01 <= y0 <= 1.01
        assert -0.01 <= x1 <= 1.01
        assert -0.01 <= y1 <= 1.01


def test_second_sample_pdf_also_ingests_without_error():
    other = Path("data/samples/originals/export_gas_compressor_26-KA-902.pdf")
    if not other.exists():
        pytest.skip("second sample PDF not present")
    result = PdfNativeAdapter().ingest(other, pid="export_gas_A")
    assert len(result.elements) > 500
