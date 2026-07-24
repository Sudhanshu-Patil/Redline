import pytest
from pydantic import ValidationError

from src.canonical.model import BBox, CanonicalDocument, Element, make_element_id


def make_bbox(**overrides) -> BBox:
    defaults = dict(
        page=0, x0=10.0, y0=20.0, x1=110.0, y1=70.0, page_width=200.0, page_height=100.0
    )
    defaults.update(overrides)
    return BBox(**defaults)


def test_bbox_normalized_is_page_relative_fraction():
    box = make_bbox()
    x0, y0, x1, y1 = box.normalized
    assert x0 == pytest.approx(0.05)
    assert y0 == pytest.approx(0.2)
    assert x1 == pytest.approx(0.55)
    assert y1 == pytest.approx(0.7)


def test_bbox_rejects_inverted_coordinates():
    with pytest.raises(ValidationError):
        make_bbox(x0=100.0, x1=10.0)


def test_bbox_units_default_to_pdf_points():
    assert make_bbox().unit == "pdf_points"


def test_make_element_id_is_stable_and_readable():
    assert make_element_id("pair1_A", "pdf_native", 3) == "pair1_A:pdf_native:00003"


def test_element_requires_confidence_in_unit_interval():
    with pytest.raises(ValidationError):
        Element(
            id="x:pdf_native:00001",
            type="tag",
            text="PSV-9066A",
            bbox=make_bbox(),
            source_adapter="pdf_native",
            extraction_confidence=1.5,
        )


def test_canonical_document_round_trips_through_json():
    doc = CanonicalDocument(
        pid="pair1_A",
        format="pdf_native",
        revision_label="Rev A",
        page_count=1,
        raw_source_path="data/samples/pair1/A.pdf",
        elements=[
            Element(
                id=make_element_id("pair1_A", "pdf_native", 1),
                type="setpoint",
                text="PSV 9066A 257 bar(g)",
                bbox=make_bbox(),
                attributes={"tag": "PSV-9066A"},
                source_adapter="pdf_native",
                extraction_confidence=0.98,
            )
        ],
    )
    restored = CanonicalDocument.model_validate_json(doc.model_dump_json())
    assert restored == doc
