"""Tests for the delta markup overlay: real PDF/DXF files, real pymupdf and
matplotlib rendering -- there's no clean way to fake "open and draw on the
actual source document" without losing the thing under test, same
integration-test posture as test_dwg.py against the committed sample files.

Deltas come from actually running compute_delta() on the real sample pairs
(deterministic, no LLM) rather than hand-built Delta objects, so these tests
exercise the real ingest -> delta -> markup pipeline end to end, the same
path write_report()'s CLI runs in production.
"""

from pathlib import Path

import fitz
import pytest

from src.canonical.model import BBox, CanonicalDocument
from src.delta.colors import STATUS_COLORS
from src.delta.engine import Delta, compute_delta
from src.ingest.dwg import DwgAdapter
from src.ingest.pdf_native import PdfNativeAdapter
from src.markup.overlay import (
    _boxes_for_side,
    _change_label,
    _hex_to_rgb01,
    _resolve_dxf_path,
    render_markup_png,
    save_dxf_markup,
    save_pdf_markup,
    write_markup,
)

PAIR1_A = Path("data/samples/originals/lift_gas_compressor_26-KA-901.pdf")
PAIR1_B = Path("data/samples/pair1/B.pdf")
PAIR3_A = Path("data/samples/pair3/A.dxf")
PAIR3_B = Path("data/samples/pair3/B.dxf")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def bbox(**overrides) -> BBox:
    defaults = {
        "page": 0, "x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0,
        "page_width": 100.0, "page_height": 100.0,
    }  # fmt: skip
    defaults.update(overrides)
    return BBox(**defaults)


@pytest.fixture(scope="module")
def pdf_doc_a():
    return PdfNativeAdapter().ingest(PAIR1_A, pid="ov_pair1_A", revision_label="as-provided")


@pytest.fixture(scope="module")
def pdf_doc_b():
    return PdfNativeAdapter().ingest(PAIR1_B, pid="ov_pair1_B", revision_label="rev B")


@pytest.fixture(scope="module")
def pdf_deltas(pdf_doc_a, pdf_doc_b):
    return compute_delta(pdf_doc_a, pdf_doc_b).deltas


@pytest.fixture(scope="module")
def dxf_doc_a():
    return DwgAdapter().ingest(PAIR3_A, pid="ov_pair3_A", revision_label="rev A")


@pytest.fixture(scope="module")
def dxf_doc_b():
    return DwgAdapter().ingest(PAIR3_B, pid="ov_pair3_B", revision_label="rev B")


@pytest.fixture(scope="module")
def dxf_deltas(dxf_doc_a, dxf_doc_b):
    return compute_delta(dxf_doc_a, dxf_doc_b).deltas


class TestHexToRgb01:
    def test_converts_known_color(self):
        r, g, b = _hex_to_rgb01("#0ca30c")
        assert r == pytest.approx(12 / 255)
        assert g == pytest.approx(163 / 255)
        assert b == pytest.approx(12 / 255)

    def test_all_status_colors_parse(self):
        for hex_color in STATUS_COLORS.values():
            r, g, b = _hex_to_rgb01(hex_color)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0


class TestResolveDxfPath:
    """The .dwg branch needs the real ODA File Converter (same
    environment-dependent boundary as test_dwg.py's own conversion tests),
    so this only proves the dispatch: a .dwg source is routed through
    convert_dwg_to_dxf(), a non-.dwg source is returned untouched."""

    def _doc(self, raw_source_path: str) -> CanonicalDocument:
        fmt = "dwg" if raw_source_path.lower().endswith(".dwg") else "pdf_native"
        return CanonicalDocument(
            pid="x", format=fmt, page_count=1, elements=[], raw_source_path=raw_source_path
        )

    def test_non_dwg_source_path_is_returned_unchanged(self):
        result = _resolve_dxf_path(self._doc(str(PAIR3_A)))
        assert result == PAIR3_A

    def test_dwg_source_path_is_routed_through_the_oda_converter(self, monkeypatch):
        converted = Path("converted/output.dxf")
        called_with = {}

        def fake_convert(path: Path) -> Path:
            called_with["path"] = path
            return converted

        monkeypatch.setattr("src.ingest.dwg.convert_dwg_to_dxf", fake_convert)

        result = _resolve_dxf_path(self._doc("drawing.dwg"))

        assert result == converted
        assert called_with["path"] == Path("drawing.dwg")


class TestChangeLabel:
    def test_added_label(self):
        d = Delta(change_type="added", element_type="note", new_text="NEW NOTE", confidence=1.0)
        assert _change_label(d) == "added note: NEW NOTE"

    def test_removed_label(self):
        d = Delta(change_type="removed", element_type="note", old_text="OLD NOTE", confidence=1.0)
        assert _change_label(d) == "removed note: OLD NOTE"

    def test_modified_label(self):
        d = Delta(
            change_type="modified", element_type="setpoint",
            old_text="SP = 257 bar (g)", new_text="SP = 260 bar (g)", confidence=1.0,
        )  # fmt: skip
        assert _change_label(d) == "modified setpoint: SP = 257 bar (g) -> SP = 260 bar (g)"


class TestBoxesForSide:
    def test_side_a_includes_removed_and_modified_old(self):
        deltas = [
            Delta(change_type="removed", element_type="note", old_bbox=bbox(), confidence=1.0),
            Delta(
                change_type="modified", element_type="setpoint",
                old_bbox=bbox(x0=1), new_bbox=bbox(x0=2), confidence=1.0,
            ),
            Delta(change_type="added", element_type="note", new_bbox=bbox(), confidence=1.0),
        ]  # fmt: skip
        result = _boxes_for_side(deltas, "a")
        assert [d.change_type for d, _ in result] == ["removed", "modified"]
        assert result[1][1].x0 == 1  # old_bbox, not new_bbox

    def test_side_b_includes_added_and_modified_new(self):
        deltas = [
            Delta(change_type="removed", element_type="note", old_bbox=bbox(), confidence=1.0),
            Delta(
                change_type="modified", element_type="setpoint",
                old_bbox=bbox(x0=1), new_bbox=bbox(x0=2), confidence=1.0,
            ),
            Delta(change_type="added", element_type="note", new_bbox=bbox(), confidence=1.0),
        ]  # fmt: skip
        result = _boxes_for_side(deltas, "b")
        assert [d.change_type for d, _ in result] == ["modified", "added"]
        assert result[0][1].x0 == 2  # new_bbox, not old_bbox

    def test_missing_bbox_is_skipped_not_crashed(self):
        deltas = [Delta(change_type="removed", element_type="note", old_bbox=None, confidence=1.0)]
        assert _boxes_for_side(deltas, "a") == []

    def test_empty_deltas_returns_empty(self):
        assert _boxes_for_side([], "a") == []
        assert _boxes_for_side([], "b") == []


class TestPdfMarkup:
    def test_save_writes_valid_pdf_with_annotations(self, pdf_doc_b, pdf_deltas, tmp_path):
        out_path = tmp_path / "markup_b.pdf"
        result = save_pdf_markup(pdf_doc_b, pdf_deltas, "b", out_path)
        assert result == out_path
        assert out_path.exists()

        reopened = fitz.open(out_path)
        annots = list(reopened[0].annots())
        expected_boxes = len(_boxes_for_side(pdf_deltas, "b"))
        assert expected_boxes > 0  # Pair 1 has real added/modified deltas
        assert len(annots) == expected_boxes + 3  # + one legend annot per status
        reopened.close()

    def test_tooltip_content_matches_change_label(self, pdf_doc_b, pdf_deltas, tmp_path):
        out_path = tmp_path / "markup_b.pdf"
        save_pdf_markup(pdf_doc_b, pdf_deltas, "b", out_path)
        reopened = fitz.open(out_path)
        rect_annots = [a for a in reopened[0].annots() if a.type[1] == "Square"]
        boxes = _boxes_for_side(pdf_deltas, "b")
        labels = {_change_label(d) for d, _ in boxes}
        annot_contents = {a.info["content"] for a in rect_annots}
        assert annot_contents == labels
        reopened.close()

    def test_original_source_file_is_never_modified(self, pdf_doc_b, pdf_deltas, tmp_path):
        original_bytes_before = PAIR1_B.read_bytes()
        save_pdf_markup(pdf_doc_b, pdf_deltas, "b", tmp_path / "out.pdf")
        assert PAIR1_B.read_bytes() == original_bytes_before

    def test_render_markup_png_returns_valid_png(self, pdf_doc_b, pdf_deltas):
        png_bytes = render_markup_png(pdf_doc_b, pdf_deltas, "b")
        assert png_bytes.startswith(_PNG_MAGIC)
        assert len(png_bytes) > 1000

    def test_side_with_no_relevant_deltas_still_renders(self, pdf_doc_a):
        # side "b" logic against doc_a's own deltas produces no boxes for
        # "removed"-only content, but rendering must not crash.
        removed_only = [
            Delta(change_type="removed", element_type="note", old_bbox=bbox(page=0), confidence=1.0)
        ]  # fmt: skip
        png_bytes = render_markup_png(pdf_doc_a, removed_only, "b")
        assert png_bytes.startswith(_PNG_MAGIC)


class TestDxfMarkup:
    def test_save_writes_valid_png(self, dxf_doc_b, dxf_deltas, tmp_path):
        out_path = tmp_path / "markup_b.png"
        result = save_dxf_markup(dxf_doc_b, dxf_deltas, "b", out_path)
        assert result == out_path
        assert out_path.read_bytes().startswith(_PNG_MAGIC)

    def test_boxes_present_for_side_with_real_deltas(self, dxf_deltas):
        # Pair 3's ground truth includes an added valve and modified setpoint/
        # geometry/dimension -- side "b" must have boxes to draw.
        assert len(_boxes_for_side(dxf_deltas, "b")) > 0
        assert len(_boxes_for_side(dxf_deltas, "a")) > 0

    def test_render_markup_png_dispatches_to_dxf_path(self, dxf_doc_b, dxf_deltas):
        png_bytes = render_markup_png(dxf_doc_b, dxf_deltas, "b")
        assert png_bytes.startswith(_PNG_MAGIC)

    def test_original_source_files_never_modified(self, dxf_doc_a, dxf_doc_b, dxf_deltas, tmp_path):
        a_before = PAIR3_A.read_bytes()
        b_before = PAIR3_B.read_bytes()
        save_dxf_markup(dxf_doc_a, dxf_deltas, "a", tmp_path / "a.png")
        save_dxf_markup(dxf_doc_b, dxf_deltas, "b", tmp_path / "b.png")
        assert PAIR3_A.read_bytes() == a_before
        assert PAIR3_B.read_bytes() == b_before


class TestWriteMarkupDispatch:
    def test_pdf_pair_produces_pdf_files(self, pdf_doc_a, pdf_doc_b, pdf_deltas, tmp_path):
        paths = write_markup(pdf_doc_a, pdf_doc_b, pdf_deltas, tmp_path, basename="t")
        assert set(paths.keys()) == {"a", "b"}
        assert paths["a"].suffix == ".pdf"
        assert paths["b"].suffix == ".pdf"
        assert paths["a"].name == "t_A.pdf"
        assert paths["a"].exists()
        assert paths["b"].exists()

    def test_dwg_pair_produces_png_files(self, dxf_doc_a, dxf_doc_b, dxf_deltas, tmp_path):
        paths = write_markup(dxf_doc_a, dxf_doc_b, dxf_deltas, tmp_path, basename="t")
        assert paths["a"].suffix == ".png"
        assert paths["b"].suffix == ".png"
        assert paths["a"].exists()
        assert paths["b"].exists()

    def test_creates_output_dir_if_missing(self, dxf_doc_a, dxf_doc_b, dxf_deltas, tmp_path):
        out_dir = tmp_path / "nested" / "markup"
        paths = write_markup(dxf_doc_a, dxf_doc_b, dxf_deltas, out_dir, basename="t")
        assert paths["a"].exists()
