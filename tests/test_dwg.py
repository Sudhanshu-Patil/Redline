"""DWG/DXF adapter tests: in-memory ezdxf documents for unit coverage, plus
the committed Pair 3 files for integration."""

from pathlib import Path

import ezdxf
import pytest

from src.config import settings
from src.ingest.dwg import DwgAdapter, convert_dwg_to_dxf, resolve_oda_converter

PAIR3_A = Path("data/samples/pair3/A.dxf")
PAIR3_B = Path("data/samples/pair3/B.dxf")


def _ingest_memory_doc(doc, tmp_path, pid="mem") :
    path = tmp_path / "mem.dxf"
    doc.saveas(path)
    return DwgAdapter().ingest(path, pid=pid)


@pytest.fixture
def basic_doc():
    doc = ezdxf.new("R2018")
    doc.layers.add("TEXT", color=7)
    doc.layers.add("PIPING", color=4)
    return doc


class TestEntityExtraction:
    def test_text_entity_classified_through_shared_rules(self, basic_doc, tmp_path):
        msp = basic_doc.modelspace()
        msp.add_text("SP = 257 bar (g)", height=2.0, dxfattribs={"layer": "TEXT"}).set_placement(
            (10, 10)
        )
        result = _ingest_memory_doc(basic_doc, tmp_path)
        assert len(result.elements) == 1
        el = result.elements[0]
        assert el.type == "setpoint"
        assert el.attributes["value"] == "257"
        assert el.attributes["layer"] == "TEXT"
        assert el.attributes["entity_type"] == "TEXT"
        assert el.bbox.unit == "dxf_units"

    def test_insert_with_attrib_yields_block_and_tag_elements(self, basic_doc, tmp_path):
        bubble = basic_doc.blocks.new(name="INSTR_BUBBLE")
        bubble.add_circle((0, 0), radius=4)
        bubble.add_attdef("TAG", insert=(-3, -1), height=1.6)
        msp = basic_doc.modelspace()
        ref = msp.add_blockref("INSTR_BUBBLE", (50, 50), dxfattribs={"layer": "PIPING"})
        ref.add_auto_attribs({"TAG": "PIT-9062"})

        result = _ingest_memory_doc(basic_doc, tmp_path)
        inserts = [e for e in result.elements if e.attributes.get("entity_type") == "INSERT"]
        attribs = [e for e in result.elements if e.attributes.get("entity_type") == "ATTRIB"]
        assert len(inserts) == 1
        assert inserts[0].type == "geometry"
        assert inserts[0].attributes["block_name"] == "INSTR_BUBBLE"
        assert len(attribs) == 1
        assert attribs[0].text == "PIT-9062"
        assert attribs[0].type == "instrument_loop"  # via the hyphenated-tag rule
        assert attribs[0].attributes["parent_block"] == "INSTR_BUBBLE"
        assert attribs[0].attributes["attrib_tag"] == "TAG"

    def test_plain_geometry_carries_layer_and_entity_type(self, basic_doc, tmp_path):
        msp = basic_doc.modelspace()
        msp.add_line((0, 0), (100, 0), dxfattribs={"layer": "PIPING"})
        msp.add_circle((50, 20), radius=5, dxfattribs={"layer": "PIPING"})
        result = _ingest_memory_doc(basic_doc, tmp_path)
        assert {e.attributes["entity_type"] for e in result.elements} == {"LINE", "CIRCLE"}
        assert all(e.type == "geometry" and e.text == "" for e in result.elements)
        assert all(e.attributes["layer"] == "PIPING" for e in result.elements)

    def test_dimension_override_text_wins(self, basic_doc, tmp_path):
        msp = basic_doc.modelspace()
        dim = msp.add_linear_dim(base=(0, -5), p1=(0, 0), p2=(60, 0), text="600")
        dim.render()
        result = _ingest_memory_doc(basic_doc, tmp_path)
        dims = [e for e in result.elements if e.type == "dimension"]
        assert len(dims) == 1
        assert dims[0].text == "600"

    def test_dimension_measured_value_when_no_override(self, basic_doc, tmp_path):
        msp = basic_doc.modelspace()
        dim = msp.add_linear_dim(base=(0, -5), p1=(0, 0), p2=(60, 0))
        dim.render()
        result = _ingest_memory_doc(basic_doc, tmp_path)
        dims = [e for e in result.elements if e.type == "dimension"]
        assert len(dims) == 1
        assert dims[0].text == f"{60:.{settings.dwg_dim_decimals}f}"

    def test_mtext_notes_split_into_note_elements(self, basic_doc, tmp_path):
        basic_doc.layers.add("NOTES", color=7)
        msp = basic_doc.modelspace()
        msp.add_mtext(
            "1. RELIEF TO HP FLARE.\n2. HEAT TRACING PER SPEC.",
            dxfattribs={"layer": "NOTES", "char_height": 2.0, "insert": (5, 5), "width": 80},
        )
        result = _ingest_memory_doc(basic_doc, tmp_path)
        notes = [e for e in result.elements if e.type == "note"]
        assert [n.attributes["note_number"] for n in notes] == ["1", "2"]

    def test_normalized_bboxes_within_unit_square(self, basic_doc, tmp_path):
        msp = basic_doc.modelspace()
        msp.add_line((-50, -20), (150, 90), dxfattribs={"layer": "PIPING"})
        msp.add_text("26BL9072", height=2.0, dxfattribs={"layer": "TEXT"}).set_placement((10, 40))
        result = _ingest_memory_doc(basic_doc, tmp_path)
        for e in result.elements:
            x0, y0, x1, y1 = e.bbox.normalized
            assert -0.001 <= x0 <= 1.001 and -0.001 <= y1 <= 1.001


class TestDwgConversion:
    def test_explicit_config_wins(self, monkeypatch):
        monkeypatch.setattr(settings, "oda_file_converter", r"C:\tools\ODAFileConverter.exe")
        assert resolve_oda_converter() == r"C:\tools\ODAFileConverter.exe"

    def test_binary_dwg_without_converter_gives_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "oda_file_converter", "")
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr("src.ingest.dwg._ODA_CANDIDATES", [])
        dwg = tmp_path / "drawing.dwg"
        dwg.write_bytes(b"AC1032 fake dwg")
        with pytest.raises(RuntimeError, match="ODA File Converter"):
            convert_dwg_to_dxf(dwg)

    def test_ingest_routes_dwg_through_converter(self, monkeypatch, tmp_path):
        """A .dwg path must invoke conversion, then parse the produced DXF."""
        target = ezdxf.new("R2018")
        target.layers.add("TEXT", color=7)
        target.modelspace().add_text(
            "26BL9072", height=2.0, dxfattribs={"layer": "TEXT"}
        ).set_placement((0, 0))
        converted = tmp_path / "converted.dxf"
        target.saveas(converted)

        calls = {}

        def fake_convert(path):
            calls["src"] = path
            return converted

        monkeypatch.setattr("src.ingest.dwg.convert_dwg_to_dxf", fake_convert)
        dwg = tmp_path / "drawing.dwg"
        dwg.write_bytes(b"AC1032 fake dwg")

        result = DwgAdapter().ingest(dwg, pid="dwg_route")
        assert calls["src"] == dwg
        assert [e.text for e in result.elements] == ["26BL9072"]
        assert result.raw_source_path == str(dwg)


pair3_missing = not (PAIR3_A.exists() and PAIR3_B.exists())


@pytest.fixture(scope="module")
def docs():
    a = DwgAdapter().ingest(PAIR3_A, pid="p3A")
    b = DwgAdapter().ingest(PAIR3_B, pid="p3B")
    return a, b


@pytest.mark.skipif(pair3_missing, reason="pair3 DXF files not present (run make data)")
class TestPair3Integration:

    def test_ground_truth_markers(self, docs):
        a, b = docs
        ta = [e.text for e in a.elements]
        tb = [e.text for e in b.elements]
        assert "SP = 257 bar (g)" in ta and "SP = 257 bar (g)" not in tb
        assert "SP = 260 bar (g)" in tb
        assert "43BL9020" in tb and "43BL9020" not in ta
        assert "600" in ta and "750" in tb

    def test_attrib_tags_are_instrument_loops(self, docs):
        a, _ = docs
        loops = [e for e in a.elements if e.type == "instrument_loop"]
        assert {e.text for e in loops} == {"PIT-9062", "PSV-9066A"}

    def test_moved_valve_same_key_different_position(self, docs):
        a, b = docs

        def valve_insert_near_label(doc):
            label = next(e for e in doc.elements if e.text == "26BL9075")
            inserts = [
                e for e in doc.elements if e.attributes.get("block_name") == "VALVE_GATE"
            ]
            return min(
                inserts,
                key=lambda e: abs(e.bbox.x0 - label.bbox.x0) + abs(e.bbox.y0 - label.bbox.y0),
            )

        va, vb = valve_insert_near_label(a), valve_insert_near_label(b)
        assert va.attributes["layer"] == vb.attributes["layer"] == "PIPING"
        assert (va.bbox.x0, va.bbox.y0) != (vb.bbox.x0, vb.bbox.y0)

    def test_ingest_is_deterministic(self, docs):
        again = DwgAdapter().ingest(PAIR3_A, pid="p3A")
        assert again.model_dump() == docs[0].model_dump()

    def test_emits_trace_spans(self, tmp_path, monkeypatch):
        import json as jsonlib

        from src.observability import tracing

        monkeypatch.setattr(tracing.settings, "traces_dir", tmp_path)
        with tracing.trace("dwg_test") as trace_id:
            DwgAdapter().ingest(PAIR3_A, pid="span_check")
        records = [
            jsonlib.loads(line)
            for line in (tmp_path / f"{trace_id}.jsonl").read_text().splitlines()
        ]
        ingest = next(r for r in records if r["name"] == "dwg.ingest")
        assert ingest["attributes"]["elements_extracted"] > 0
