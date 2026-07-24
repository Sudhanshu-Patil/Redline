"""Synthesize sample pairs 1, 2 and 4 (IMPLEMENTATION_PLAN.md §6), reproducibly.

Pair 1: Lift Gas Compressor P&ID as A; B = A with the documented §6 edit list
        applied via pymupdf redact-and-reinsert. Every edit is asserted to hit
        exactly the expected number of matches, and the resulting B is
        re-ingested through the Phase 1 adapter to verify each new value is
        extractable and each old value is gone -- synthesis fails loudly
        rather than producing a silently-wrong ground truth.
Pair 2: B of Pair 1 rasterized at OCR_DPI into an image-only PDF (no text
        layer) -- the scanned-adapter exercise with a known ground truth.
Pair 4: negative control -- the two as-provided companion documents, which
        are NOT a revision pair; ground truth says "expect a low-alignment
        warning, not a delta dump".

Run: uv run python scripts/synthesize_pairs.py   (or: make data)
"""

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.schema import (  # noqa: E402
    DocRef,
    ExpectedDelta,
    GroundTruth,
    GTLocator,
    PairManifest,
)
from src.canonical.model import CanonicalDocument  # noqa: E402
from src.config import settings  # noqa: E402
from src.ingest.pdf_native import PdfNativeAdapter  # noqa: E402
from src.observability import tracing  # noqa: E402
from src.observability.logging import get_logger  # noqa: E402

log = get_logger("synthesize_pairs")

ORIGINALS = REPO_ROOT / "data" / "samples" / "originals"
SAMPLES = REPO_ROOT / "data" / "samples"
LIFT_GAS = ORIGINALS / "lift_gas_compressor_26-KA-901.pdf"
EXPORT_GAS = ORIGINALS / "export_gas_compressor_26-KA-902.pdf"

CREATED_BY = "scripts/synthesize_pairs.py"
FONT = "helv"  # source uses Calibri 5.5pt; helv at the same size is the closest base-14 font
FONTSIZE = 5.5

# Inset applied to redaction rects so they never bleed into the neighbouring
# note rows, which sit only ~5.7pt apart in the notes columns.
REDACT_INSET = 0.25

NOTE_37_TEXT = "37. PSV 9066A/B SET PRESSURE REVISED TO 260 BAR(G)."


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _insert_replacement(page: fitz.Page, rect: fitz.Rect, text: str) -> None:
    baseline_y = rect.y1 - 0.21 * FONTSIZE
    page.insert_text((rect.x0, baseline_y), text, fontsize=FONTSIZE, fontname=FONT, color=(0, 0, 0))


def _redact(page: fitz.Page, rect: fitz.Rect) -> None:
    page.add_redact_annot(
        fitz.Rect(
            rect.x0 - 0.5,  # tiny x margin so glyph edges are fully covered
            rect.y0 + REDACT_INSET,
            rect.x1 + 0.5,
            rect.y1 - REDACT_INSET,
        )
    )


def _apply_redactions_text_only(page: fitz.Page) -> None:
    """Remove redacted text but leave images and line art untouched --
    critical on a P&ID where symbols sit close to their labels."""
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE
    )


def _find_line_rect(page: fitz.Page, startswith: str, x_min: float = 0.0) -> fitz.Rect:
    """Rect of the first dict-mode line whose text starts with `startswith`."""
    d = page.get_text("dict")
    for block in d["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text.startswith(startswith) and line["bbox"][0] >= x_min:
                return fitz.Rect(line["bbox"])
    raise AssertionError(f"no line starting with {startswith!r} found (x_min={x_min})")


def _search_exactly(page: fitz.Page, needle: str, expect: int) -> list[fitz.Rect]:
    hits = page.search_for(needle)
    assert len(hits) == expect, f"{needle!r}: expected {expect} hits, got {len(hits)}"
    return sorted(hits, key=lambda r: (r.y0, r.x0))


def _extract_texts(pdf_path: Path, pid: str) -> tuple[CanonicalDocument, list[str]]:
    doc = PdfNativeAdapter().ingest(pdf_path, pid=pid)
    return doc, [e.text for e in doc.elements]


def build_pair1() -> None:
    with tracing.span("synthesize.pair1"):
        out_dir = SAMPLES / "pair1"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / "B.pdf"

        doc = fitz.open(LIFT_GAS)
        page = doc[0]
        expected: list[ExpectedDelta] = []

        # --- Edit 1+2: PSV 9066A/B set pressure 257 -> 260 bar(g) ---
        sp_hits = _search_exactly(page, "SP = 257 bar (g)", expect=2)
        # Disambiguate by which PSV bubble each label sits nearest to: the
        # 9066A bubble is around x~856,y~80; 9066B around x~1025,y~70.
        for rect in sp_hits:
            near = "PSV-9066A" if rect.x0 < 950 else "PSV-9066B"
            expected.append(
                ExpectedDelta(
                    gt_id=f"GT-SP-{near[-5:]}",
                    change_type="modified",
                    element_type="setpoint",
                    old_text="SP = 257 bar (g)",
                    new_text="SP = 260 bar (g)",
                    locator=GTLocator(
                        page=0, bbox=(rect.x0, rect.y0, rect.x1, rect.y1), near_text=near
                    ),
                    rationale=f"Relief set pressure uprated at {near}; tests duplicate-text "
                    "disambiguation (two identical SP strings on one sheet).",
                )
            )

        # --- Edit 3: PIT-9062 HH alarm 245 -> 250 ---
        hh_hits = _search_exactly(page, "HH: 245", expect=1)
        expected.append(
            ExpectedDelta(
                gt_id="GT-HH-9062",
                change_type="modified",
                element_type="setpoint",
                old_text="HH: 245",
                new_text="HH: 250",
                locator=GTLocator(
                    page=0,
                    bbox=(hh_hits[0].x0, hh_hits[0].y0, hh_hits[0].x1, hh_hits[0].y1),
                    near_text="PIT-9062",
                ),
                rationale="Trip limit change on an instrument bubble annotation -- the classic "
                "'small number nobody notices' revision.",
            )
        )

        # --- Edit 4: FLOW RATE 19057 -> 20500 kg/h (design data table) ---
        flow_hits = _search_exactly(page, "19057", expect=1)
        expected.append(
            ExpectedDelta(
                gt_id="GT-FLOW",
                change_type="modified",
                element_type="text_block",
                old_text="19057",
                new_text="20500",
                locator=GTLocator(
                    page=0,
                    bbox=(flow_hits[0].x0, flow_hits[0].y0, flow_hits[0].x1, flow_hits[0].y1),
                    near_text="FLOW RATE",
                ),
                rationale="Design-basis table value change, far from any tag key -- exercises "
                "embedding/proximity matching rather than exact-key matching.",
            )
        )

        # --- Edit 5: delete note 30 (definition line; the three NOTE 30
        # callouts on the drawing deliberately stay -- see PROVENANCE) ---
        marker_rect = _find_line_rect(page, "30.", x_min=240)
        # Full body *line* rect, not just the searched substring -- the line
        # continues "- HYDRATE MITIGATION (25°C)" past any search needle.
        body_rect = _find_line_rect(page, "SAFETY CRITICAL HEAT TRACING", x_min=240)
        note30_rect = fitz.Rect(marker_rect) | body_rect
        expected.append(
            ExpectedDelta(
                gt_id="GT-NOTE30-DEL",
                change_type="removed",
                element_type="note",
                old_text="30. SAFETY CRITICAL HEAT TRACING - HYDRATE MITIGATION (25°C)",
                new_text=None,
                locator=GTLocator(
                    page=0,
                    bbox=(note30_rect.x0, note30_rect.y0, note30_rect.x1, note30_rect.y1),
                    near_text="29.",
                ),
                rationale="Deleted note definition; its three on-drawing NOTE 30 callouts are "
                "left in place (documented dangling-reference case).",
            )
        )

        # --- Apply all redactions, then insert replacements ---
        for rect in sp_hits:
            _redact(page, rect)
        _redact(page, hh_hits[0])
        _redact(page, flow_hits[0])
        _redact(page, note30_rect)
        _apply_redactions_text_only(page)

        for rect in sp_hits:
            _insert_replacement(page, rect, "SP = 260 bar (g)")
        _insert_replacement(page, hh_hits[0], "HH: 250")
        _insert_replacement(page, flow_hits[0], "20500")

        # --- Edit 6: add note 37 below note 36 in the third notes column ---
        note36_rect = _find_line_rect(page, "36.", x_min=500)
        # One row step down, plus clearance for the revision-cloud outline
        # drawn around notes 35/36 in the original.
        offset = 5.76 + 3.0
        note37_rect = fitz.Rect(
            note36_rect.x0,
            note36_rect.y0 + offset,
            note36_rect.x0 + 200,
            note36_rect.y1 + offset,
        )
        _insert_replacement(page, note37_rect, NOTE_37_TEXT)
        expected.append(
            ExpectedDelta(
                gt_id="GT-NOTE37-ADD",
                change_type="added",
                element_type="note",
                old_text=None,
                new_text=NOTE_37_TEXT,
                locator=GTLocator(
                    page=0,
                    bbox=(note37_rect.x0, note37_rect.y0, note37_rect.x1, note37_rect.y1),
                    near_text="36.",
                ),
                rationale="Added MOC-style note recording the PSV set-pressure change.",
            )
        )

        doc.save(out_pdf, garbage=4, deflate=True)
        doc.close()

        # --- Verify: re-ingest B through the real adapter ---
        _, texts = _extract_texts(out_pdf, pid="pair1_B_verify")
        joined = "\n".join(texts)
        assert joined.count("SP = 260 bar (g)") == 2, "expected 2 modified SP setpoints in B"
        assert "SP = 257 bar (g)" not in joined, "old SP value still present in B"
        assert "HH: 250" in joined, "modified HH setpoint missing in B"
        assert "HH: 245" not in joined, "old HH value still present in B"
        assert "20500" in joined, "modified flow rate missing in B"
        assert "19057" not in joined, "old flow rate still present in B"
        assert "SAFETY CRITICAL HEAT TRACING" not in joined, "note 30 body still present in B"
        assert "HYDRATE MITIGATION" not in joined, "note 30 line tail survived the redaction"
        assert NOTE_37_TEXT in joined, "added note 37 missing in B"
        # Neighbours of the deleted note must have survived the redaction.
        assert "POWER AT COMPRESSOR COUPLING" in joined, "note 28 damaged by redaction"
        assert "CASE 8A" in joined, "note 29 damaged by redaction"
        assert "LL SET POINT IS OVERRIDEN" in joined, "note 31 damaged by redaction"
        # The dangling callouts stay.
        assert "NOTE 30" in joined, "NOTE 30 callouts should remain in B"

        ground_truth = GroundTruth(
            pair_id="pair1",
            expected_deltas=expected,
            notes=(
                "B was produced from A by scripts/synthesize_pairs.py. The three on-drawing "
                "'NOTE 30' callouts intentionally remain after the note-30 definition was "
                "deleted; they are NOT ground-truth changes. The title-block revision was "
                "deliberately not bumped -- the edit list is exactly the six entries here."
            ),
        )
        (out_dir / "ground_truth.json").write_text(
            ground_truth.model_dump_json(indent=2), encoding="utf-8"
        )
        manifest = PairManifest(
            pair_id="pair1",
            description="Primary synthetic pair: Lift Gas Compressor P&ID vs MOC-style edits "
            "(2x PSV set pressure, PIT HH limit, flow rate, note removed, note added).",
            doc_a=DocRef(
                path=_rel(LIFT_GAS),
                pid="pair1_A",
                format="pdf_native",
                revision_label="as-provided",
            ),
            doc_b=DocRef(
                path=_rel(out_pdf),
                pid="pair1_B",
                format="pdf_native",
                revision_label="synthetic rev B",
            ),
            ground_truth_path=_rel(out_dir / "ground_truth.json"),
            created_by=CREATED_BY,
        )
        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        log.info("pair1 built", extra={"extra_fields": {"edits": len(expected)}})


def build_pair2() -> None:
    with tracing.span("synthesize.pair2", dpi=settings.ocr_dpi):
        pair1_dir = SAMPLES / "pair1"
        out_dir = SAMPLES / "pair2"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / "B_scanned.pdf"

        src = fitz.open(pair1_dir / "B.pdf")
        page = src[0]
        pix = page.get_pixmap(dpi=settings.ocr_dpi)
        scanned = fitz.open()
        new_page = scanned.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(page.rect, pixmap=pix)
        scanned.save(out_pdf, garbage=4, deflate=True)
        scanned.close()
        src.close()

        # Verify: image-only, no extractable text layer.
        check = fitz.open(out_pdf)
        assert check.page_count == 1
        residual_text = check[0].get_text().strip()
        assert residual_text == "", f"scanned B has a text layer: {residual_text[:60]!r}"
        check.close()

        # Same ground truth as pair1 -- it is the same revision delta, only
        # the B-side carrier format differs.
        gt = GroundTruth.model_validate_json(
            (pair1_dir / "ground_truth.json").read_text(encoding="utf-8")
        )
        gt.pair_id = "pair2"
        gt.notes += (
            " Pair 2 note: B is the Pair 1 B rasterized at "
            f"{settings.ocr_dpi} dpi with no text layer; expected deltas are identical, but "
            "extracted text on the B side will carry OCR confidence/noise."
        )
        (out_dir / "ground_truth.json").write_text(gt.model_dump_json(indent=2), encoding="utf-8")

        manifest = PairManifest(
            pair_id="pair2",
            description="Scanned-format pair: native A vs rasterized (image-only) B of Pair 1.",
            doc_a=DocRef(
                path=_rel(LIFT_GAS),
                pid="pair2_A",
                format="pdf_native",
                revision_label="as-provided",
            ),
            doc_b=DocRef(
                path=_rel(out_pdf),
                pid="pair2_B",
                format="pdf_scanned",
                revision_label="synthetic rev B (rasterized)",
            ),
            ground_truth_path=_rel(out_dir / "ground_truth.json"),
            created_by=CREATED_BY,
        )
        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        log.info("pair2 built", extra={"extra_fields": {"pdf": str(out_pdf)}})


def build_pair4() -> None:
    with tracing.span("synthesize.pair4"):
        out_dir = SAMPLES / "pair4"
        out_dir.mkdir(parents=True, exist_ok=True)

        ground_truth = GroundTruth(
            pair_id="pair4",
            negative_control=True,
            expected_deltas=[],
            notes=(
                "26-KA-901 (Lift Gas Compressor) vs 26-KA-902 (Export Gas Compressor) as "
                "provided. These are companion documents for different equipment, NOT a "
                "revision pair. Expected engine behaviour: surface a low-alignment-rate "
                "warning instead of reporting hundreds of spurious adds/removes. Eval treats "
                "this pair as pass/fail on that warning, not on delta P/R."
            ),
        )
        (out_dir / "ground_truth.json").write_text(
            ground_truth.model_dump_json(indent=2), encoding="utf-8"
        )
        manifest = PairManifest(
            pair_id="pair4",
            description="Negative control: two different compressors' P&IDs presented as if "
            "they were a revision pair.",
            doc_a=DocRef(
                path=_rel(LIFT_GAS),
                pid="pair4_A",
                format="pdf_native",
                revision_label="26-KA-901 as-provided",
            ),
            doc_b=DocRef(
                path=_rel(EXPORT_GAS),
                pid="pair4_B",
                format="pdf_native",
                revision_label="26-KA-902 as-provided",
            ),
            ground_truth_path=_rel(out_dir / "ground_truth.json"),
            created_by=CREATED_BY,
        )
        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        log.info("pair4 built")


# --- Pair 3: DWG/DXF ------------------------------------------------------
#
# Provenance decision (documented in PROVENANCE.md): no PDF->DXF converter
# preserves the text layer (text becomes vector outlines, destroying the keys
# the delta engine matches on), and no conversion tooling ships on a stock
# machine. So A.dxf is AUTHORED here with ezdxf as a compact P&ID-style
# schematic whose tag vocabulary mirrors the Lift Gas Compressor drawing, and
# B.dxf applies a documented edit list. Real files, real parser, exact
# ground truth -- and regenerable from this script alone.

PAIR3_SP_OLD = "SP = 257 bar (g)"
PAIR3_SP_NEW = "SP = 260 bar (g)"
PAIR3_DIM_OLD = "600"
PAIR3_DIM_NEW = "750"
PAIR3_ADDED_VALVE = "43BL9020"
PAIR3_MOVED_VALVE = "26BL9075"


def _author_pair3_drawing(edited: bool):
    """Build the Pair 3 model in memory; `edited` applies the B-side edits."""
    import ezdxf

    doc = ezdxf.new("R2018", setup=True)
    for name, color in [
        ("PIPING", 4), ("INSTRUMENTS", 2), ("TEXT", 7),
        ("NOTES", 7), ("EQUIPMENT", 1), ("DIMS", 3),
    ]:  # fmt: skip
        doc.layers.add(name, color=color)

    # Block: gate valve (bowtie)
    valve = doc.blocks.new(name="VALVE_GATE")
    valve.add_lwpolyline([(0, 0), (6, 3), (0, 3), (6, 0), (0, 0)], close=True)

    # Block: instrument bubble with a TAG attribute
    bubble = doc.blocks.new(name="INSTR_BUBBLE")
    bubble.add_circle((0, 0), radius=4)
    bubble.add_attdef("TAG", insert=(-3.4, -0.8), height=1.6)

    msp = doc.modelspace()

    # Main process line with a size/spec label
    msp.add_line((10, 40), (150, 40), dxfattribs={"layer": "PIPING"})
    msp.add_text(
        '6"-VF-43-9029-AC21S-00', height=2.2, dxfattribs={"layer": "TEXT"}
    ).set_placement((55, 43))

    # Two gate valves on the line; B moves 26BL9075 8 units right, 6 up
    msp.add_blockref("VALVE_GATE", (40, 38.5), dxfattribs={"layer": "PIPING"})
    msp.add_text("26BL9072", height=1.8, dxfattribs={"layer": "TEXT"}).set_placement((38, 34))
    moved_insert = (128, 44.5) if edited else (120, 38.5)
    msp.add_blockref("VALVE_GATE", moved_insert, dxfattribs={"layer": "PIPING"})
    msp.add_text(PAIR3_MOVED_VALVE, height=1.8, dxfattribs={"layer": "TEXT"}).set_placement(
        (moved_insert[0] - 2, moved_insert[1] - 4.5)
    )

    # Added in B only: a third valve teed off the line
    if edited:
        msp.add_blockref("VALVE_GATE", (90, 20), dxfattribs={"layer": "PIPING"})
        msp.add_text(PAIR3_ADDED_VALVE, height=1.8, dxfattribs={"layer": "TEXT"}).set_placement(
            (88, 15.5)
        )
        msp.add_line((93, 23), (93, 40), dxfattribs={"layer": "PIPING"})

    # Removed in B: a drain stub off the main line
    if not edited:
        msp.add_line((70, 40), (70, 25), dxfattribs={"layer": "PIPING"})

    # Instrument bubbles (block + ATTRIB tag)
    pit = msp.add_blockref("INSTR_BUBBLE", (90, 70), dxfattribs={"layer": "INSTRUMENTS"})
    pit.add_auto_attribs({"TAG": "PIT-9062"})
    msp.add_text("HH: 245", height=1.6, dxfattribs={"layer": "INSTRUMENTS"}).set_placement(
        (96, 71)
    )
    psv = msp.add_blockref("INSTR_BUBBLE", (140, 75), dxfattribs={"layer": "INSTRUMENTS"})
    psv.add_auto_attribs({"TAG": "PSV-9066A"})
    msp.add_text(
        PAIR3_SP_NEW if edited else PAIR3_SP_OLD,
        height=1.6,
        dxfattribs={"layer": "INSTRUMENTS"},
    ).set_placement((133, 81))

    # Equipment outline + title
    msp.add_lwpolyline(
        [(20, 55), (60, 55), (60, 90), (20, 90)], close=True, dxfattribs={"layer": "EQUIPMENT"}
    )
    msp.add_text("26-KA-901 GAS LIFT SKID", height=3.0, dxfattribs={"layer": "TEXT"}).set_placement(
        (20, 95)
    )

    # Notes block (MTEXT)
    msp.add_mtext(
        "1. RELIEF TO HP FLARE.\n2. HEAT TRACING PER SPEC 26-HT-02.",
        dxfattribs={"layer": "NOTES", "char_height": 1.8, "insert": (10, 12), "width": 70},
    )

    # Dimension with explicit text override; B revises the stated length
    dim = msp.add_linear_dim(
        base=(40, 30),
        p1=(40, 38.5),
        p2=(120, 38.5),
        text=PAIR3_DIM_NEW if edited else PAIR3_DIM_OLD,
        dxfattribs={"layer": "DIMS"},
    )
    dim.render()
    return doc


def build_pair3() -> None:
    with tracing.span("synthesize.pair3"):
        from src.ingest.dwg import DwgAdapter

        out_dir = SAMPLES / "pair3"
        out_dir.mkdir(parents=True, exist_ok=True)
        path_a = out_dir / "A.dxf"
        path_b = out_dir / "B.dxf"
        _author_pair3_drawing(edited=False).saveas(path_a)
        _author_pair3_drawing(edited=True).saveas(path_b)

        # Verify through the real adapter, exactly like pair 1.
        doc_a = DwgAdapter().ingest(path_a, pid="pair3_A_verify")
        doc_b = DwgAdapter().ingest(path_b, pid="pair3_B_verify")
        texts_a = [e.text for e in doc_a.elements]
        texts_b = [e.text for e in doc_b.elements]
        assert PAIR3_SP_OLD in texts_a and PAIR3_SP_OLD not in texts_b
        assert PAIR3_SP_NEW in texts_b and PAIR3_SP_NEW not in texts_a
        assert PAIR3_ADDED_VALVE in texts_b and PAIR3_ADDED_VALVE not in texts_a
        assert PAIR3_DIM_OLD in texts_a and PAIR3_DIM_NEW in texts_b
        assert "PIT-9062" in texts_a and "PIT-9062" in texts_b

        def insert_bbox(doc, label_text):
            # bbox of the VALVE_GATE insert nearest its label text
            inserts = [
                e for e in doc.elements if e.attributes.get("block_name") == "VALVE_GATE"
            ]
            label = next(e for e in doc.elements if e.text == label_text)
            return min(
                inserts,
                key=lambda e: abs(e.bbox.x0 - label.bbox.x0) + abs(e.bbox.y0 - label.bbox.y0),
            ).bbox

        moved_a = insert_bbox(doc_a, PAIR3_MOVED_VALVE)
        moved_b = insert_bbox(doc_b, PAIR3_MOVED_VALVE)
        assert (moved_a.x0, moved_a.y0) != (moved_b.x0, moved_b.y0), "valve move not applied"

        removed_lines_a = [
            e
            for e in doc_a.elements
            if e.attributes.get("entity_type") == "LINE" and e.bbox.x0 == e.bbox.x1
        ]
        assert removed_lines_a, "vertical drain line missing from A"

        expected = [
            ExpectedDelta(
                gt_id="GT3-SP",
                change_type="modified",
                element_type="setpoint",
                old_text=PAIR3_SP_OLD,
                new_text=PAIR3_SP_NEW,
                locator=GTLocator(page=0, near_text="PSV-9066A"),
                rationale="Text entity value change beside a tagged instrument block.",
            ),
            ExpectedDelta(
                gt_id="GT3-MOVE",
                change_type="modified",
                element_type="geometry",
                old_text=PAIR3_MOVED_VALVE,
                new_text=PAIR3_MOVED_VALVE,
                locator=GTLocator(
                    page=0,
                    bbox=(moved_a.x0, moved_a.y0, moved_a.x1, moved_a.y1),
                    near_text="VALVE_GATE",
                ),
                rationale="Moved block reference (same block, same layer): exercises the "
                "geometry-aware match rule (plan §4.2), which must pair the two INSERTs "
                "by layer+entity+block despite the offset. Label text moves with it.",
            ),
            ExpectedDelta(
                gt_id="GT3-DIM",
                change_type="modified",
                element_type="dimension",
                old_text=PAIR3_DIM_OLD,
                new_text=PAIR3_DIM_NEW,
                locator=GTLocator(page=0, near_text="DIMS"),
                rationale="Dimension text override revised.",
            ),
            ExpectedDelta(
                gt_id="GT3-ADD-VALVE",
                change_type="added",
                element_type="geometry",
                old_text=None,
                new_text=PAIR3_ADDED_VALVE,
                locator=GTLocator(page=0, near_text="VALVE_GATE"),
                rationale="New valve insert + label + tie-in line (counted as the labelled "
                "valve; the tie-in LINE is a second added geometry).",
            ),
            ExpectedDelta(
                gt_id="GT3-DEL-DRAIN",
                change_type="removed",
                element_type="geometry",
                old_text="",
                new_text=None,
                locator=GTLocator(page=0, bbox=(60, 25 - 12, 60 + 0.1, 40 - 12)),
                rationale="Drain stub LINE removed; no text key -- only the geometry rule "
                "can catch it.",
            ),
        ]
        # GT3-DEL-DRAIN carries old_text="" (a LINE has no text); schema wants
        # removed entries to have old_text -- "" is meaningful here.

        ground_truth = GroundTruth(
            pair_id="pair3",
            expected_deltas=expected,
            notes=(
                "A.dxf and B.dxf are authored programmatically (see "
                "scripts/synthesize_pairs.py::_author_pair3_drawing) because PDF->DXF "
                "conversion flattens text to outlines. The moved valve's LABEL text also "
                "moves with GT3-MOVE (same string, new position) -- position-only text "
                "moves are expected to align as unchanged-or-modified, never add/remove."
            ),
        )
        (out_dir / "ground_truth.json").write_text(
            ground_truth.model_dump_json(indent=2), encoding="utf-8"
        )
        manifest = PairManifest(
            pair_id="pair3",
            description="DWG-format pair: authored DXF P&ID schematic vs edited revision "
            "(text change, moved block, dimension revision, added valve, removed line).",
            doc_a=DocRef(
                path=_rel(path_a), pid="pair3_A", format="dwg", revision_label="rev A"
            ),
            doc_b=DocRef(
                path=_rel(path_b), pid="pair3_B", format="dwg", revision_label="rev B"
            ),
            ground_truth_path=_rel(out_dir / "ground_truth.json"),
            created_by=CREATED_BY,
        )
        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        log.info("pair3 built", extra={"extra_fields": {"edits": len(expected)}})


def main() -> None:
    with tracing.trace("synthesize_pairs"):
        assert LIFT_GAS.exists() and EXPORT_GAS.exists(), "original sample PDFs missing"
        build_pair1()
        build_pair2()
        build_pair3()
        build_pair4()
    sizes = {
        p.relative_to(SAMPLES).as_posix(): f"{p.stat().st_size / 1024:.0f} KB"
        for p in sorted(SAMPLES.rglob("*"))
        if p.is_file() and p.suffix in {".pdf", ".json", ".dxf"}
    }
    print(json.dumps(sizes, indent=2))
    print("All pairs built and verified.")


if __name__ == "__main__":
    main()
