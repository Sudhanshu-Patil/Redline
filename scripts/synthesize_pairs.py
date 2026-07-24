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


def main() -> None:
    with tracing.trace("synthesize_pairs"):
        assert LIFT_GAS.exists() and EXPORT_GAS.exists(), "original sample PDFs missing"
        build_pair1()
        build_pair2()
        build_pair4()
    sizes = {
        p.relative_to(SAMPLES).as_posix(): f"{p.stat().st_size / 1024:.0f} KB"
        for p in sorted(SAMPLES.rglob("*"))
        if p.is_file() and p.suffix in {".pdf", ".json"}
    }
    print(json.dumps(sizes, indent=2))
    print("All pairs built and verified.")


if __name__ == "__main__":
    main()
