"""Native PDF adapter: pymupdf text extraction -> canonical elements.

Classification is deterministic regex + local layout heuristics grounded in the
two real sample P&IDs (see scratchpad inspection notes) rather than guessed
patterns. Known, deliberate scope boundaries:

- Vector/graphic symbols (valve triangles, instrument bubble circles) are not
  shape-detected; the *tag text* next to/inside them is what the delta engine
  needs, and that text is extracted directly. `geometry` elements are a DWG
  adapter concept (plan §3) where layer/entity_type actually add signal.
- Table structure (title block, revision table) is not detected as
  `table_cell` — these P&IDs are diagram-heavy, not table-heavy documents.
- Instrument-bubble clustering is a local, block-scoped heuristic (function
  code immediately followed by a 3-6 digit loop number, optionally a 1-3
  digit unit code). Real drawings sometimes split one bubble's annotations
  (tag, loop number, HH/LL setpoint) across adjacent-but-separate pymupdf
  text blocks (observed for PIT-9062 in the Lift Gas Compressor P&ID) — in
  that case each block's fragment is still extracted correctly on its own,
  just not merged into a single element. Documented as a known limitation,
  not silently hidden.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

from src.canonical.model import BBox, CanonicalDocument, Element, ElementType, make_element_id
from src.ingest.base import IngestAdapter
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

RawBBox = tuple[float, float, float, float]
RawLine = tuple[str, RawBBox]

# --- Patterns grounded in the sample P&IDs (see PROVENANCE.md) ---

_NOTE_MARKER_RE = re.compile(r"^\d{1,3}(-\d{1,3})?\.$")
# Some notes carry the marker and the start of the body on one physical line
# (observed from note 16 onward in the Lift Gas Compressor P&ID -- notes 1-15
# have the bare marker on its own line, later ones don't). Both layouts are
# real, so both are handled and normalized to the same element shape.
_NOTE_INLINE_RE = re.compile(r"^(\d{1,3}(?:-\d{1,3})?)\.\s+(\S.*)$")
_SECTION_HEADER_RE = re.compile(r"^(NOTES|HOLDS|WORK PACKS?):$")
_NOTE_REF_RE = re.compile(r"^NOTE\s+\d+(\s*,\s*\d+)*$")

# SIZE"-SERVICE-UNIT-SEQ-SPEC-INSULATION, e.g. 6"-VF-43-9029-AC21S-00
_LINE_NUMBER_RE = re.compile(r'^\d+(\.\d+)?"-[A-Z]{2,4}-\d{2,4}-\d{3,6}-[A-Z0-9]+-\d{2}$')

# UNIT + EQUIPMENT-CODE + SEQ (+optional parallel-unit letter), e.g. 26BL9072, 43GT9019
_VALVE_TAG_RE = re.compile(r"^\d{2}[A-Z]{2,4}\d{3,5}[A-Z]?$")

# UNIT-DISCIPLINE-SEQ, e.g. 26-HA-911, 26-KA-902
_EQUIP_TAG_RE = re.compile(r"^\d{2}-[A-Z]{1,3}-\d{3,4}[A-Z]?$")

_SIZE_TRANSITION_RE = re.compile(r'^\d+"?\s*[xX]\s*\d+"?$')

_SETPOINT_INLINE_RE = re.compile(r"^(SP|SET\s*PRESSURE)\s*=\s*([\d.]+)\s*(.*)$", re.IGNORECASE)
_SETPOINT_LIMIT_RE = re.compile(r"^(HH|LL|H|L)\s*[:=]\s*([\d.]+)$")

# Loop numbers are 3-6 digits with an optional parallel-unit letter (9066A);
# unit/area codes are 1-3 digits. This distinction is what keeps e.g. "PI" +
# "26" (function code + unit code, not a loop number) from being
# mis-clustered into an instrument_loop — see docstring above.
_INSTRUMENT_LOOP_NUM_RE = re.compile(r"^\d{3,6}[A-Z]?$")
_UNIT_CODE_RE = re.compile(r"^\d{1,3}$")

# Hyphenated single-token form ("PIT-9062", "PSV-9066A") — how DWG ATTRIBs
# and prose references carry instrument tags. Function code validated against
# INSTRUMENT_FUNCTION_CODES at match time.
_HYPHENATED_INSTRUMENT_RE = re.compile(r"^([A-Z]{2,5})-(\d{3,6}[A-Z]?)$")

_SINGLE_LETTER_RE = re.compile(r"^[A-Z]$")

_VALVE_STATUS_FLAGS = {"LO", "LC", "CSO", "CSC", "ZSO", "ZSC", "NO", "NC"}

INSTRUMENT_FUNCTION_CODES = {
    "PIT", "PI", "PIC", "PDIT", "PDI", "PDT", "PDIC",
    "TIT", "TIC", "TI", "TE", "TT",
    "FIT", "FIC", "FI", "FE", "FT", "FQI",
    "LIT", "LIC", "LI", "LT", "LG",
    "PSV", "PSE", "PSI", "PSH", "PSL", "PSHH", "PSLL",
    "XV", "HV", "ESDV", "RS",
}  # fmt: skip


def _union_bbox(boxes: list[RawBBox]) -> RawBBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


class ElementDraft:
    __slots__ = ("type", "text", "raw_bbox", "attributes")

    def __init__(
        self,
        type_: ElementType,
        text: str,
        raw_bbox: RawBBox,
        attributes: dict[str, str] | None = None,
    ):
        self.type = type_
        self.text = text
        self.raw_bbox = raw_bbox
        self.attributes = attributes or {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ElementDraft):
            return NotImplemented
        return (
            self.type == other.type
            and self.text == other.text
            and self.attributes == other.attributes
        )

    def __repr__(self) -> str:
        return f"ElementDraft({self.type!r}, {self.text!r}, attributes={self.attributes!r})"


def classify_block_lines(lines: list[RawLine]) -> list[ElementDraft]:
    """Classify one text block's lines (in original document order) into elements.

    Pure function, independent of pymupdf, so classification rules are unit
    testable against hand-built line lists without a real PDF.
    """
    drafts: list[ElementDraft] = []
    i = 0
    n = len(lines)
    while i < n:
        text, bbox = lines[i]
        stripped = text.strip()

        m_marker = _NOTE_MARKER_RE.match(stripped)
        m_inline = _NOTE_INLINE_RE.match(stripped) if not m_marker else None
        if m_marker or m_inline:
            if m_marker:
                note_number = stripped.rstrip(".")
                body_parts: list[str] = []
            else:
                assert m_inline is not None
                note_number = m_inline.group(1)
                body_parts = [m_inline.group(2)]
            body_boxes = [bbox]

            j = i + 1
            while j < n:
                nxt_text, nxt_bbox = lines[j]
                nxt_stripped = nxt_text.strip()
                if (
                    _NOTE_MARKER_RE.match(nxt_stripped)
                    or _NOTE_INLINE_RE.match(nxt_stripped)
                    or _SECTION_HEADER_RE.match(nxt_stripped)
                ):
                    break
                body_parts.append(nxt_stripped)
                body_boxes.append(nxt_bbox)
                j += 1

            body = " ".join(body_parts).strip()
            drafts.append(
                ElementDraft(
                    "note",
                    f"{note_number}. {body}".strip(),
                    _union_bbox(body_boxes),
                    {"kind": "definition", "note_number": note_number},
                )
            )
            i = j
            continue

        if _NOTE_REF_RE.match(stripped):
            refs = re.findall(r"\d+", stripped)
            drafts.append(
                ElementDraft("note", stripped, bbox, {"kind": "reference", "refs": ",".join(refs)})
            )
            i += 1
            continue

        if _LINE_NUMBER_RE.match(stripped):
            drafts.append(ElementDraft("line_number", stripped, bbox))
            i += 1
            continue

        m = _HYPHENATED_INSTRUMENT_RE.match(stripped)
        if m and m.group(1) in INSTRUMENT_FUNCTION_CODES:
            drafts.append(
                ElementDraft(
                    "instrument_loop",
                    stripped,
                    bbox,
                    {"function": m.group(1), "loop_number": m.group(2)},
                )
            )
            i += 1
            continue

        if _VALVE_TAG_RE.match(stripped):
            drafts.append(ElementDraft("valve", stripped, bbox))
            i += 1
            continue

        if _EQUIP_TAG_RE.match(stripped):
            drafts.append(ElementDraft("tag", stripped, bbox))
            i += 1
            continue

        m = _SETPOINT_INLINE_RE.match(stripped)
        if m:
            drafts.append(
                ElementDraft(
                    "setpoint",
                    stripped,
                    bbox,
                    {"setpoint_type": "SP", "value": m.group(2), "unit": m.group(3).strip()},
                )
            )
            i += 1
            continue

        m = _SETPOINT_LIMIT_RE.match(stripped)
        if m:
            drafts.append(
                ElementDraft(
                    "setpoint", stripped, bbox, {"setpoint_type": m.group(1), "value": m.group(2)}
                )
            )
            i += 1
            continue

        if _SIZE_TRANSITION_RE.match(stripped):
            drafts.append(ElementDraft("dimension", stripped, bbox, {"kind": "size_transition"}))
            i += 1
            continue

        if stripped in INSTRUMENT_FUNCTION_CODES and i + 1 < n:
            loop_text, loop_bbox = lines[i + 1]
            loop_stripped = loop_text.strip()
            if _INSTRUMENT_LOOP_NUM_RE.match(loop_stripped):
                consumed = [bbox, loop_bbox]
                attrs = {"function": stripped, "loop_number": loop_stripped}
                advance = 2
                if i + 2 < n:
                    unit_text, unit_bbox = lines[i + 2]
                    unit_stripped = unit_text.strip()
                    if _UNIT_CODE_RE.match(unit_stripped):
                        attrs["unit"] = unit_stripped
                        consumed.append(unit_bbox)
                        advance = 3
                drafts.append(
                    ElementDraft(
                        "instrument_loop",
                        f"{stripped}-{loop_stripped}",
                        _union_bbox(consumed),
                        attrs,
                    )
                )
                i += advance
                continue

        if stripped in _VALVE_STATUS_FLAGS:
            drafts.append(ElementDraft("text_block", stripped, bbox, {"kind": "valve_status_flag"}))
            i += 1
            continue

        if _SINGLE_LETTER_RE.match(stripped):
            drafts.append(ElementDraft("text_block", stripped, bbox, {"kind": "flag"}))
            i += 1
            continue

        drafts.append(ElementDraft("text_block", stripped, bbox))
        i += 1

    return drafts


class PdfNativeAdapter(IngestAdapter):
    format = "pdf_native"

    def ingest(
        self, path: Path, pid: str, revision_label: str | None = None
    ) -> CanonicalDocument:
        with tracing.span("pdf_native.ingest", pid=pid, path=str(path)) as sp:
            fitz_doc = fitz.open(path)
            elements: list[Element] = []
            seq = 0
            for page_index, page in enumerate(fitz_doc):
                page_width, page_height = page.rect.width, page.rect.height
                page_dict = page.get_text("dict")
                for block in page_dict["blocks"]:
                    if block.get("type") != 0:
                        continue
                    raw_lines: list[RawLine] = []
                    for line in block["lines"]:
                        line_text = "".join(span["text"] for span in line["spans"])
                        line_text = " ".join(line_text.split())
                        if not line_text:
                            continue
                        raw_lines.append((line_text, tuple(line["bbox"])))
                    if not raw_lines:
                        continue
                    for draft in classify_block_lines(raw_lines):
                        x0, y0, x1, y1 = draft.raw_bbox
                        elements.append(
                            Element(
                                id=make_element_id(pid, self.format, seq),
                                type=draft.type,
                                text=draft.text,
                                bbox=BBox(
                                    page=page_index,
                                    x0=x0,
                                    y0=y0,
                                    x1=x1,
                                    y1=y1,
                                    page_width=page_width,
                                    page_height=page_height,
                                    unit="pdf_points",
                                ),
                                attributes=draft.attributes,
                                source_adapter=self.format,
                                # Native embedded text -> extraction is exact, not OCR'd.
                                extraction_confidence=1.0,
                            )
                        )
                        seq += 1

            page_count = fitz_doc.page_count
            canonical_doc = CanonicalDocument(
                pid=pid,
                format=self.format,
                revision_label=revision_label,
                page_count=page_count,
                elements=elements,
                raw_source_path=str(path),
            )
            sp["elements_extracted"] = len(elements)
            sp["pages"] = page_count
            log.info(
                "pdf_native ingest complete",
                extra={
                    "extra_fields": {"pid": pid, "n_elements": len(elements), "pages": page_count}
                },
            )
            return canonical_doc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a native PDF into canonical JSON.")
    parser.add_argument("path", type=Path)
    parser.add_argument("pid")
    parser.add_argument("--revision-label", default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON here instead of stdout")
    args = parser.parse_args()

    result = PdfNativeAdapter().ingest(args.path, args.pid, args.revision_label)
    output = result.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote {len(result.elements)} elements to {args.out}")
    else:
        print(output)
