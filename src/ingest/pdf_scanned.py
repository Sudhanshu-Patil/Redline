"""Scanned PDF adapter: pymupdf rasterization -> tesseract OCR -> canonical
elements, with a vision-LLM second read for low-confidence regions.

Design notes (deliberate choices, see PROVENANCE/README):

- Rasterization uses pymupdf's own renderer rather than pdf2image+poppler.
  pymupdf is already a core dependency and renders identically; this removes
  an entire system-level dependency (poppler) from the install story. The
  plan named pdf2image as the vehicle -- the intent (raster + OCR) is
  unchanged.
- Tesseract word TSV is regrouped into lines/blocks using tesseract's own
  (block_num, par_num, line_num) hierarchy, then each block's lines feed the
  SAME pure classifier the native adapter uses (classify_block_lines). One
  classification ruleset, two extraction front-ends.
- Bboxes are converted from rendered-image pixels back to PDF points
  (72/dpi), so scanned elements share a coordinate space with native ones --
  this is what lets Pair 2 (native A vs scanned B) align on bbox proximity
  and lets the Phase 9 overlay draw on the PDF directly.
- Vision fallback: elements whose OCR confidence < OCR_CONFIDENCE_THRESHOLD
  are re-read by the vision LLM (lowest-confidence first, capped at
  VISION_FALLBACK_MAX_REGIONS per document). If no API key is configured the
  fallback is skipped and elements keep their low-confidence OCR text --
  ingest never hard-fails on a missing key. Original OCR text is preserved
  in attributes["ocr_text"] whenever the LLM replaces it.
"""

import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from src.canonical.model import BBox, CanonicalDocument, Element, make_element_id
from src.config import settings
from src.ingest.base import IngestAdapter
from src.ingest.pdf_native import RawLine, classify_block_lines
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

_WINDOWS_TESSERACT_DEFAULTS = [
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
]


def resolve_tesseract_cmd() -> str | None:
    """Explicit config -> PATH -> well-known install dirs. None if not found."""
    if settings.tesseract_cmd:
        return settings.tesseract_cmd
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _WINDOWS_TESSERACT_DEFAULTS:
        if candidate.exists():
            return str(candidate)
    return None


def tesseract_available() -> bool:
    return resolve_tesseract_cmd() is not None


@dataclass
class OcrLine:
    """One tesseract-detected text line, in rendered-image pixel coords."""

    block_num: int
    line_key: tuple[int, int, int]  # (block, par, line) -- tesseract's hierarchy
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float  # mean word confidence, 0-100


def group_tsv_into_lines(tsv: dict[str, list]) -> list[OcrLine]:
    """Group tesseract image_to_data output (DICT format) into text lines.

    Pure function over the TSV dict so it is unit-testable without tesseract.
    Word rows with conf == -1 (structural rows) or empty text are dropped.
    """
    lines: dict[tuple[int, int, int], list[int]] = {}
    n = len(tsv["text"])
    for i in range(n):
        text = tsv["text"][i]
        conf = float(tsv["conf"][i])
        if conf < 0 or not str(text).strip():
            continue
        key = (tsv["block_num"][i], tsv["par_num"][i], tsv["line_num"][i])
        lines.setdefault(key, []).append(i)

    result: list[OcrLine] = []
    for key in sorted(lines):
        idxs = lines[key]
        words = [str(tsv["text"][i]).strip() for i in idxs]
        confs = [float(tsv["conf"][i]) for i in idxs]
        x0 = min(tsv["left"][i] for i in idxs)
        y0 = min(tsv["top"][i] for i in idxs)
        x1 = max(tsv["left"][i] + tsv["width"][i] for i in idxs)
        y1 = max(tsv["top"][i] + tsv["height"][i] for i in idxs)
        result.append(
            OcrLine(
                block_num=key[0],
                line_key=key,
                text=" ".join(words),
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                confidence=sum(confs) / len(confs),
            )
        )
    return result


# Stray marks OCR'd from line art (bubble circles, leader lines) that glue to
# real tokens: 'PIT \', '( TIT', '9062 /'. Deliberately excludes . " * # -
# which are meaningful in note markers, line sizes, and flags.
_OCR_EDGE_JUNK = "\\/(){}[]|"


def clean_ocr_line_text(text: str) -> str:
    """Trim line-art noise from an OCR line's edges (scanned path only).

    1. Drop leading/trailing whitespace-separated tokens containing no
       alphanumerics at all (a lone '\\' or '(' is never real drawing text).
    2. Strip junk-set characters from the outer edges of what remains.
    Interior punctuation is never touched -- 'SP = 260 bar (g)' keeps its
    structure, '30.' keeps its marker period.
    """
    tokens = text.split()
    while tokens and not any(ch.isalnum() for ch in tokens[0]):
        tokens.pop(0)
    while tokens and not any(ch.isalnum() for ch in tokens[-1]):
        tokens.pop()
    if not tokens:
        return ""
    cleaned = " ".join(tokens)
    return cleaned.lstrip(_OCR_EDGE_JUNK).rstrip(_OCR_EDGE_JUNK).strip()


def cluster_lines_spatially(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Group OCR lines into spatial clusters (vertical stacks with horizontal
    overlap), replacing tesseract's block_num grouping.

    Rationale: under PSM 11 (sparse text) tesseract tends to put every line in
    its own block, so block-scoped classification never sees an instrument
    bubble's function code and loop number together ("PIT" / "9062") and the
    two-line clustering in classify_block_lines can't fire. Real geometry is
    the reliable signal: bubble text is a tight vertical stack, note columns
    are tall stacks -- both are exactly "lines stacked with overlap".

    Join rule between vertically adjacent lines A (above) and B (below):
      - vertical gap (B.y0 - A.y1) < 0.9x the taller line's height, allowing
        slight overlap (> -0.5x), and
      - horizontal overlap of at least 40% of the narrower line's width.
    Transitive closure via union-find; groups and their lines come out in
    reading order.
    """
    n = len(lines)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (lines[i].y0, lines[i].x0))
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for a_pos, i in enumerate(order):
        a = lines[i]
        h_a = a.y1 - a.y0
        for j in order[a_pos + 1 :]:
            b = lines[j]
            h = max(h_a, b.y1 - b.y0)
            gap = b.y0 - a.y1
            if gap > 0.9 * h:
                break  # sorted by y0; everything further is even lower
            if gap < -0.5 * h:
                continue
            overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            min_width = min(a.x1 - a.x0, b.x1 - b.x0)
            if min_width > 0 and overlap >= 0.4 * min_width:
                union(i, j)

    groups: dict[int, list[OcrLine]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    ordered_groups = sorted(
        groups.values(), key=lambda g: (min(ln.y0 for ln in g), min(ln.x0 for ln in g))
    )
    for group in ordered_groups:
        group.sort(key=lambda ln: (ln.y0, ln.x0))
    return ordered_groups


class VisionReader(Protocol):
    """Structural interface for the vision fallback; LLMClient satisfies it."""

    @property
    def is_configured(self) -> bool: ...

    def read_image_text(self, png_bytes: bytes, context_hint: str = "") -> str: ...


def _boxes_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class PdfScannedAdapter(IngestAdapter):
    format = "pdf_scanned"

    def __init__(self, vision_client: VisionReader | None = None) -> None:
        self._vision = vision_client

    def _get_vision(self) -> VisionReader | None:
        if self._vision is not None:
            return self._vision
        if not settings.vision_fallback_enabled:
            return None
        from src.chat.llm import LLMClient  # deferred: anthropic import not needed for pure OCR

        return LLMClient()

    def ingest(
        self, path: Path, pid: str, revision_label: str | None = None
    ) -> CanonicalDocument:
        cmd = resolve_tesseract_cmd()
        if cmd is None:
            raise RuntimeError(
                "tesseract binary not found (set TESSERACT_CMD, or install tesseract)"
            )
        pytesseract.pytesseract.tesseract_cmd = cmd

        with tracing.span("pdf_scanned.ingest", pid=pid, path=str(path)) as sp:
            fitz_doc = fitz.open(path)
            elements: list[Element] = []
            seq = 0
            n_low_conf_total = 0

            for page_index, page in enumerate(fitz_doc):
                page_width_pt, page_height_pt = page.rect.width, page.rect.height
                scale = 72.0 / settings.ocr_dpi  # rendered px -> PDF points

                with tracing.span("pdf_scanned.render", page=page_index) as rsp:
                    pix = page.get_pixmap(dpi=settings.ocr_dpi)
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    rsp["px"] = f"{pix.width}x{pix.height}"

                with tracing.span("pdf_scanned.tesseract", page=page_index) as tsp:
                    tsv = pytesseract.image_to_data(
                        image,
                        output_type=pytesseract.Output.DICT,
                        config=f"--psm {settings.ocr_tesseract_psm}",
                    )
                    ocr_lines = group_tsv_into_lines(tsv)
                    tsp["lines"] = len(ocr_lines)
                    tsp["psm"] = settings.ocr_tesseract_psm

                # Group lines spatially (tesseract's block_num is unreliable
                # under sparse-text PSM) and classify each group.
                for block_lines in cluster_lines_spatially(ocr_lines):
                    raw_lines: list[RawLine] = []
                    kept_lines: list[OcrLine] = []
                    for ln in block_lines:
                        cleaned = clean_ocr_line_text(ln.text)
                        if not cleaned:
                            continue
                        raw_lines.append((cleaned, (ln.x0, ln.y0, ln.x1, ln.y1)))
                        kept_lines.append(ln)
                    if not raw_lines:
                        continue
                    block_lines = kept_lines
                    drafts = classify_block_lines(raw_lines)
                    for draft in drafts:
                        # Confidence: min over OCR lines overlapping the draft's
                        # bbox (a draft may merge several lines; the weakest
                        # link is what we trust it at).
                        contributing = [
                            ln.confidence
                            for ln in block_lines
                            if _boxes_intersect(
                                draft.raw_bbox, (ln.x0, ln.y0, ln.x1, ln.y1)
                            )
                        ]
                        conf01 = (min(contributing) / 100.0) if contributing else 0.0
                        if conf01 < settings.ocr_confidence_threshold / 100.0:
                            n_low_conf_total += 1

                        x0, y0, x1, y1 = draft.raw_bbox
                        elements.append(
                            Element(
                                id=make_element_id(pid, self.format, seq),
                                type=draft.type,
                                text=draft.text,
                                bbox=BBox(
                                    page=page_index,
                                    x0=x0 * scale,
                                    y0=y0 * scale,
                                    x1=x1 * scale,
                                    y1=y1 * scale,
                                    page_width=page_width_pt,
                                    page_height=page_height_pt,
                                    unit="pdf_points",
                                ),
                                attributes=draft.attributes,
                                source_adapter=self.format,
                                extraction_confidence=max(0.0, min(1.0, conf01)),
                            )
                        )
                        seq += 1

                self._apply_vision_fallback(elements, image, scale, page_index)

            page_count = fitz_doc.page_count
            doc = CanonicalDocument(
                pid=pid,
                format=self.format,
                revision_label=revision_label,
                page_count=page_count,
                elements=elements,
                raw_source_path=str(path),
            )
            sp["elements_extracted"] = len(elements)
            sp["low_confidence_elements"] = n_low_conf_total
            log.info(
                "pdf_scanned ingest complete",
                extra={
                    "extra_fields": {
                        "pid": pid,
                        "n_elements": len(elements),
                        "low_conf": n_low_conf_total,
                    }
                },
            )
            return doc

    def _apply_vision_fallback(
        self, elements: list[Element], image: Image.Image, scale: float, page_index: int
    ) -> None:
        """Re-read the worst low-confidence elements on this page via vision LLM."""
        threshold01 = settings.ocr_confidence_threshold / 100.0
        candidates = [
            e
            for e in elements
            if e.bbox.page == page_index
            and e.extraction_confidence < threshold01
            # Vector-art fragments OCR as short symbol runs ('�}', 'Tn *');
            # require some alphanumeric substance before spending an LLM call.
            and sum(ch.isalnum() for ch in e.text) >= 2
        ]
        if not candidates:
            return

        vision = self._get_vision()
        if vision is None or not vision.is_configured:
            with tracing.span("pdf_scanned.vision_fallback", page=page_index) as sp:
                sp["status"] = "skipped_not_configured"
                sp["candidates"] = len(candidates)
            log.warning(
                "vision fallback skipped: LLM not configured",
                extra={"extra_fields": {"low_conf_candidates": len(candidates)}},
            )
            for e in candidates:
                e.attributes["ocr_fallback"] = "unavailable"
            return

        candidates.sort(key=lambda e: e.extraction_confidence)
        selected = candidates[: settings.vision_fallback_max_regions]
        with tracing.span(
            "pdf_scanned.vision_fallback", page=page_index, regions=len(selected)
        ) as sp:
            n_replaced = 0
            for element in selected:
                # Crop with padding, in rendered-image pixel space.
                pad = 6
                px0 = int(element.bbox.x0 / scale) - pad
                py0 = int(element.bbox.y0 / scale) - pad
                px1 = int(element.bbox.x1 / scale) + pad
                py1 = int(element.bbox.y1 / scale) + pad
                crop = image.crop(
                    (max(0, px0), max(0, py0), min(image.width, px1), min(image.height, py1))
                )
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                try:
                    llm_text = vision.read_image_text(
                        buf.getvalue(),
                        context_hint=f"OCR read this as {element.text!r} with low confidence.",
                    )
                except Exception as exc:
                    log.warning(
                        "vision fallback call failed",
                        extra={"extra_fields": {"element": element.id, "error": str(exc)}},
                    )
                    element.attributes["ocr_fallback"] = "error"
                    continue
                if llm_text and llm_text != element.text:
                    element.attributes["ocr_text"] = element.text
                    element.attributes["ocr_fallback"] = "vision_llm"
                    element.text = llm_text
                    element.extraction_confidence = settings.vision_fallback_confidence
                    n_replaced += 1
                elif llm_text == element.text:
                    # LLM agreed with tesseract -- keep text, boost confidence.
                    element.attributes["ocr_fallback"] = "vision_llm_confirmed"
                    element.extraction_confidence = settings.vision_fallback_confidence
                else:
                    element.attributes["ocr_fallback"] = "vision_llm_unreadable"
            sp["replaced"] = n_replaced


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a scanned PDF into canonical JSON.")
    parser.add_argument("path", type=Path)
    parser.add_argument("pid")
    parser.add_argument("--revision-label", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = PdfScannedAdapter().ingest(args.path, args.pid, args.revision_label)
    output = result.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote {len(result.elements)} elements to {args.out}")
    else:
        print(output)
