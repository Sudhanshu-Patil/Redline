"""Delta markup overlay (plan §8): draws the computed Delta list back onto
the source document(s) as colored bounding-box highlights -- green=added,
red=removed, amber=modified (STATUS_COLORS, the same convention delta/report.py
uses for its charts, so a reviewer sees one consistent color language across
the report and the annotated drawing).

Two rendering paths, per plan §8:

- PDF (native or scanned): both adapters store bboxes in pdf_points matching
  the *source* PDF's own page.rect exactly (plan §3), so a delta's bbox can
  be drawn directly onto a copy of the real PDF via pymupdf's annotation API
  -- no coordinate transform, pixel-accurate registration. Each box also
  carries a native PDF popup annotation (`Annot.set_info`) as its tooltip,
  viewable in any PDF reader.

- DWG/DXF: there's no equivalent "annotate the real file" API for DXF in
  this stack, so ezdxf's matplotlib backend (ezdxf.addons.drawing) rasterizes
  the drawing first. Overlay boxes are added as matplotlib patches in the
  DXF's own model-space data coordinates (recovered by reversing the
  canonical model's page-normalization offset -- see `_dxf_extents`) rather
  than pixel/fractional coordinates: `Frontend.draw_layout` auto-fits and
  re-pads the axes limits during drawing (measured: a requested xlim of
  (10, 150) came out as (3, 157) after drawing), so only data-coordinate
  placement is robust to that padding -- pixel-fraction math would have to
  fight it and still be off by the pad amount.
"""

import io
from pathlib import Path
from typing import Literal

import ezdxf
import ezdxf.bbox
import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ezdxf.addons.drawing import Frontend, RenderContext  # noqa: E402
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.canonical.model import BBox, CanonicalDocument  # noqa: E402
from src.delta.colors import STATUS_COLORS  # noqa: E402
from src.delta.engine import Delta  # noqa: E402
from src.observability import tracing  # noqa: E402
from src.observability.logging import get_logger  # noqa: E402

log = get_logger(__name__)

Side = Literal["a", "b"]


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _change_label(delta: Delta) -> str:
    if delta.change_type == "added":
        return f"added {delta.element_type}: {delta.new_text or ''}"
    if delta.change_type == "removed":
        return f"removed {delta.element_type}: {delta.old_text or ''}"
    return f"modified {delta.element_type}: {delta.old_text or ''} -> {delta.new_text or ''}"


def _boxes_for_side(deltas: list[Delta], side: Side) -> list[tuple[Delta, BBox]]:
    """Which delta+bbox pairs belong on which side's rendering: side 'a'
    shows what used to be there (removed, and modified's old position);
    side 'b' shows what's there now (added, and modified's new position)."""
    result = []
    for d in deltas:
        if side == "a" and d.change_type in ("removed", "modified") and d.old_bbox:
            result.append((d, d.old_bbox))
        elif side == "b" and d.change_type in ("added", "modified") and d.new_bbox:
            result.append((d, d.new_bbox))
    return result


# --- PDF (native / scanned) ---


def _annotate_pdf_doc(doc: CanonicalDocument, deltas: list[Delta], side: Side) -> fitz.Document:
    with tracing.span("markup.annotate_pdf", pid=doc.pid, side=side) as sp:
        pdf = fitz.open(doc.raw_source_path)
        boxes = _boxes_for_side(deltas, side)
        for delta, bbox in boxes:
            page = pdf[bbox.page]
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=_hex_to_rgb01(STATUS_COLORS[delta.change_type]))
            annot.set_border(width=1.5)
            annot.set_opacity(0.9)
            annot.set_info(content=_change_label(delta))
            annot.update()
        if boxes:
            _add_pdf_legend(pdf[boxes[0][1].page])
        sp["boxes"] = len(boxes)
        return pdf


def _add_pdf_legend(page: fitz.Page) -> None:
    """One small freetext annotation per status, stacked in the top-left
    corner with an opaque white fill so the legend stays legible regardless
    of what drawing content happens to sit underneath -- no fixed corner is
    guaranteed empty across arbitrary P&IDs, so legibility-over-content is
    the safer default (freetext annots have one text_color each, hence three
    annotations rather than one multi-color block)."""
    row_h = 14
    x0, y0 = page.rect.x0 + 8, page.rect.y0 + 8
    for i, (key, label) in enumerate(_LEGEND_LABELS):
        rect = fitz.Rect(x0, y0 + i * row_h, x0 + 90, y0 + row_h + i * row_h)
        annot = page.add_freetext_annot(
            rect, label, fontsize=8,
            text_color=_hex_to_rgb01(STATUS_COLORS[key]), fill_color=(1, 1, 1),
        )  # fmt: skip
        annot.set_border(width=0.5)
        annot.set_opacity(0.95)
        annot.update()


_LEGEND_LABELS: list[tuple[str, str]] = [
    ("added", "Added"), ("removed", "Removed"), ("modified", "Modified"),
]  # fmt: skip


def save_pdf_markup(
    doc: CanonicalDocument, deltas: list[Delta], side: Side, out_path: Path
) -> Path:
    pdf = _annotate_pdf_doc(doc, deltas, side)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(out_path)
    pdf.close()
    return out_path


def render_pdf_markup_png(
    doc: CanonicalDocument, deltas: list[Delta], side: Side, dpi: int = 120
) -> bytes:
    boxes = _boxes_for_side(deltas, side)
    page_index = boxes[0][1].page if boxes else 0
    pdf = _annotate_pdf_doc(doc, deltas, side)
    png_bytes: bytes = pdf[page_index].get_pixmap(dpi=dpi).tobytes("png")
    pdf.close()
    return png_bytes


# --- DWG/DXF ---


def _resolve_dxf_path(doc: CanonicalDocument) -> Path:
    path = Path(doc.raw_source_path)
    if path.suffix.lower() != ".dwg":
        return path
    from src.ingest.dwg import convert_dwg_to_dxf

    return convert_dwg_to_dxf(path)


def _dxf_extents(msp: ezdxf.layouts.Modelspace) -> tuple[float, float, float, float]:
    """min_x, min_y, max_x, max_y in true DXF model-space units -- the same
    computation src/ingest/dwg.py used to normalize canonical bboxes to a
    (0,0) origin, so `canonical_x + min_x` recovers the real DXF coordinate.
    """
    ext = ezdxf.bbox.extents(msp, fast=True)
    return ext.extmin.x, ext.extmin.y, ext.extmax.x, ext.extmax.y


def _render_dxf_figure(
    doc: CanonicalDocument, deltas: list[Delta], side: Side, dpi: int
) -> Figure:
    with tracing.span("markup.render_dxf", pid=doc.pid, side=side) as sp:
        dxf_path = _resolve_dxf_path(doc)
        dxf_doc = ezdxf.readfile(dxf_path)
        msp = dxf_doc.modelspace()
        min_x, min_y, max_x, max_y = _dxf_extents(msp)
        w, h = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)

        fig = plt.figure(figsize=(w / 72, h / 72), dpi=dpi)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.axis("off")

        ctx = RenderContext(dxf_doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)

        boxes = _boxes_for_side(deltas, side)
        for delta, bbox in boxes:
            x0, y0 = bbox.x0 + min_x, bbox.y0 + min_y
            width, height = max(bbox.x1 - bbox.x0, 0.01), max(bbox.y1 - bbox.y0, 0.01)
            color = _hex_to_rgb01(STATUS_COLORS[delta.change_type])
            ax.add_patch(
                Rectangle((x0, y0), width, height, fill=False, edgecolor=color, linewidth=1.5)
            )
        if boxes:
            handles = [
                Rectangle((0, 0), 1, 1, fill=False, edgecolor=_hex_to_rgb01(STATUS_COLORS[k]))
                for k, _ in _LEGEND_LABELS
            ]
            ax.legend(
                handles, [label for _, label in _LEGEND_LABELS],
                loc="upper right", fontsize=8, framealpha=0.85,
            )  # fmt: skip
        sp["boxes"] = len(boxes)
        return fig


def save_dxf_markup(
    doc: CanonicalDocument, deltas: list[Delta], side: Side, out_path: Path, dpi: int = 150
) -> Path:
    fig = _render_dxf_figure(doc, deltas, side, dpi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def render_dxf_markup_png(
    doc: CanonicalDocument, deltas: list[Delta], side: Side, dpi: int = 100
) -> bytes:
    fig = _render_dxf_figure(doc, deltas, side, dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# --- format dispatch ---

_EXT_BY_FORMAT = {"pdf_native": "pdf", "pdf_scanned": "pdf", "dwg": "png"}


def render_markup_png(doc: CanonicalDocument, deltas: list[Delta], side: Side) -> bytes:
    if doc.format == "dwg":
        return render_dxf_markup_png(doc, deltas, side)
    return render_pdf_markup_png(doc, deltas, side)


def write_markup(
    doc_a: CanonicalDocument,
    doc_b: CanonicalDocument,
    deltas: list[Delta],
    out_dir: Path,
    basename: str,
) -> dict[str, Path]:
    """Writes one annotated export per side (PDF w/ real annotations for the
    PDF formats, PNG for DWG), named `{basename}_A.<ext>` / `{basename}_B.<ext>`."""
    with tracing.span("markup.write", pid_a=doc_a.pid, pid_b=doc_b.pid) as sp:
        paths: dict[str, Path] = {}
        sides: tuple[tuple[Side, CanonicalDocument], ...] = (("a", doc_a), ("b", doc_b))
        for side, doc in sides:
            ext = _EXT_BY_FORMAT[doc.format]
            out_path = out_dir / f"{basename}_{side.upper()}.{ext}"
            if doc.format == "dwg":
                save_dxf_markup(doc, deltas, side, out_path)
            else:
                save_pdf_markup(doc, deltas, side, out_path)
            paths[side] = out_path
        sp["files"] = [str(p) for p in paths.values()]
        return paths


if __name__ == "__main__":
    import argparse

    from eval.schema import load_manifest
    from src.delta.engine import compute_delta
    from src.ingest.dwg import DwgAdapter
    from src.ingest.pdf_native import PdfNativeAdapter
    from src.ingest.pdf_scanned import PdfScannedAdapter

    _ADAPTERS = {
        "pdf_native": PdfNativeAdapter,
        "pdf_scanned": PdfScannedAdapter,
        "dwg": DwgAdapter,
    }

    parser = argparse.ArgumentParser(description="Render the delta markup overlay for a pair.")
    parser.add_argument("manifest", type=Path, help="path to a pair's manifest.json")
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    doc_a = _ADAPTERS[manifest.doc_a.format]().ingest(
        Path(manifest.doc_a.path),
        pid=manifest.doc_a.pid,
        revision_label=manifest.doc_a.revision_label,
    )
    doc_b = _ADAPTERS[manifest.doc_b.format]().ingest(
        Path(manifest.doc_b.path),
        pid=manifest.doc_b.pid,
        revision_label=manifest.doc_b.revision_label,
    )
    delta_report = compute_delta(doc_a, doc_b)
    written = write_markup(doc_a, doc_b, delta_report.deltas, args.out, basename=manifest.pair_id)
    for side, path in written.items():
        print(f"wrote side {side}: {path}")
