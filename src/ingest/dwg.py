"""DWG/DXF adapter: ezdxf parse -> canonical elements (plan §2, §3).

Reads DXF natively. Genuine binary .dwg input is auto-converted to DXF first
when an ODA File Converter is available (config ODA_FILE_CONVERTER, or on
PATH); otherwise a clear error explains the one-time setup. This mirrors the
plan's stance: ezdxf against a DXF export, conversion provenance explicit.

What becomes an element (the DWG format's richer structure earning its keep,
plan §3):

- TEXT / MTEXT     -> classified through the SAME classify_block_lines rules
                      the PDF adapters use (tags, setpoints, notes, ...), so
                      cross-format text keys match; attributes carry
                      layer/entity_type.
- INSERT           -> one `geometry` element per block reference
                      (attributes: block_name, layer, entity_type=INSERT),
                      plus one element per ATTRIB so tagged symbols (e.g. an
                      instrument bubble with TAG=PIT-9062) expose their text
                      to exact-key matching.
- DIMENSION        -> `dimension` element; text is the override if set, else
                      the measured value rounded per config.
- LINE/LWPOLYLINE/
  CIRCLE/ARC/POLYLINE -> `geometry` elements with empty text; layer +
                      entity_type + bbox make them matchable by the delta
                      engine's geometry rule (§4.2).

Bboxes are model-space units (unit="dxf_units"), page 0, with page dimensions
taken from the union of all element bboxes so `.normalized` works for
cross-format comparison.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import ezdxf
import ezdxf.bbox
from ezdxf.document import Drawing
from ezdxf.entities import Dimension, DXFGraphic, Insert, MText

from src.canonical.model import BBox, CanonicalDocument, Element, ElementType, make_element_id
from src.config import settings
from src.ingest.base import IngestAdapter
from src.ingest.pdf_native import classify_block_lines
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

_GEOMETRY_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "SPLINE", "ELLIPSE"}

_ODA_CANDIDATES = [
    Path("C:/Program Files/ODA/ODAFileConverter/ODAFileConverter.exe"),
]


def resolve_oda_converter() -> str | None:
    if settings.oda_file_converter:
        return settings.oda_file_converter
    on_path = shutil.which("ODAFileConverter")
    if on_path:
        return on_path
    for candidate in _ODA_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def convert_dwg_to_dxf(dwg_path: Path) -> Path:
    """Convert a binary .dwg to .dxf via ODA File Converter (folder-based CLI)."""
    converter = resolve_oda_converter()
    if converter is None:
        raise RuntimeError(
            f"{dwg_path.name} is a binary DWG; converting it requires the free ODA File "
            "Converter (set ODA_FILE_CONVERTER or add ODAFileConverter to PATH). "
            "Alternatively export the drawing to DXF and ingest that directly."
        )
    with tracing.span("dwg.oda_convert", src=str(dwg_path)) as sp:
        out_dir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
        in_dir = out_dir / "in"
        in_dir.mkdir()
        shutil.copy2(dwg_path, in_dir / dwg_path.name)
        # ODAFileConverter InputDir OutputDir Version FileType Recurse Audit
        subprocess.run(
            [converter, str(in_dir), str(out_dir), "ACAD2018", "DXF", "0", "1"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        produced = out_dir / (dwg_path.stem + ".dxf")
        if not produced.exists():
            raise RuntimeError(f"ODA converter produced no DXF for {dwg_path.name}")
        sp["output"] = str(produced)
        return produced


def _entity_bbox(entity: DXFGraphic) -> tuple[float, float, float, float] | None:
    box = ezdxf.bbox.extents([entity], fast=True)
    if not box.has_data:
        return None
    return (box.extmin.x, box.extmin.y, box.extmax.x, box.extmax.y)


class _Draft:
    __slots__ = ("type", "text", "raw_bbox", "attributes")

    def __init__(
        self,
        type_: ElementType,
        text: str,
        raw_bbox: tuple[float, float, float, float],
        attributes: dict[str, str],
    ):
        self.type = type_
        self.text = text
        self.raw_bbox = raw_bbox
        self.attributes = attributes


def _classify_text(text: str) -> tuple[ElementType, dict[str, str]]:
    """Run one text string through the shared classifier rules."""
    drafts = classify_block_lines([(text, (0.0, 0.0, 1.0, 1.0))])
    if len(drafts) == 1:
        return drafts[0].type, dict(drafts[0].attributes)
    # Multi-line text that classified into several drafts: keep it whole as a
    # text_block; per-draft splitting would need per-line bboxes DXF lacks.
    return "text_block", {}


class DwgAdapter(IngestAdapter):
    format = "dwg"

    def ingest(
        self, path: Path, pid: str, revision_label: str | None = None
    ) -> CanonicalDocument:
        with tracing.span("dwg.ingest", pid=pid, path=str(path)) as sp:
            source_path = path
            if path.suffix.lower() == ".dwg":
                source_path = convert_dwg_to_dxf(path)
                sp["converted_from_dwg"] = True

            doc: Drawing = ezdxf.readfile(source_path)
            drafts: list[_Draft] = []

            for entity in doc.modelspace():
                etype = entity.dxftype()
                layer = str(entity.dxf.layer)
                bbox = _entity_bbox(entity)
                if bbox is None:
                    continue

                if etype == "TEXT":
                    text = str(entity.dxf.text).strip()
                    if not text:
                        continue
                    el_type, attrs = _classify_text(text)
                    attrs.update({"layer": layer, "entity_type": etype})
                    drafts.append(_Draft(el_type, text, bbox, attrs))

                elif etype == "MTEXT":
                    plain = cast(MText, entity).plain_text(split=False)
                    text = cast(str, plain).strip()
                    if not text:
                        continue
                    lines = [(ln, bbox) for ln in text.splitlines() if ln.strip()]
                    classified = classify_block_lines(lines)
                    for d in classified:
                        attrs = dict(d.attributes)
                        attrs.update({"layer": layer, "entity_type": etype})
                        drafts.append(_Draft(d.type, d.text, d.raw_bbox, attrs))

                elif etype == "INSERT":
                    block_name = str(entity.dxf.name)
                    drafts.append(
                        _Draft(
                            "geometry",
                            block_name,
                            bbox,
                            {
                                "layer": layer,
                                "entity_type": "INSERT",
                                "block_name": block_name,
                            },
                        )
                    )
                    for attrib in cast(Insert, entity).attribs:
                        atext = str(attrib.dxf.text).strip()
                        if not atext:
                            continue
                        abox = _entity_bbox(attrib) or bbox
                        el_type, attrs = _classify_text(atext)
                        attrs.update(
                            {
                                "layer": layer,
                                "entity_type": "ATTRIB",
                                "attrib_tag": str(attrib.dxf.tag),
                                "parent_block": block_name,
                            }
                        )
                        drafts.append(_Draft(el_type, atext, abox, attrs))

                elif etype == "DIMENSION":
                    override = str(entity.dxf.text).strip()
                    if override and override != "<>":
                        text = override
                    else:
                        try:
                            measurement = cast(Dimension, entity).get_measurement()
                            text = f"{float(measurement):.{settings.dwg_dim_decimals}f}"
                        except Exception:
                            text = ""
                    drafts.append(
                        _Draft(
                            "dimension",
                            text,
                            bbox,
                            {"layer": layer, "entity_type": etype, "kind": "dimension"},
                        )
                    )

                elif etype in _GEOMETRY_TYPES:
                    drafts.append(
                        _Draft(
                            "geometry",
                            "",
                            bbox,
                            {"layer": layer, "entity_type": etype},
                        )
                    )

            if not drafts:
                raise RuntimeError(f"no supported entities found in {source_path.name}")

            min_x = min(d.raw_bbox[0] for d in drafts)
            min_y = min(d.raw_bbox[1] for d in drafts)
            max_x = max(d.raw_bbox[2] for d in drafts)
            max_y = max(d.raw_bbox[3] for d in drafts)
            page_w = (max_x - min_x) or 1.0
            page_h = (max_y - min_y) or 1.0

            elements = [
                Element(
                    id=make_element_id(pid, self.format, seq),
                    type=d.type,
                    text=d.text,
                    bbox=BBox(
                        page=0,
                        x0=d.raw_bbox[0] - min_x,
                        y0=d.raw_bbox[1] - min_y,
                        x1=d.raw_bbox[2] - min_x,
                        y1=d.raw_bbox[3] - min_y,
                        page_width=page_w,
                        page_height=page_h,
                        unit="dxf_units",
                    ),
                    attributes=d.attributes,
                    source_adapter=self.format,
                    extraction_confidence=1.0,
                )
                for seq, d in enumerate(drafts)
            ]

            result = CanonicalDocument(
                pid=pid,
                format=self.format,
                revision_label=revision_label,
                page_count=1,
                elements=elements,
                raw_source_path=str(path),
            )
            sp["elements_extracted"] = len(elements)
            log.info(
                "dwg ingest complete",
                extra={"extra_fields": {"pid": pid, "n_elements": len(elements)}},
            )
            return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a DXF/DWG into canonical JSON.")
    parser.add_argument("path", type=Path)
    parser.add_argument("pid")
    parser.add_argument("--revision-label", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = DwgAdapter().ingest(args.path, args.pid, args.revision_label)
    output = result.model_dump_json(indent=2)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Wrote {len(result.elements)} elements to {args.out}")
    else:
        print(output)
