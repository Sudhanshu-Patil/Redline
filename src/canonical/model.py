from typing import Literal

from pydantic import BaseModel, Field, model_validator

ElementType = Literal[
    "tag",
    "instrument_loop",
    "valve",
    "line_number",
    "note",
    "dimension",
    "setpoint",
    "table_cell",
    "text_block",
    "geometry",
]

BBoxUnit = Literal["pdf_points", "pixels", "dxf_units"]


class BBox(BaseModel):
    """Bounding box in the source document's native units.

    `page_width`/`page_height` are the dimensions of that same page in the
    same unit, carried alongside the box so any consumer can derive
    page-relative fractional coordinates (`normalized`) without a side
    lookup. This is what lets the delta engine's bbox-proximity matching
    (plan §4) compare boxes across adapters whose native units differ —
    PDF points vs. scanned-image pixels vs. DXF model-space units.
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    unit: BBoxUnit = "pdf_points"

    @model_validator(mode="after")
    def _check_ordering(self) -> "BBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"BBox has inverted coordinates: {self!r}")
        return self

    @property
    def normalized(self) -> tuple[float, float, float, float]:
        """Page-relative fractional coords (0-1), for cross-format comparison."""
        w = self.page_width or 1.0
        h = self.page_height or 1.0
        return (self.x0 / w, self.y0 / h, self.x1 / w, self.y1 / h)


class Element(BaseModel):
    id: str
    type: ElementType
    text: str
    bbox: BBox
    attributes: dict[str, str] = Field(default_factory=dict)
    source_adapter: str
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class CanonicalDocument(BaseModel):
    pid: str
    format: Literal["pdf_native", "pdf_scanned", "dwg"]
    revision_label: str | None = None
    page_count: int
    elements: list[Element]
    raw_source_path: str


def make_element_id(pid: str, adapter: str, seq: int) -> str:
    """Stable, human-legible element id. Convention shared by every adapter."""
    return f"{pid}:{adapter}:{seq:05d}"
