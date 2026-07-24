"""Shared test fixtures and factories, imported across test modules to avoid
duplicating fake clients / canonical-model builders per file."""

from src.canonical.model import BBox, Element


def make_bbox(**overrides: float | int | str) -> BBox:
    """A valid BBox with sensible defaults; override only the fields a test cares about."""
    defaults: dict[str, float | int | str] = {
        "page": 0, "x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 70.0,
        "page_width": 200.0, "page_height": 100.0,
    }  # fmt: skip
    defaults.update(overrides)
    return BBox(**defaults)  # type: ignore[arg-type]


def make_element(
    text: str,
    conf: float = 1.0,
    seq: int = 0,
    element_type: str = "text_block",
    source_adapter: str = "pdf_scanned",
    **bbox_overrides: float | int | str,
) -> Element:
    """A valid Element with sensible defaults; used by scanned-adapter unit tests."""
    return Element(
        id=f"t:{source_adapter}:{seq:05d}",
        type=element_type,  # type: ignore[arg-type]
        text=text,
        bbox=make_bbox(**bbox_overrides),
        source_adapter=source_adapter,
        extraction_confidence=conf,
    )


class NoVisionClient:
    """A VisionReader that reports unconfigured and fails loudly if called --
    for integration tests that must not touch the network."""

    @property
    def is_configured(self) -> bool:
        return False

    def read_image_text(self, png_bytes: bytes, context_hint: str = "") -> str:
        raise AssertionError("vision client must not be called when unconfigured")
