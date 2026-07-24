"""Shared test fixtures and factories, imported across test modules to avoid
duplicating fake clients / canonical-model builders per file."""

import itertools

import numpy as np

from src.canonical.model import BBox, Element, make_element_id


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


_id_counter = itertools.count()


def make_positioned_element(
    text: str, element_type: str = "tag", seq: int = 0, x0: float = 0.1, y0: float = 0.1,
    x1: float = 0.2, y1: float = 0.2, conf: float = 1.0, **attrs: str,
) -> Element:  # fmt: skip
    """Element with a bbox given directly in normalized-ish page coordinates
    (page_width=page_height=1, so raw == normalized) -- used by delta engine
    tests where exact relative positions matter.

    `seq` is accepted for call-site readability but NOT used for id
    generation -- a global counter guarantees every element gets a unique id
    regardless of which "side" (A or B) it's built for, so accidental id
    collisions can never mask a real matching bug.
    """
    del seq
    return Element(
        id=make_element_id("test", "test", next(_id_counter)),
        type=element_type,  # type: ignore[arg-type]
        text=text,
        bbox=BBox(page=0, x0=x0, y0=y0, x1=x1, y1=y1, page_width=1.0, page_height=1.0),
        attributes=attrs,
        source_adapter="test",
        extraction_confidence=conf,
    )


class FakeEmbedder:
    """Deterministic stand-in for the real embedder: each text's vector is
    supplied explicitly by the test, so pairwise cosine similarity is
    exactly what the test intends. Vectors must be pre-normalized (unit
    length) to match the real embedder's `normalize_embeddings=True`
    contract, since src.delta.align computes similarity as a raw dot product.
    """

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectors[t] for t in texts], dtype=float)


class NoVisionClient:
    """A VisionReader that reports unconfigured and fails loudly if called --
    for integration tests that must not touch the network."""

    @property
    def is_configured(self) -> bool:
        return False

    def read_image_text(self, png_bytes: bytes, context_hint: str = "") -> str:
        raise AssertionError("vision client must not be called when unconfigured")
