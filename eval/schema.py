"""Typed schemas for sample-pair manifests and delta ground truth.

Written by scripts/synthesize_pairs.py (Phase 2/4), consumed by the eval
harness (Phase 10) and tests. Keeping them as pydantic models means a pair
directory is either valid or loudly broken -- no ad-hoc JSON drift.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.canonical.model import ElementType


class DocRef(BaseModel):
    path: str  # repo-root-relative, forward slashes
    pid: str
    format: Literal["pdf_native", "pdf_scanned", "dwg"]
    revision_label: str


class PairManifest(BaseModel):
    pair_id: str
    description: str
    doc_a: DocRef
    doc_b: DocRef
    ground_truth_path: str  # repo-root-relative
    created_by: str  # tool + version that produced this pair, for provenance


class GTLocator(BaseModel):
    """Where the change lives on the page.

    bbox is in PDF points of the *source* side of the change: A-side
    coordinates for removed/modified elements, B-side for added ones.
    Approximate on purpose -- eval matches primarily on text, using bbox
    only to disambiguate duplicates (e.g. the two identical PSV setpoint
    strings in Pair 1).
    """

    page: int
    bbox: tuple[float, float, float, float] | None = None
    near_text: str | None = None


class ExpectedDelta(BaseModel):
    gt_id: str
    change_type: Literal["added", "removed", "modified"]
    element_type: ElementType
    old_text: str | None = None  # None for added
    new_text: str | None = None  # None for removed
    locator: GTLocator
    rationale: str


class GroundTruth(BaseModel):
    pair_id: str
    negative_control: bool = False
    expected_deltas: list[ExpectedDelta] = Field(default_factory=list)
    notes: str = ""


def load_manifest(path: Path) -> PairManifest:
    return PairManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_ground_truth(path: Path) -> GroundTruth:
    return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
