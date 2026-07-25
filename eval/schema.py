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


class QAItem(BaseModel):
    """One labeled chat question (plan §10): grounds both the retrieval-
    quality eval (recall@k/MRR against `expected_citation_texts`) and the
    chat groundedness/correctness judge eval. Citations are matched by
    text, not element id, for the same reason GTLocator matches deltas by
    text -- ids are adapter/ingestion-order-dependent, text is stable.
    """

    qa_id: str
    pair_id: str
    question: str
    answerable: bool
    # Texts of elements that would ground a correct answer -- empty for
    # answerable=False (deliberately unanswerable) questions.
    expected_citation_texts: list[str] = Field(default_factory=list)
    expected_answer_summary: str
    # Marks items hand-scored by a human for judge validation (BRIEF's
    # "validate the judge" bar) -- see eval/datasets/human_labels_*.json.
    # Originally a 5-of-15 subset; expanded to all 15 items for pair1
    # (2026-07-25) once a live eval run surfaced real judge output worth
    # checking against every question, not just a sample -- there's no
    # training/tuning step here for "held out" to protect against, so the
    # name just marks "independently human-checked," and there's no reason
    # not to check everything once the actual outputs already exist.
    held_out: bool = False


class QADataset(BaseModel):
    pair_id: str
    items: list[QAItem]


def load_qa_dataset(path: Path) -> QADataset:
    return QADataset.model_validate_json(path.read_text(encoding="utf-8"))


class HumanLabel(BaseModel):
    """A hand-assigned score for one held-out QA item's actual chat output,
    used only to validate the LLM judge against (never fed into scoring the
    system itself) -- see eval/run_eval.py's judge/human agreement report.
    """

    qa_id: str
    correctness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    notes: str = ""


class HumanLabelSet(BaseModel):
    pair_id: str
    labels: list[HumanLabel]


def load_human_labels(path: Path) -> HumanLabelSet:
    return HumanLabelSet.model_validate_json(path.read_text(encoding="utf-8"))
