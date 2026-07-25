"""Eval metrics (plan §10): delta precision/recall/F1 against labeled ground
truth, and judge/human agreement statistics for the chat groundedness eval
(the "validate the judge" bar the brief calls out explicitly).

Delta matching is text-based, not id/bbox-based, for the same reason
GTLocator (eval/schema.py) stores near_text rather than element ids: ids are
adapter/ingestion-order-dependent, text is stable across runs. Pair 2's B
side is OCR'd, so matching allows a fuzzy fallback -- the same
SequenceMatcher ratio and 0.85 threshold tests/test_pdf_scanned_integration.py
already uses for cross-adapter text consistency -- tried only after an exact
match fails.
"""

from difflib import SequenceMatcher

import numpy as np
from pydantic import BaseModel

from eval.schema import ExpectedDelta, GroundTruth
from src.config import settings
from src.delta.engine import Delta, DeltaReport
from src.observability import tracing

FUZZY_MATCH_THRESHOLD = 0.85


def _texts_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= FUZZY_MATCH_THRESHOLD


def _expected_matches_actual(expected: ExpectedDelta, actual: Delta) -> bool:
    return (
        expected.change_type == actual.change_type
        and expected.element_type == actual.element_type
        and _texts_match(expected.old_text, actual.old_text)
        and _texts_match(expected.new_text, actual.new_text)
    )


def _locator_distance(expected: ExpectedDelta, actual: Delta) -> float:
    """Lower is better; disambiguates duplicate-text matches (e.g. Pair 1's
    two identical 'SP = 257 bar (g)' -> 'SP = 260 bar (g)' edits, one per
    PSV) via bbox proximity to the ground truth locator. 0.0 (no
    preference) when either side lacks a bbox to compare."""
    if expected.locator.bbox is None:
        return 0.0
    bbox = actual.new_bbox or actual.old_bbox
    if bbox is None:
        return 0.0
    ex0, ey0, ex1, ey1 = expected.locator.bbox
    ecx, ecy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
    acx, acy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
    return float(np.hypot(ecx - acx, ecy - acy))


class DeltaMatch(BaseModel):
    gt_id: str
    old_text: str | None
    new_text: str | None


class DeltaMatchResult(BaseModel):
    true_positives: list[DeltaMatch]
    false_negatives: list[ExpectedDelta]  # ground truth edits the engine missed
    false_positives: list[Delta]  # engine-reported edits with no matching ground truth


def match_deltas(actual: list[Delta], expected: list[ExpectedDelta]) -> DeltaMatchResult:
    """Greedy bipartite matching, closest-locator-first -- same pattern as
    src/delta/align.py::_greedy_assign, so a duplicate-text ground truth
    entry can't steal the wrong candidate purely by list order."""
    with tracing.span("eval.delta.match", actual=len(actual), expected=len(expected)) as sp:
        candidates = [
            (_locator_distance(e, a), e, a)
            for e in expected
            for a in actual
            if _expected_matches_actual(e, a)
        ]
        candidates.sort(key=lambda c: (c[0], c[1].gt_id))

        used_expected: set[str] = set()
        used_actual_ids: set[int] = set()
        true_positives: list[DeltaMatch] = []
        for _dist, e, a in candidates:
            if e.gt_id in used_expected or id(a) in used_actual_ids:
                continue
            true_positives.append(
                DeltaMatch(gt_id=e.gt_id, old_text=a.old_text, new_text=a.new_text)
            )
            used_expected.add(e.gt_id)
            used_actual_ids.add(id(a))

        false_negatives = [e for e in expected if e.gt_id not in used_expected]
        false_positives = [a for a in actual if id(a) not in used_actual_ids]
        sp["tp"] = len(true_positives)
        sp["fn"] = len(false_negatives)
        sp["fp"] = len(false_positives)
        return DeltaMatchResult(
            true_positives=true_positives,
            false_negatives=false_negatives,
            false_positives=false_positives,
        )


class PRF1(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def precision_recall_f1(tp: int, fp: int, fn: int) -> PRF1:
    # No predictions and nothing expected is a vacuous perfect score, not a
    # divide-by-zero -- only meaningful for Pair 4-style zero-edit cases,
    # which evaluate_negative_control() handles separately in practice.
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF1(
        precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4), tp=tp, fp=fp, fn=fn
    )  # fmt: skip


def _is_below_match_threshold(d: Delta) -> bool:
    """Same floor src/delta/align.py's tier-3 embedding match and
    src/delta/report.py's chart filter use: below this length there's no
    real signal for any matching approach to work with, so this project
    treats it as a known, deliberate precision-over-recall category (see
    PROVENANCE.md's Phase 5 findings) rather than a genuine miss worth
    chasing -- 234 of Pair 1's 237 raw false positives are exactly this.
    """
    text = d.new_text if d.new_text is not None else d.old_text
    if not text:
        return False
    return len(text.strip()) < settings.alignment_min_embed_text_len


class NegativeControlResult(BaseModel):
    warning_raised: bool
    passed: bool


def evaluate_negative_control(report: DeltaReport) -> NegativeControlResult:
    """Pair 4 (plan §6) is two unrelated documents, not a revision pair --
    a correct engine surfaces the low-alignment warning instead of hundreds
    of spurious adds/removes, so pass/fail is on the warning, not P/R."""
    raised = len(report.warnings) > 0
    return NegativeControlResult(warning_raised=raised, passed=raised)


class PairDeltaEval(BaseModel):
    pair_id: str
    negative_control: bool
    prf1: PRF1 | None = None
    # Same TPs/FNs as prf1, but false positives below the matching-confidence
    # floor (single characters / empty fragments -- never real signal, see
    # _is_below_match_threshold) are excluded, so precision reflects genuine
    # misses rather than being dominated by an accepted, documented tradeoff.
    prf1_excluding_noise: PRF1 | None = None
    noise_false_positives: int = 0
    negative_control_result: NegativeControlResult | None = None
    match: DeltaMatchResult | None = None


def evaluate_pair_delta(report: DeltaReport, ground_truth: GroundTruth) -> PairDeltaEval:
    if ground_truth.negative_control:
        return PairDeltaEval(
            pair_id=ground_truth.pair_id,
            negative_control=True,
            negative_control_result=evaluate_negative_control(report),
        )
    match = match_deltas(report.deltas, ground_truth.expected_deltas)
    tp, fn = len(match.true_positives), len(match.false_negatives)
    prf1 = precision_recall_f1(tp, len(match.false_positives), fn)
    noise_fps = sum(1 for d in match.false_positives if _is_below_match_threshold(d))
    prf1_excluding_noise = precision_recall_f1(tp, len(match.false_positives) - noise_fps, fn)
    return PairDeltaEval(
        pair_id=ground_truth.pair_id,
        negative_control=False,
        prf1=prf1,
        prf1_excluding_noise=prf1_excluding_noise,
        noise_false_positives=noise_fps,
        match=match,
    )


class AgreementRow(BaseModel):
    qa_id: str
    human_correctness: int
    human_groundedness: int
    judge_correctness: int
    judge_groundedness: int


class JudgeAgreement(BaseModel):
    n: int
    exact_agreement_correctness: float
    exact_agreement_groundedness: float
    mean_abs_diff_correctness: float
    mean_abs_diff_groundedness: float


def compute_judge_agreement(rows: list[AgreementRow]) -> JudgeAgreement | None:
    """Validates the LLM judge against hand-assigned scores on the held-out
    subset (BRIEF's explicit "validate the judge" bar) -- reports simple
    exact-agreement rate and mean absolute difference rather than a
    correlation coefficient like Cohen's kappa, which needs more samples
    than a small held-out set provides to mean anything reliable.
    """
    if not rows:
        return None
    n = len(rows)
    exact_c = sum(1 for r in rows if r.human_correctness == r.judge_correctness) / n
    exact_g = sum(1 for r in rows if r.human_groundedness == r.judge_groundedness) / n
    mad_c = sum(abs(r.human_correctness - r.judge_correctness) for r in rows) / n
    mad_g = sum(abs(r.human_groundedness - r.judge_groundedness) for r in rows) / n
    return JudgeAgreement(
        n=n,
        exact_agreement_correctness=round(exact_c, 4),
        exact_agreement_groundedness=round(exact_g, 4),
        mean_abs_diff_correctness=round(mad_c, 4),
        mean_abs_diff_groundedness=round(mad_g, 4),
    )
