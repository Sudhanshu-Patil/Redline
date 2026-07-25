"""Tests for eval/metrics.py: delta P/R/F1 matching, negative-control
detection, and judge/human agreement -- all pure/deterministic, hand-built
fixtures throughout."""

import pytest

from eval.metrics import (
    AgreementRow,
    _is_below_match_threshold,
    _texts_match,
    compute_judge_agreement,
    evaluate_negative_control,
    evaluate_pair_delta,
    match_deltas,
    precision_recall_f1,
)
from eval.schema import ExpectedDelta, GroundTruth, GTLocator
from src.delta.engine import Delta, DeltaReport, DeltaStats


def expected(
    gt_id, change_type, element_type, old_text=None, new_text=None, bbox=None, near_text=None
):
    return ExpectedDelta(
        gt_id=gt_id, change_type=change_type, element_type=element_type,
        old_text=old_text, new_text=new_text,
        locator=GTLocator(page=0, bbox=bbox, near_text=near_text),
        rationale="test",
    )  # fmt: skip


def actual(change_type, element_type, old_text=None, new_text=None, old_bbox=None, new_bbox=None):
    return Delta(
        change_type=change_type, element_type=element_type, old_text=old_text, new_text=new_text,
        old_bbox=old_bbox, new_bbox=new_bbox, confidence=1.0,
    )  # fmt: skip


def bbox(x0, y0, x1, y1, page=0, w=1000.0, h=1000.0):
    from src.canonical.model import BBox

    return BBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1, page_width=w, page_height=h)


class TestTextsMatch:
    def test_exact_match(self):
        assert _texts_match("SP = 257 bar (g)", "SP = 257 bar (g)") is True

    def test_both_none_matches(self):
        assert _texts_match(None, None) is True

    def test_one_none_does_not_match(self):
        assert _texts_match(None, "text") is False
        assert _texts_match("text", None) is False

    def test_fuzzy_match_above_threshold(self):
        # OCR-style near-miss: single char difference on a long string
        a = "PRIMARY SEAL GAS IS TAKEN DOWNSTREAM"
        b = "PRIMARY SEAL GA5 IS TAKEN DOWNSTREAM"
        assert _texts_match(a, b) is True

    def test_dissimilar_text_does_not_match(self):
        assert _texts_match("SP = 257 bar (g)", "HH: 245") is False

    def test_both_empty_matches(self):
        assert _texts_match("", "") is True


class TestIsBelowMatchThreshold:
    def test_single_char_is_below(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "alignment_min_embed_text_len", 2)
        d = actual("added", "text_block", new_text="U")
        assert _is_below_match_threshold(d) is True

    def test_normal_text_is_not_below(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "alignment_min_embed_text_len", 2)
        d = actual("added", "text_block", new_text="NOTE 29")
        assert _is_below_match_threshold(d) is False

    def test_geometry_with_no_text_is_not_flagged_as_noise(self):
        d = actual("removed", "geometry", old_text="")
        assert _is_below_match_threshold(d) is False


class TestMatchDeltas:
    def test_exact_text_match_is_true_positive(self):
        exp = [expected("GT1", "modified", "setpoint", "SP = 257 bar (g)", "SP = 260 bar (g)")]
        act = [actual("modified", "setpoint", "SP = 257 bar (g)", "SP = 260 bar (g)")]
        result = match_deltas(act, exp)
        assert len(result.true_positives) == 1
        assert result.true_positives[0].gt_id == "GT1"
        assert result.false_negatives == []
        assert result.false_positives == []

    def test_missing_expected_is_false_negative(self):
        exp = [expected("GT1", "removed", "note", old_text="30. HEAT TRACING")]
        act: list[Delta] = []
        result = match_deltas(act, exp)
        assert len(result.false_negatives) == 1
        assert result.false_negatives[0].gt_id == "GT1"

    def test_unexpected_actual_is_false_positive(self):
        exp: list[ExpectedDelta] = []
        act = [actual("added", "text_block", new_text="U")]
        result = match_deltas(act, exp)
        assert len(result.false_positives) == 1

    def test_type_mismatch_does_not_match(self):
        exp = [expected("GT1", "added", "geometry", new_text="43BL9020")]
        act = [actual("added", "valve", new_text="43BL9020")]
        result = match_deltas(act, exp)
        assert result.true_positives == []
        assert len(result.false_negatives) == 1
        assert len(result.false_positives) == 1

    def test_duplicate_text_disambiguated_by_locator_proximity(self):
        """Pair 1's real case: two identical SP edits (PSV-9066A/B) must
        each claim their own nearest actual delta, not double-claim one."""
        exp = [
            expected(
                "GT-A", "modified", "setpoint", "SP = 257 bar (g)", "SP = 260 bar (g)",
                bbox=(840.0, 60.0, 876.0, 66.0),
            ),
            expected(
                "GT-B", "modified", "setpoint", "SP = 257 bar (g)", "SP = 260 bar (g)",
                bbox=(1038.0, 69.0, 1074.0, 75.0),
            ),
        ]  # fmt: skip
        act = [
            actual(
                "modified", "setpoint", "SP = 257 bar (g)", "SP = 260 bar (g)",
                old_bbox=bbox(840.0, 60.0, 876.0, 66.0), new_bbox=bbox(840.0, 60.0, 876.0, 66.0),
            ),
            actual(
                "modified", "setpoint", "SP = 257 bar (g)", "SP = 260 bar (g)",
                old_bbox=bbox(1038.0, 69.0, 1074.0, 75.0),
                new_bbox=bbox(1038.0, 69.0, 1074.0, 75.0),
            ),
        ]  # fmt: skip
        result = match_deltas(act, exp)
        assert len(result.true_positives) == 2  # both claimed, none stolen
        assert result.false_negatives == []
        assert result.false_positives == []

    def test_greedy_assignment_never_reuses_an_actual_delta(self):
        exp = [
            expected("GT-A", "added", "note", new_text="same text"),
            expected("GT-B", "added", "note", new_text="same text"),
        ]
        act = [actual("added", "note", new_text="same text")]  # only one real delta
        result = match_deltas(act, exp)
        assert len(result.true_positives) == 1
        assert len(result.false_negatives) == 1


class TestPrecisionRecallF1:
    def test_perfect_score(self):
        r = precision_recall_f1(tp=5, fp=0, fn=0)
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0

    def test_no_predictions_and_nothing_expected_is_vacuous_perfect(self):
        r = precision_recall_f1(tp=0, fp=0, fn=0)
        assert r.precision == 1.0
        assert r.recall == 1.0

    def test_all_false_positives(self):
        r = precision_recall_f1(tp=0, fp=10, fn=0)
        assert r.precision == 0.0
        assert r.recall == 1.0  # nothing expected, nothing missed

    def test_all_false_negatives(self):
        r = precision_recall_f1(tp=0, fp=0, fn=10)
        assert r.recall == 0.0
        assert r.precision == 1.0  # nothing predicted, nothing wrong

    def test_known_values(self):
        r = precision_recall_f1(tp=5, fp=3, fn=1)
        assert r.precision == pytest.approx(0.625)
        assert r.recall == pytest.approx(0.8333, abs=1e-3)


class TestNegativeControl:
    def test_warning_present_passes(self):
        report = DeltaReport(
            pid_a="A", pid_b="B", deltas=[],
            stats=DeltaStats(
                total_a=0, total_b=0, matched=0, unchanged=0, added=0, removed=0, modified=0,
                matched_by_tier={}, alignment_rate=0.0, exact_key_rate=0.0,
            ),
            warnings=["Low exact-key alignment"],
        )  # fmt: skip
        result = evaluate_negative_control(report)
        assert result.warning_raised is True
        assert result.passed is True

    def test_no_warning_fails(self):
        report = DeltaReport(
            pid_a="A", pid_b="B", deltas=[],
            stats=DeltaStats(
                total_a=0, total_b=0, matched=0, unchanged=0, added=0, removed=0, modified=0,
                matched_by_tier={}, alignment_rate=0.0, exact_key_rate=0.0,
            ),
        )  # fmt: skip
        result = evaluate_negative_control(report)
        assert result.warning_raised is False
        assert result.passed is False


class TestEvaluatePairDelta:
    def test_negative_control_pair_skips_prf1(self):
        gt = GroundTruth(pair_id="pair4", negative_control=True, expected_deltas=[])
        report = DeltaReport(
            pid_a="A", pid_b="B", deltas=[],
            stats=DeltaStats(
                total_a=0, total_b=0, matched=0, unchanged=0, added=0, removed=0, modified=0,
                matched_by_tier={}, alignment_rate=0.0, exact_key_rate=0.0,
            ),
            warnings=["Low exact-key alignment"],
        )  # fmt: skip
        result = evaluate_pair_delta(report, gt)
        assert result.negative_control is True
        assert result.prf1 is None
        assert result.negative_control_result is not None
        assert result.negative_control_result.passed is True

    def test_noise_excluded_precision_is_higher_when_noise_present(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "alignment_min_embed_text_len", 2)
        gt = GroundTruth(
            pair_id="p", negative_control=False,
            expected_deltas=[expected("GT1", "added", "note", new_text="real edit")],
        )  # fmt: skip
        deltas = [
            Delta(change_type="added", element_type="note", new_text="real edit", confidence=1.0),
            Delta(change_type="added", element_type="text_block", new_text="U", confidence=1.0),
            Delta(change_type="removed", element_type="text_block", old_text="C", confidence=1.0),
        ]
        report = DeltaReport(
            pid_a="A", pid_b="B", deltas=deltas,
            stats=DeltaStats(
                total_a=3, total_b=3, matched=0, unchanged=0, added=2, removed=1, modified=0,
                matched_by_tier={}, alignment_rate=0.0, exact_key_rate=0.0,
            ),
        )  # fmt: skip
        result = evaluate_pair_delta(report, gt)
        assert result.prf1 is not None
        assert result.prf1_excluding_noise is not None
        assert result.noise_false_positives == 2
        assert result.prf1_excluding_noise.precision > result.prf1.precision
        assert result.prf1_excluding_noise.fp == 0


class TestJudgeAgreement:
    def test_perfect_agreement(self):
        rows = [
            AgreementRow(
                qa_id="Q1", human_correctness=5, human_groundedness=5,
                judge_correctness=5, judge_groundedness=5,
            ),
            AgreementRow(
                qa_id="Q2", human_correctness=3, human_groundedness=4,
                judge_correctness=3, judge_groundedness=4,
            ),
        ]  # fmt: skip
        agreement = compute_judge_agreement(rows)
        assert agreement is not None
        assert agreement.exact_agreement_correctness == 1.0
        assert agreement.exact_agreement_groundedness == 1.0
        assert agreement.mean_abs_diff_correctness == 0.0

    def test_partial_disagreement(self):
        rows = [
            AgreementRow(
                qa_id="Q1", human_correctness=5, human_groundedness=5,
                judge_correctness=3, judge_groundedness=5,
            ),
        ]  # fmt: skip
        agreement = compute_judge_agreement(rows)
        assert agreement is not None
        assert agreement.exact_agreement_correctness == 0.0
        assert agreement.mean_abs_diff_correctness == 2.0
        assert agreement.exact_agreement_groundedness == 1.0

    def test_empty_rows_returns_none(self):
        assert compute_judge_agreement([]) is None
