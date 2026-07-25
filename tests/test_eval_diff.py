"""Tests for eval/eval_diff.py: pure comparison logic against hand-built
Scorecards -- no real eval run needed."""

from datetime import UTC, datetime

from eval.eval_diff import _diff_metric, diff_scorecards, print_diff
from eval.metrics import PRF1, PairDeltaEval
from eval.retrieval_eval import RetrievalEvalResult
from eval.run_eval import ChatEvalSummary, DeltaEvalSummary, Scorecard


def prf1(p, r, f) -> PRF1:
    return PRF1(precision=p, recall=r, f1=f, tp=0, fp=0, fn=0)


def scorecard(
    f1=0.8, correctness=4.0, groundedness=4.0, refusal_acc=1.0, over_refusal=0.0,
    recall=0.8, mrr=0.7,
) -> Scorecard:  # fmt: skip
    pair_eval = PairDeltaEval(pair_id="pair1", negative_control=False, prf1=prf1(0.8, 0.8, f1))
    return Scorecard(
        generated_at=datetime.now(UTC),
        delta=DeltaEvalSummary(per_pair=[pair_eval], aggregate=prf1(0.8, 0.8, f1)),
        chat=ChatEvalSummary(
            items=[], avg_correctness=correctness, avg_groundedness=groundedness,
            refusal_accuracy=refusal_acc, over_refusal_rate=over_refusal, judge_agreement=None,
        ),
        retrieval=RetrievalEvalResult(queries=[], recall_at_k=recall, mean_reciprocal_rank=mrr),
        cost_latency_report_path="eval/cost_latency_report.md",
    )  # fmt: skip


class TestDiffMetric:
    def test_improvement_for_higher_is_better(self):
        d = _diff_metric("m", 0.5, 0.8, higher_is_better=True)
        assert d.delta == 0.3
        assert d.regressed is False

    def test_regression_for_higher_is_better(self):
        d = _diff_metric("m", 0.8, 0.5, higher_is_better=True)
        assert d.delta == -0.3
        assert d.regressed is True

    def test_regression_for_lower_is_better(self):
        d = _diff_metric("m", 0.1, 0.3, higher_is_better=False)
        assert d.regressed is True

    def test_improvement_for_lower_is_better(self):
        d = _diff_metric("m", 0.3, 0.1, higher_is_better=False)
        assert d.regressed is False

    def test_missing_value_is_not_a_regression(self):
        d = _diff_metric("m", None, 0.5)
        assert d.regressed is False
        assert d.delta is None

    def test_no_change_is_not_a_regression(self):
        d = _diff_metric("m", 0.5, 0.5)
        assert d.regressed is False
        assert d.delta == 0.0


class TestDiffScorecards:
    def test_identical_scorecards_no_regression(self):
        old = scorecard()
        new = scorecard()
        diffs = diff_scorecards(old, new)
        assert print_diff(diffs) is False

    def test_f1_regression_detected(self):
        old = scorecard(f1=0.9)
        new = scorecard(f1=0.5)
        diffs = diff_scorecards(old, new)
        assert print_diff(diffs) is True

    def test_over_refusal_increase_is_a_regression(self):
        old = scorecard(over_refusal=0.0)
        new = scorecard(over_refusal=0.2)
        diffs = diff_scorecards(old, new)
        assert print_diff(diffs) is True

    def test_over_refusal_decrease_is_not_a_regression(self):
        old = scorecard(over_refusal=0.2)
        new = scorecard(over_refusal=0.0)
        diffs = diff_scorecards(old, new)
        assert print_diff(diffs) is False

    def test_missing_aggregate_handled_gracefully(self):
        old = scorecard()
        old.delta.aggregate = None
        new = scorecard()
        diffs = diff_scorecards(old, new)
        # must not crash; delta-F1 row present but non-comparable
        f1_row = next(d for d in diffs if d.label == "delta F1 (aggregate)")
        assert f1_row.regressed is False
