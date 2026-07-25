"""Tests for eval/run_eval.py's pure/deterministic glue: the --delta-only
empty-summary contract and _display_path's out-of-repo fallback (a real bug
caught live: eval.run_eval crashed with ValueError when --out pointed
outside the repo, since Path.relative_to() only accepts a genuine subpath).
"""

from datetime import UTC, datetime
from pathlib import Path

from eval.metrics import PRF1, PairDeltaEval
from eval.run_eval import (
    _EMPTY_CHAT_SUMMARY,
    _EMPTY_RETRIEVAL_RESULT,
    REPO_ROOT,
    DeltaEvalSummary,
    Scorecard,
    _display_path,
    print_scorecard,
)


class TestDisplayPath:
    def test_in_repo_path_is_relative(self):
        path = REPO_ROOT / "eval" / "scorecard.json"
        result = _display_path(path)
        assert result in ("eval\\scorecard.json", "eval/scorecard.json")

    def test_out_of_repo_path_falls_back_to_absolute(self, tmp_path):
        # tmp_path is pytest's own tmp dir, guaranteed outside the repo.
        outside = tmp_path / "scorecard.json"
        result = _display_path(outside)
        assert result == str(outside.resolve())

    def test_relative_input_is_resolved_first(self, monkeypatch):
        # A relative --out (e.g. the Makefile's "eval/scorecard.json") must
        # resolve against cwd before the repo-relative check, not raise on
        # the relative-vs-absolute mismatch Path.relative_to() would hit.
        monkeypatch.chdir(REPO_ROOT)
        result = _display_path(Path("eval/scorecard.json"))
        assert "eval" in result and "scorecard.json" in result


class TestEmptySummaries:
    def test_empty_chat_summary_has_no_items_and_none_averages(self):
        assert _EMPTY_CHAT_SUMMARY.items == []
        assert _EMPTY_CHAT_SUMMARY.avg_correctness is None
        assert _EMPTY_CHAT_SUMMARY.avg_groundedness is None
        assert _EMPTY_CHAT_SUMMARY.judge_agreement is None

    def test_empty_retrieval_result_has_no_queries(self):
        assert _EMPTY_RETRIEVAL_RESULT.queries == []
        assert _EMPTY_RETRIEVAL_RESULT.recall_at_k is None


def _scorecard(chat=_EMPTY_CHAT_SUMMARY, retrieval=_EMPTY_RETRIEVAL_RESULT) -> Scorecard:
    prf1 = PRF1(precision=1.0, recall=1.0, f1=1.0, tp=1, fp=0, fn=0)
    return Scorecard(
        generated_at=datetime.now(UTC),
        delta=DeltaEvalSummary(
            per_pair=[PairDeltaEval(pair_id="pair1", negative_control=False, prf1=prf1)],
            aggregate=prf1,
        ),
        chat=chat,
        retrieval=retrieval,
        cost_latency_report_path="eval/cost_latency_report.md",
    )


class TestPrintScorecardDeltaOnly:
    def test_prints_skipped_message_when_chat_and_retrieval_are_empty(self, capsys):
        print_scorecard(_scorecard())
        out = capsys.readouterr().out
        assert "skipped (--delta-only" in out
        assert "Chat groundedness (LLM judge)" not in out

    def test_still_prints_delta_and_cost_sections(self, capsys):
        print_scorecard(_scorecard())
        out = capsys.readouterr().out
        assert "Delta P/R/F1" in out
        assert "Cost/latency analysis" in out
