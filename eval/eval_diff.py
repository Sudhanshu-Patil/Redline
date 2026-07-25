"""Compares two eval/scorecard.json runs (plan §10 / `make eval-diff`) so a
regression is visibly flagged, not just eyeballed. Exits non-zero if any
tracked metric regressed, so this is CI-usable as a gate, not just a
human-readable report.
"""

import sys
from pathlib import Path

from pydantic import BaseModel

from eval.run_eval import Scorecard


def load_scorecard(path: Path) -> Scorecard:
    return Scorecard.model_validate_json(path.read_text(encoding="utf-8"))


class MetricDiff(BaseModel):
    label: str
    old: float | None
    new: float | None
    delta: float | None
    regressed: bool


def _diff_metric(
    label: str, old: float | None, new: float | None, higher_is_better: bool = True
) -> MetricDiff:
    if old is None or new is None:
        return MetricDiff(label=label, old=old, new=new, delta=None, regressed=False)
    delta = round(new - old, 4)
    regressed = delta < 0 if higher_is_better else delta > 0
    return MetricDiff(label=label, old=old, new=new, delta=delta, regressed=regressed)


def diff_scorecards(old: Scorecard, new: Scorecard) -> list[MetricDiff]:
    old_agg, new_agg = old.delta.aggregate, new.delta.aggregate
    # (label, old value, new value, higher_is_better)
    rows: list[tuple[str, float | None, float | None, bool]] = [
        (
            "delta F1 (aggregate)",
            old_agg.f1 if old_agg else None,
            new_agg.f1 if new_agg else None,
            True,
        ),
        (
            "delta precision (aggregate)",
            old_agg.precision if old_agg else None,
            new_agg.precision if new_agg else None,
            True,
        ),
        (
            "delta recall (aggregate)",
            old_agg.recall if old_agg else None,
            new_agg.recall if new_agg else None,
            True,
        ),
        ("chat avg correctness", old.chat.avg_correctness, new.chat.avg_correctness, True),
        ("chat avg groundedness", old.chat.avg_groundedness, new.chat.avg_groundedness, True),
        ("chat refusal accuracy", old.chat.refusal_accuracy, new.chat.refusal_accuracy, True),
        ("chat over-refusal rate", old.chat.over_refusal_rate, new.chat.over_refusal_rate, False),
        ("retrieval recall@k", old.retrieval.recall_at_k, new.retrieval.recall_at_k, True),
        (
            "retrieval MRR",
            old.retrieval.mean_reciprocal_rank,
            new.retrieval.mean_reciprocal_rank,
            True,
        ),
    ]
    return [_diff_metric(label, o, n, higher) for label, o, n, higher in rows]


def print_diff(diffs: list[MetricDiff]) -> bool:
    """Prints each metric's before/after and returns True if any regressed."""
    any_regressed = False
    for d in diffs:
        if d.old is None or d.new is None or d.delta is None:
            print(f"  {d.label}: N/A (missing in one run)")
            continue
        if d.regressed:
            trend = "REGRESSED"
        elif d.delta > 0:
            trend = "improved"
        else:
            trend = "unchanged"
        flag = "  <<<" if d.regressed else ""
        print(f"  {d.label}: {d.old:.4f} -> {d.new:.4f} ({d.delta:+.4f}, {trend}){flag}")
        any_regressed = any_regressed or d.regressed
    return any_regressed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two eval scorecards; flag regressions.")
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args()

    old_scorecard = load_scorecard(args.old)
    new_scorecard = load_scorecard(args.new)
    print(f"Comparing {args.old} -> {args.new}\n")
    metric_diffs = diff_scorecards(old_scorecard, new_scorecard)
    has_regression = print_diff(metric_diffs)

    if has_regression:
        print("\nREGRESSION DETECTED.")
        sys.exit(1)
    print("\nNo regressions detected.")
