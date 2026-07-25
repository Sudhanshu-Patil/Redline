"""Eval harness orchestrator (plan §10 / `make eval`): runs delta P/R/F1
across all labeled pairs, chat groundedness (LLM-judge, validated against
hand-scored held-out labels), retrieval-quality eval, and the cost/latency
analysis -- prints a scorecard and writes eval/scorecard.json.
"""

import statistics
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from eval.cost_latency_report import generate_cost_latency_report
from eval.judge import JudgeParseError, JudgeScore, judge_answer
from eval.metrics import (
    PRF1,
    AgreementRow,
    JudgeAgreement,
    PairDeltaEval,
    compute_judge_agreement,
    evaluate_pair_delta,
    precision_recall_f1,
)
from eval.retrieval_eval import RetrievalEvalResult, evaluate_retrieval
from eval.schema import (
    DocRef,
    QAItem,
    load_ground_truth,
    load_human_labels,
    load_manifest,
    load_qa_dataset,
)
from src.canonical.model import CanonicalDocument
from src.chat.answer import answer_question
from src.chat.index import ChatIndex
from src.delta.engine import compute_delta
from src.ingest.base import AdapterFormat, IngestAdapter
from src.ingest.dwg import DwgAdapter
from src.ingest.pdf_native import PdfNativeAdapter
from src.ingest.pdf_scanned import PdfScannedAdapter
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
_ADAPTERS: dict[AdapterFormat, type[IngestAdapter]] = {
    "pdf_native": PdfNativeAdapter,
    "pdf_scanned": PdfScannedAdapter,
    "dwg": DwgAdapter,
}
_DELTA_PAIR_IDS = ["pair1", "pair2", "pair3", "pair4"]
_CHAT_PAIR_MANIFEST = REPO_ROOT / "data" / "samples" / "pair1" / "manifest.json"
_CHAT_QA_PATH = REPO_ROOT / "eval" / "datasets" / "qa_pair1.json"
_HUMAN_LABELS_PATH = REPO_ROOT / "eval" / "datasets" / "human_labels_pair1.json"


def _ingest(doc_ref: DocRef) -> CanonicalDocument:
    return _ADAPTERS[doc_ref.format]().ingest(
        REPO_ROOT / doc_ref.path, pid=doc_ref.pid, revision_label=doc_ref.revision_label
    )


class DeltaEvalSummary(BaseModel):
    per_pair: list[PairDeltaEval]
    # Sum of raw tp/fp/fn across all non-negative-control pairs, not an
    # average of per-pair rates -- a pair with more real edits isn't
    # under-weighted relative to a pair with few.
    aggregate: PRF1 | None


def run_delta_eval() -> DeltaEvalSummary:
    with tracing.span("eval.delta_suite", pairs=len(_DELTA_PAIR_IDS)):
        per_pair = []
        for pair_id in _DELTA_PAIR_IDS:
            manifest = load_manifest(REPO_ROOT / "data" / "samples" / pair_id / "manifest.json")
            ground_truth = load_ground_truth(REPO_ROOT / manifest.ground_truth_path)
            doc_a = _ingest(manifest.doc_a)
            doc_b = _ingest(manifest.doc_b)
            report = compute_delta(doc_a, doc_b)
            per_pair.append(evaluate_pair_delta(report, ground_truth))
            log.info(
                "delta eval: pair scored",
                extra={"extra_fields": {"pair_id": pair_id}},
            )

        scored_prf1 = [p.prf1 for p in per_pair if not p.negative_control and p.prf1 is not None]
        aggregate = None
        if scored_prf1:
            tp = sum(p.tp for p in scored_prf1)
            fp = sum(p.fp for p in scored_prf1)
            fn = sum(p.fn for p in scored_prf1)
            aggregate = precision_recall_f1(tp, fp, fn)
        return DeltaEvalSummary(per_pair=per_pair, aggregate=aggregate)


class ChatEvalItem(BaseModel):
    qa_id: str
    question: str
    answerable: bool
    answer_text: str
    refused: bool
    correct_refusal: bool
    citations: list[str]
    invalid_citations: list[str]
    judge: JudgeScore | None = None


class ChatEvalSummary(BaseModel):
    items: list[ChatEvalItem]
    avg_correctness: float | None
    avg_groundedness: float | None
    refusal_accuracy: float  # fraction of unanswerable QAs correctly refused
    over_refusal_rate: float  # fraction of answerable QAs incorrectly refused
    judge_agreement: JudgeAgreement | None


def run_chat_eval(qa_items: list[QAItem]) -> tuple[ChatEvalSummary, RetrievalEvalResult]:
    with tracing.span("eval.chat_suite", items=len(qa_items)):
        manifest = load_manifest(_CHAT_PAIR_MANIFEST)
        doc_a = _ingest(manifest.doc_a)
        doc_b = _ingest(manifest.doc_b)
        index = ChatIndex(collection_name=f"eval_{manifest.pair_id}")
        index.index_document(doc_a)
        index.index_document(doc_b)

        retrieval_result = evaluate_retrieval(index, qa_items)

        human_by_id = {}
        if _HUMAN_LABELS_PATH.exists():
            human_labels = load_human_labels(_HUMAN_LABELS_PATH).labels
            human_by_id = {label.qa_id: label for label in human_labels}

        items: list[ChatEvalItem] = []
        agreement_rows: list[AgreementRow] = []
        for qa in qa_items:
            answer = answer_question(index, qa.question)
            correct_refusal = answer.refused == (not qa.answerable)
            try:
                judge = judge_answer(qa, answer)
            except JudgeParseError as exc:
                # A free-tier model occasionally not following the judge
                # format is a real, expected risk (unlike a chat/delta bug)
                # -- one bad reply must not lose the rest of a run that
                # includes a slow scanned-PDF ingest and dozens of other
                # LLM calls. Recorded as judge=None (excluded from the
                # averages/agreement report below) rather than faked.
                log.warning(
                    "judge reply unparseable; scoring this item as ungraded",
                    extra={"extra_fields": {"qa_id": qa.qa_id, "error": str(exc)}},
                )
                judge = None
            items.append(
                ChatEvalItem(
                    qa_id=qa.qa_id,
                    question=qa.question,
                    answerable=qa.answerable,
                    answer_text=answer.answer_text,
                    refused=answer.refused,
                    correct_refusal=correct_refusal,
                    citations=[c.element_id for c in answer.citations],
                    invalid_citations=answer.invalid_citation_ids,
                    judge=judge,
                )
            )
            if judge and qa.held_out and qa.qa_id in human_by_id:
                human = human_by_id[qa.qa_id]
                agreement_rows.append(
                    AgreementRow(
                        qa_id=qa.qa_id,
                        human_correctness=human.correctness,
                        human_groundedness=human.groundedness,
                        judge_correctness=judge.correctness,
                        judge_groundedness=judge.groundedness,
                    )
                )
            log.info("chat eval: item scored", extra={"extra_fields": {"qa_id": qa.qa_id}})

        judged = [i for i in items if i.judge is not None]
        avg_correctness = (
            statistics.fmean(i.judge.correctness for i in judged if i.judge) if judged else None
        )
        avg_groundedness = (
            statistics.fmean(i.judge.groundedness for i in judged if i.judge) if judged else None
        )
        unanswerable = [i for i in items if not i.answerable]
        answerable = [i for i in items if i.answerable]
        refusal_accuracy = (
            sum(1 for i in unanswerable if i.correct_refusal) / len(unanswerable)
            if unanswerable
            else 1.0
        )
        over_refusal_rate = (
            sum(1 for i in answerable if i.refused) / len(answerable) if answerable else 0.0
        )

        summary = ChatEvalSummary(
            items=items,
            avg_correctness=round(avg_correctness, 4) if avg_correctness is not None else None,
            avg_groundedness=round(avg_groundedness, 4) if avg_groundedness is not None else None,
            refusal_accuracy=round(refusal_accuracy, 4),
            over_refusal_rate=round(over_refusal_rate, 4),
            judge_agreement=compute_judge_agreement(agreement_rows),
        )
        return summary, retrieval_result


class Scorecard(BaseModel):
    generated_at: datetime
    delta: DeltaEvalSummary
    chat: ChatEvalSummary
    retrieval: RetrievalEvalResult
    cost_latency_report_path: str


def run_full_eval(out_path: Path) -> Scorecard:
    with tracing.span("eval.run_full"):
        delta_summary = run_delta_eval()
        qa_items = load_qa_dataset(_CHAT_QA_PATH).items
        chat_summary, retrieval_result = run_chat_eval(qa_items)

        cost_latency_path = out_path.parent / "cost_latency_report.md"
        cost_latency_path.parent.mkdir(parents=True, exist_ok=True)
        cost_latency_path.write_text(generate_cost_latency_report(), encoding="utf-8")

        # .resolve() first: out_path (and therefore cost_latency_path) may be
        # relative (e.g. the Makefile passes --out eval/scorecard.json), and
        # Path.relative_to() refuses to compare a relative path against
        # REPO_ROOT (absolute) even when one genuinely is inside the other.
        scorecard = Scorecard(
            generated_at=datetime.now(UTC),
            delta=delta_summary,
            chat=chat_summary,
            retrieval=retrieval_result,
            cost_latency_report_path=str(cost_latency_path.resolve().relative_to(REPO_ROOT)),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
        return scorecard


def print_scorecard(scorecard: Scorecard) -> None:
    print("=== Delta P/R/F1 ===")
    for p in scorecard.delta.per_pair:
        if p.negative_control:
            result = p.negative_control_result
            status = "PASS" if result and result.passed else "FAIL"
            warned = result.warning_raised if result else False
            print(f"  {p.pair_id}: negative control -- warning_raised={warned} ({status})")
        elif p.prf1 and p.prf1_excluding_noise:
            print(
                f"  {p.pair_id}: P={p.prf1.precision:.4f} R={p.prf1.recall:.4f} "
                f"F1={p.prf1.f1:.4f}  |  excl.-noise P={p.prf1_excluding_noise.precision:.4f} "
                f"R={p.prf1_excluding_noise.recall:.4f} F1={p.prf1_excluding_noise.f1:.4f} "
                f"(noise_fps={p.noise_false_positives})"
            )
    if scorecard.delta.aggregate:
        a = scorecard.delta.aggregate
        print(
            f"  AGGREGATE (raw, non-negative-control pairs): "
            f"P={a.precision:.4f} R={a.recall:.4f} F1={a.f1:.4f}"
        )

    print("\n=== Chat groundedness (LLM judge) ===")
    print(f"  avg correctness (1-5): {scorecard.chat.avg_correctness}")
    print(f"  avg groundedness (1-5): {scorecard.chat.avg_groundedness}")
    refusal_acc = scorecard.chat.refusal_accuracy
    over_refusal = scorecard.chat.over_refusal_rate
    print(f"  refusal accuracy (unanswerable correctly refused): {refusal_acc:.1%}")
    print(f"  over-refusal rate (answerable incorrectly refused): {over_refusal:.1%}")
    ja = scorecard.chat.judge_agreement
    if ja:
        print(
            f"  judge/human agreement (n={ja.n}): "
            f"exact correctness={ja.exact_agreement_correctness:.1%}, "
            f"exact groundedness={ja.exact_agreement_groundedness:.1%}, "
            f"MAD correctness={ja.mean_abs_diff_correctness}, "
            f"MAD groundedness={ja.mean_abs_diff_groundedness}"
        )
    else:
        print("  judge/human agreement: no held-out human labels found")

    print("\n=== Retrieval quality ===")
    print(f"  recall@k: {scorecard.retrieval.recall_at_k}")
    print(f"  MRR: {scorecard.retrieval.mean_reciprocal_rank}")

    print(f"\nCost/latency analysis: {scorecard.cost_latency_report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full eval harness and write a scorecard.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "scorecard.json")
    args = parser.parse_args()

    result_scorecard = run_full_eval(args.out)
    print_scorecard(result_scorecard)
    print(f"\nwrote {args.out}")
