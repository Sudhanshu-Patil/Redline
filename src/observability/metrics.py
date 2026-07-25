"""Metrics aggregation: reads every trace file under settings.traces_dir and
computes a snapshot (latency percentiles per stage, LLM token/cost totals,
delta counts, error rate) -- pulled live at request time, never a separately
maintained store (plan §9). served via metrics_server.py's FastAPI app.
"""

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import settings

# Span names whose attributes carry LLM token/cost telemetry (src/chat/llm.py).
_LLM_SPAN_NAMES = {"llm.complete", "llm.read_image_text"}
_DELTA_COMPUTE_SPAN = "delta.compute"
# Retrieval QUALITY (recall@k, MRR) needs ground truth no live query has, so
# it's only ever known offline -- eval/retrieval_eval.py emits a span under
# this distinct name, separate from chat.retrieve.hybrid/.exact/.vector
# (src/chat/index.py), which carry production latency/candidate-count
# telemetry on every real call and have no ground truth to score against.
# An earlier version of this filter matched any "chat.retrieve*" span,
# which triple-counted "queries" (one hybrid call always nests one exact +
# one vector sub-span) -- caught while wiring up eval/retrieval_eval.py
# (Phase 10), the first real code that could ever populate recall_at_k/mrr.
_RETRIEVAL_EVAL_SPAN_NAME = "eval.retrieval_query"


class LatencyStats(BaseModel):
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class LLMModelTelemetry(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLMTelemetry(BaseModel):
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    by_model: dict[str, LLMModelTelemetry]
    # Every LLMClient.complete() caller shares the "llm.complete" span name
    # (chat answers, eval judge scoring, ...) -- purpose is an explicit tag
    # (src/chat/llm.py) so cost_latency_report.py can report "cost per chat
    # turn" without it being silently blended with judge-eval spend. Calls
    # from before this tag existed, or read_image_text (OCR vision fallback,
    # already unambiguous by span name), land under "untagged".
    by_purpose: dict[str, LLMModelTelemetry] = Field(default_factory=dict)


class DeltaTotals(BaseModel):
    runs: int
    total_added: int
    total_removed: int
    total_modified: int
    avg_alignment_rate: float | None
    avg_exact_key_rate: float | None


class RetrievalMetrics(BaseModel):
    queries: int
    recall_at_k: float | None
    mean_reciprocal_rank: float | None


class MetricsSnapshot(BaseModel):
    generated_at: datetime
    trace_files_scanned: int
    total_spans: int
    error_rate: float
    latency_by_stage: dict[str, LatencyStats]
    llm: LLMTelemetry
    delta: DeltaTotals
    retrieval: RetrievalMetrics | None  # None until Phase 8 emits retrieval spans


def _iter_spans(traces_dir: Path) -> list[dict]:
    spans: list[dict] = []
    if not traces_dir.exists():
        return spans
    for path in sorted(traces_dir.glob("*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                spans.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            # A trace file being written concurrently, or corrupted by a
            # killed process -- skip it rather than fail the whole snapshot.
            continue
    return spans


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    idx = max(0, min(98, round(pct) - 1))
    return quantiles[idx]


def _latency_by_stage(spans: list[dict]) -> dict[str, LatencyStats]:
    durations_by_name: dict[str, list[float]] = {}
    for s in spans:
        durations_by_name.setdefault(s["name"], []).append(s["duration_ms"])

    result = {}
    for name, durations in durations_by_name.items():
        durations.sort()
        result[name] = LatencyStats(
            count=len(durations),
            mean_ms=round(statistics.fmean(durations), 3),
            p50_ms=round(_percentile(durations, 50), 3),
            p95_ms=round(_percentile(durations, 95), 3),
            p99_ms=round(_percentile(durations, 99), 3),
            max_ms=round(max(durations), 3),
        )
    return result


def _accumulate(
    bucket: dict[str, LLMModelTelemetry], key: str, in_tok: int, out_tok: int, cost: float
) -> None:
    entry = bucket.setdefault(
        key, LLMModelTelemetry(calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0)
    )
    entry.calls += 1
    entry.input_tokens += in_tok
    entry.output_tokens += out_tok
    entry.cost_usd = round(entry.cost_usd + cost, 6)


def _llm_telemetry(spans: list[dict]) -> LLMTelemetry:
    by_model: dict[str, LLMModelTelemetry] = {}
    by_purpose: dict[str, LLMModelTelemetry] = {}
    total_calls = total_in = total_out = 0
    total_cost = 0.0
    for s in spans:
        if s["name"] not in _LLM_SPAN_NAMES:
            continue
        attrs = s.get("attributes", {})
        if "input_tokens" not in attrs:
            continue  # e.g. a call that errored before usage was recorded
        model = attrs.get("model", "unknown")
        purpose = attrs.get("purpose") or "untagged"
        in_tok = int(attrs.get("input_tokens", 0))
        out_tok = int(attrs.get("output_tokens", 0))
        cost = float(attrs.get("cost_usd", 0.0))

        _accumulate(by_model, model, in_tok, out_tok, cost)
        _accumulate(by_purpose, purpose, in_tok, out_tok, cost)

        total_calls += 1
        total_in += in_tok
        total_out += out_tok
        total_cost += cost

    return LLMTelemetry(
        total_calls=total_calls,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost_usd=round(total_cost, 6),
        by_model=by_model,
        by_purpose=by_purpose,
    )


def _delta_totals(spans: list[dict]) -> DeltaTotals:
    runs = added = removed = modified = 0
    alignment_rates: list[float] = []
    exact_key_rates: list[float] = []
    for s in spans:
        if s["name"] != _DELTA_COMPUTE_SPAN:
            continue
        attrs = s.get("attributes", {})
        if "added" not in attrs:
            continue  # older trace predating this attribute set
        runs += 1
        added += int(attrs.get("added", 0))
        removed += int(attrs.get("removed", 0))
        modified += int(attrs.get("modified", 0))
        if "alignment_rate" in attrs:
            alignment_rates.append(float(attrs["alignment_rate"]))
        if "exact_key_rate" in attrs:
            exact_key_rates.append(float(attrs["exact_key_rate"]))

    return DeltaTotals(
        runs=runs,
        total_added=added,
        total_removed=removed,
        total_modified=modified,
        avg_alignment_rate=round(statistics.fmean(alignment_rates), 4) if alignment_rates else None,
        avg_exact_key_rate=round(statistics.fmean(exact_key_rates), 4) if exact_key_rates else None,
    )


def _retrieval_metrics(spans: list[dict]) -> RetrievalMetrics | None:
    """Retrieval quality is only known once eval/retrieval_eval.py has run
    at least once against labeled data -- report None rather than fabricate
    a zero/empty result that would look like a real (if trivial) measurement.
    """
    retrieval_spans = [s for s in spans if s["name"] == _RETRIEVAL_EVAL_SPAN_NAME]
    if not retrieval_spans:
        return None
    recalls = [
        float(s["attributes"]["recall_at_k"])
        for s in retrieval_spans
        if "recall_at_k" in s.get("attributes", {})
    ]
    mrrs = [
        float(s["attributes"]["mrr"])
        for s in retrieval_spans
        if "mrr" in s.get("attributes", {})
    ]
    return RetrievalMetrics(
        queries=len(retrieval_spans),
        recall_at_k=round(statistics.fmean(recalls), 4) if recalls else None,
        mean_reciprocal_rank=round(statistics.fmean(mrrs), 4) if mrrs else None,
    )


def compute_metrics_snapshot(traces_dir: Path | None = None) -> MetricsSnapshot:
    traces_dir = traces_dir or settings.traces_dir
    trace_files = sorted(traces_dir.glob("*.jsonl")) if traces_dir.exists() else []
    spans = _iter_spans(traces_dir)

    error_count = sum(1 for s in spans if s.get("status") == "error")
    error_rate = round(error_count / len(spans), 4) if spans else 0.0

    return MetricsSnapshot(
        generated_at=datetime.now(UTC),
        trace_files_scanned=len(trace_files),
        total_spans=len(spans),
        error_rate=error_rate,
        latency_by_stage=_latency_by_stage(spans),
        llm=_llm_telemetry(spans),
        delta=_delta_totals(spans),
        retrieval=_retrieval_metrics(spans),
    )
