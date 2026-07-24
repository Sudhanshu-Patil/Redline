"""Unit tests for metrics aggregation, against hand-built trace files (no
dependency on real accumulated traces -- fast and deterministic)."""

import json

import pytest

from src.observability.metrics import compute_metrics_snapshot


def write_trace(tmp_path, trace_id: str, spans: list[dict]):
    path = tmp_path / f"{trace_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for span in spans:
            f.write(json.dumps(span) + "\n")


def span(
    name: str, duration_ms: float, status: str = "ok", trace_id: str = "t1", **attrs
) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": f"s-{name}-{duration_ms}",
        "parent_span_id": None,
        "name": name,
        "start_time": 1000.0,
        "end_time": 1000.0 + duration_ms / 1000,
        "duration_ms": duration_ms,
        "status": status,
        "attributes": attrs,
    }


class TestEmptyTracesDir:
    def test_nonexistent_dir_returns_empty_snapshot(self, tmp_path):
        snap = compute_metrics_snapshot(tmp_path / "does_not_exist")
        assert snap.trace_files_scanned == 0
        assert snap.total_spans == 0
        assert snap.error_rate == 0.0
        assert snap.latency_by_stage == {}
        assert snap.llm.total_calls == 0
        assert snap.delta.runs == 0
        assert snap.retrieval is None

    def test_empty_dir_returns_empty_snapshot(self, tmp_path):
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.trace_files_scanned == 0
        assert snap.total_spans == 0


class TestLatencyPercentiles:
    def test_known_distribution(self, tmp_path):
        # 1..100 ms, so p50=50-ish, p95=95-ish, p99=99-ish, max=100
        durations = list(range(1, 101))
        write_trace(tmp_path, "t1", [span("stage.x", float(d)) for d in durations])
        snap = compute_metrics_snapshot(tmp_path)
        stats = snap.latency_by_stage["stage.x"]
        assert stats.count == 100
        assert stats.max_ms == 100.0
        assert 48 <= stats.p50_ms <= 52
        assert 93 <= stats.p95_ms <= 97
        assert 97 <= stats.p99_ms <= 100
        assert stats.mean_ms == pytest.approx(50.5, abs=0.1)

    def test_single_sample(self, tmp_path):
        write_trace(tmp_path, "t1", [span("stage.once", 42.0)])
        snap = compute_metrics_snapshot(tmp_path)
        stats = snap.latency_by_stage["stage.once"]
        assert stats.count == 1
        assert stats.p50_ms == stats.p95_ms == stats.p99_ms == stats.max_ms == 42.0

    def test_stages_kept_separate(self, tmp_path):
        write_trace(
            tmp_path, "t1",
            [span("stage.a", 10.0), span("stage.b", 20.0), span("stage.a", 30.0)],
        )  # fmt: skip
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.latency_by_stage["stage.a"].count == 2
        assert snap.latency_by_stage["stage.b"].count == 1

    def test_aggregates_across_multiple_trace_files(self, tmp_path):
        write_trace(tmp_path, "t1", [span("stage.x", 10.0)])
        write_trace(tmp_path, "t2", [span("stage.x", 20.0)])
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.trace_files_scanned == 2
        assert snap.latency_by_stage["stage.x"].count == 2


class TestErrorRate:
    def test_computed_across_all_spans(self, tmp_path):
        write_trace(
            tmp_path, "t1",
            [
                span("a", 1.0, status="ok"),
                span("b", 1.0, status="error"),
                span("c", 1.0, status="ok"),
            ],
        )  # fmt: skip
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.total_spans == 3
        assert snap.error_rate == pytest.approx(1 / 3, abs=1e-4)


class TestLLMTelemetry:
    def test_aggregates_tokens_and_cost_by_model(self, tmp_path):
        write_trace(
            tmp_path, "t1",
            [
                span(
                    "llm.complete", 100.0, provider="anthropic", model="claude-sonnet-5",
                    input_tokens=1000, output_tokens=50, cost_usd=0.01,
                ),
                span(
                    "llm.read_image_text", 200.0, provider="anthropic", model="claude-sonnet-5",
                    input_tokens=2000, output_tokens=20, cost_usd=0.02,
                ),
                span(
                    "llm.complete", 50.0, provider="openai_compatible", model="llama-3.3-70b",
                    input_tokens=500, output_tokens=10, cost_usd=0.0,
                ),
            ],
        )  # fmt: skip
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.llm.total_calls == 3
        assert snap.llm.total_input_tokens == 3500
        assert snap.llm.total_output_tokens == 80
        assert snap.llm.total_cost_usd == pytest.approx(0.03)
        assert snap.llm.by_model["claude-sonnet-5"].calls == 2
        assert snap.llm.by_model["claude-sonnet-5"].input_tokens == 3000
        assert snap.llm.by_model["llama-3.3-70b"].calls == 1

    def test_errored_llm_call_without_usage_is_skipped_not_crashed(self, tmp_path):
        write_trace(
            tmp_path, "t1",
            [span("llm.complete", 5.0, status="error", error="Rate limit exceeded")],
        )  # fmt: skip
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.llm.total_calls == 0  # no usage recorded -> not counted

    def test_non_llm_spans_ignored(self, tmp_path):
        write_trace(tmp_path, "t1", [span("pdf_native.ingest", 5.0, elements_extracted=800)])
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.llm.total_calls == 0


class TestDeltaTotals:
    def test_aggregates_across_runs(self, tmp_path):
        write_trace(
            tmp_path, "t1",
            [
                span(
                    "delta.compute", 5.0, added=3, removed=1, modified=2,
                    alignment_rate=0.9, exact_key_rate=0.95,
                ),
                span(
                    "delta.compute", 5.0, added=1, removed=1, modified=0,
                    alignment_rate=0.5, exact_key_rate=0.3,
                ),
            ],
        )  # fmt: skip
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.delta.runs == 2
        assert snap.delta.total_added == 4
        assert snap.delta.total_removed == 2
        assert snap.delta.total_modified == 2
        assert snap.delta.avg_alignment_rate == pytest.approx(0.7)
        assert snap.delta.avg_exact_key_rate == pytest.approx(0.625)

    def test_older_trace_without_added_attribute_is_skipped(self, tmp_path):
        """Traces written before this phase's span enrichment (only
        'deltas' + 'alignment_rate') must not crash the aggregator."""
        write_trace(tmp_path, "t1", [span("delta.compute", 5.0, deltas=4, alignment_rate=0.8)])
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.delta.runs == 0

    def test_no_delta_runs_yields_none_averages(self, tmp_path):
        write_trace(tmp_path, "t1", [span("pdf_native.ingest", 5.0)])
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.delta.runs == 0
        assert snap.delta.avg_alignment_rate is None
        assert snap.delta.avg_exact_key_rate is None


class TestRetrievalMetrics:
    def test_none_when_no_retrieval_spans_exist(self, tmp_path):
        write_trace(tmp_path, "t1", [span("pdf_native.ingest", 5.0)])
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.retrieval is None

    def test_populated_when_retrieval_spans_present(self, tmp_path):
        write_trace(
            tmp_path, "t1",
            [
                span("chat.retrieve.hybrid", 10.0, recall_at_k=0.8, mrr=0.6),
                span("chat.retrieve.hybrid", 10.0, recall_at_k=1.0, mrr=1.0),
            ],
        )  # fmt: skip
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.retrieval is not None
        assert snap.retrieval.queries == 2
        assert snap.retrieval.recall_at_k == pytest.approx(0.9)
        assert snap.retrieval.mean_reciprocal_rank == pytest.approx(0.8)


class TestMalformedTraceFile:
    def test_corrupted_line_skipped_not_crashed(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"name": "ok.span", "duration_ms": 1.0, "status": "ok", "attributes": {}}\n'
            "not valid json at all\n"
            '{"name": "ok.span", "duration_ms": 2.0, "status": "ok", "attributes": {}}\n',
            encoding="utf-8",
        )
        snap = compute_metrics_snapshot(tmp_path)
        # the whole file is skipped on a parse error (simplest safe behavior)
        assert snap.trace_files_scanned == 1

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "spaced.jsonl"
        path.write_text(
            '{"name": "a", "duration_ms": 1.0, "status": "ok", "attributes": {}}\n'
            "\n\n"
            '{"name": "a", "duration_ms": 2.0, "status": "ok", "attributes": {}}\n',
            encoding="utf-8",
        )
        snap = compute_metrics_snapshot(tmp_path)
        assert snap.total_spans == 2
