"""Tests for eval/cost_latency_report.py against a hand-built MetricsSnapshot
-- no real trace directory needed, so the report's structure/content logic
is tested independent of whatever traces happen to exist on disk."""

from datetime import UTC, datetime

from eval.cost_latency_report import generate_cost_latency_report
from src.observability.metrics import (
    DeltaTotals,
    LatencyStats,
    LLMModelTelemetry,
    LLMTelemetry,
    MetricsSnapshot,
)


def latency(count=10, mean=5.0, p50=4.0, p95=9.0, p99=9.5, mx=10.0) -> LatencyStats:
    return LatencyStats(count=count, mean_ms=mean, p50_ms=p50, p95_ms=p95, p99_ms=p99, max_ms=mx)


def model_telem(calls=1, in_tok=100, out_tok=10, cost=0.0) -> LLMModelTelemetry:
    return LLMModelTelemetry(calls=calls, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)


def empty_llm() -> LLMTelemetry:
    return LLMTelemetry(
        total_calls=0, total_input_tokens=0, total_output_tokens=0, total_cost_usd=0.0, by_model={}
    )


def empty_delta(runs=0) -> DeltaTotals:
    return DeltaTotals(
        runs=runs, total_added=0, total_removed=0, total_modified=0,
        avg_alignment_rate=None, avg_exact_key_rate=None,
    )  # fmt: skip


def snapshot(**overrides) -> MetricsSnapshot:
    defaults: dict = {
        "generated_at": datetime.now(UTC),
        "trace_files_scanned": 5,
        "total_spans": 20,
        "error_rate": 0.0,
        "latency_by_stage": {},
        "llm": empty_llm(),
        "delta": empty_delta(),
        "retrieval": None,
    }
    defaults.update(overrides)
    return MetricsSnapshot(**defaults)


class TestGenerateCostLatencyReport:
    def test_includes_header_and_scan_counts(self):
        text = generate_cost_latency_report(snapshot(trace_files_scanned=42, total_spans=200))
        assert "Cost / Latency Budget Analysis" in text
        assert "42 trace file" in text
        assert "200 span" in text

    def test_delta_cost_is_always_zero_and_explained(self):
        text = generate_cost_latency_report(snapshot(delta=empty_delta(runs=7)))
        assert "Cost per delta run:** $0.0000" in text
        assert "7 run(s) measured" in text
        assert "deterministic" in text

    def test_no_chat_turns_reported_when_absent(self):
        text = generate_cost_latency_report(snapshot())
        assert "no chat turns recorded yet" in text

    def test_chat_turn_cost_reported_when_present(self):
        chat_telem = model_telem(calls=3, in_tok=300, out_tok=30, cost=0.006)
        llm = LLMTelemetry(
            total_calls=3, total_input_tokens=300, total_output_tokens=30, total_cost_usd=0.006,
            by_model={}, by_purpose={"chat_answer": chat_telem},
        )  # fmt: skip
        text = generate_cost_latency_report(snapshot(llm=llm))
        assert "Cost per chat turn:" in text
        assert "3 turn(s)" in text

    def test_judge_cost_kept_separate_from_chat_cost(self):
        llm = LLMTelemetry(
            total_calls=2, total_input_tokens=200, total_output_tokens=20, total_cost_usd=0.004,
            by_model={},
            by_purpose={
                "chat_answer": model_telem(cost=0.002),
                "eval_judge": model_telem(cost=0.002),
            },
        )  # fmt: skip
        text = generate_cost_latency_report(snapshot(llm=llm))
        assert "Cost per eval-judge call:" in text
        assert "kept separate from chat-turn cost" in text

    def test_free_tier_zero_cost_note_shown_when_calls_exist_but_cost_is_zero(self):
        llm = LLMTelemetry(
            total_calls=1, total_input_tokens=100, total_output_tokens=10,
            total_cost_usd=0.0, by_model={},
        )  # fmt: skip
        text = generate_cost_latency_report(snapshot(llm=llm))
        assert "free-tier provider" in text

    def test_latency_tables_only_include_stages_actually_present(self):
        text = generate_cost_latency_report(
            snapshot(latency_by_stage={"pdf_native.ingest": latency()})
        )
        assert "PDF native: full ingest" in text
        assert "PDF scanned" not in text
        assert "DWG/DXF" not in text

    def test_latency_row_shows_real_numbers(self):
        stage = latency(count=99, mean=12.3, p95=45.6, mx=100.0)
        text = generate_cost_latency_report(snapshot(latency_by_stage={"delta.compute": stage}))
        assert "99" in text
        assert "12.3" in text
        assert "45.6" in text
        assert "100.0" in text

    def test_scaling_section_always_present(self):
        text = generate_cost_latency_report(snapshot())
        assert "10x / 100x" in text
        assert "spatial index" in text  # delta alignment scaling note, always included

    def test_scanned_pdf_bottleneck_note_appears_only_when_measured(self):
        text_without = generate_cost_latency_report(snapshot())
        assert "real bottleneck" not in text_without

        stage = latency(mean=800000.0)
        text_with = generate_cost_latency_report(
            snapshot(latency_by_stage={"pdf_scanned.ingest": stage})
        )
        assert "real bottleneck" in text_with

    def test_embedding_load_note_appears_only_when_measured(self):
        text_without = generate_cost_latency_report(snapshot())
        assert "one-time process cost" not in text_without

        stage = latency(mean=1200.0)
        text_with = generate_cost_latency_report(
            snapshot(latency_by_stage={"delta.load_embedding_model": stage})
        )
        assert "one-time process cost" in text_with
        assert "1,200ms measured" in text_with

    def test_output_has_no_non_ascii_mojibake_risk_characters(self):
        text = generate_cost_latency_report(snapshot())
        # scaling narrative must stay plain-ASCII-safe (Windows console
        # encoding bit us once already with a middle-dot character)
        text.encode("ascii")
