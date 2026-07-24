import json

from src.observability import tracing


def test_span_writes_one_json_line_per_span(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing.settings, "traces_dir", tmp_path)

    with tracing.trace("ingest_run") as trace_id:
        with tracing.span("stage_one", pid="doc1") as sp:
            sp["elements"] = 5
        with tracing.span("stage_two"):
            pass

    trace_file = tmp_path / f"{trace_id}.jsonl"
    assert trace_file.exists()
    lines = [json.loads(line) for line in trace_file.read_text().splitlines()]

    # top-level trace span + two child spans
    assert len(lines) == 3
    names = {rec["name"] for rec in lines}
    assert names == {"ingest_run", "stage_one", "stage_two"}

    stage_one = next(r for r in lines if r["name"] == "stage_one")
    assert stage_one["attributes"] == {"pid": "doc1", "elements": 5}
    assert stage_one["status"] == "ok"
    assert stage_one["trace_id"] == trace_id
    assert stage_one["parent_span_id"] is not None


def test_span_records_error_status_and_reraises(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing.settings, "traces_dir", tmp_path)

    trace_id = None
    with tracing.trace("failing_run") as tid:
        trace_id = tid
        try:
            with tracing.span("boom"):
                raise ValueError("bad input")
        except ValueError:
            pass

    trace_file = tmp_path / f"{trace_id}.jsonl"
    lines = [json.loads(line) for line in trace_file.read_text().splitlines()]
    boom = next(r for r in lines if r["name"] == "boom")
    assert boom["status"] == "error"
    assert boom["attributes"]["error"] == "bad input"


def test_span_without_active_trace_mints_standalone_trace_id(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing.settings, "traces_dir", tmp_path)

    with tracing.span("orphan"):
        pass

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
