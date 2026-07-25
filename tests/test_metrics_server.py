"""Integration tests for the served /metrics endpoints: real FastAPI app,
real TestClient, real on-disk trace directory (temp, not the dev traces/
dir, so results are deterministic) -- as opposed to test_metrics.py, which
unit-tests the aggregation function directly.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.observability.metrics_server import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "traces_dir", tmp_path)
    trace_path = tmp_path / "t1.jsonl"
    spans = [
        {
            "trace_id": "t1", "span_id": "s1", "parent_span_id": None,
            "name": "pdf_native.ingest", "start_time": 1000.0, "end_time": 1000.05,
            "duration_ms": 50.0, "status": "ok", "attributes": {"elements_extracted": 800},
        },
        {
            "trace_id": "t1", "span_id": "s2", "parent_span_id": "s1",
            "name": "llm.complete", "start_time": 1000.05, "end_time": 1000.15,
            "duration_ms": 100.0, "status": "ok",
            "attributes": {
                "provider": "anthropic", "model": "claude-sonnet-5",
                "input_tokens": 1000, "output_tokens": 50, "cost_usd": 0.01,
            },
        },
        {
            "trace_id": "t1", "span_id": "s3", "parent_span_id": None,
            "name": "delta.compute", "start_time": 1001.0, "end_time": 1001.02,
            "duration_ms": 20.0, "status": "ok",
            "attributes": {
                "added": 3, "removed": 1, "modified": 2, "unchanged": 10,
                "alignment_rate": 0.9, "exact_key_rate": 0.95,
            },
        },
    ]  # fmt: skip
    with trace_path.open("w", encoding="utf-8") as f:
        for s in spans:
            f.write(json.dumps(s) + "\n")
    return TestClient(app)


class TestMetricsJsonEndpoint:
    def test_returns_200_and_valid_shape(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace_files_scanned"] == 1
        assert body["total_spans"] == 3
        assert body["error_rate"] == 0.0

    def test_reflects_llm_and_delta_data(self, client):
        body = client.get("/metrics").json()
        assert body["llm"]["total_calls"] == 1
        assert body["llm"]["by_model"]["claude-sonnet-5"]["input_tokens"] == 1000
        assert body["delta"]["runs"] == 1
        assert body["delta"]["total_added"] == 3

    def test_content_type_is_json(self, client):
        resp = client.get("/metrics")
        assert resp.headers["content-type"].startswith("application/json")


class TestDashboardHtmlEndpoint:
    def test_returns_200_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_contains_key_figures(self, client):
        html = client.get("/").text
        assert "pdf_native.ingest" in html
        assert "claude-sonnet-5" in html
        assert "<html" in html.lower()

    def test_dark_mode_media_query_present(self, client):
        html = client.get("/").text
        assert "prefers-color-scheme: dark" in html

    def test_body_is_in_scope_for_the_color_tokens(self, client):
        # Regression guard for a real contrast bug: --ink-primary/--page etc.
        # were defined only on .metrics-root, a *descendant* of <body> --
        # custom properties aren't visible to an ancestor of where they're
        # declared, so `body { color: var(--ink-primary) }` silently failed
        # to resolve. In dark mode this produced black text on a near-black
        # tile: the tile itself (a real .metrics-root descendant) picked up
        # the correct dark --surface, but text inheriting through the broken
        # body rule fell back to the browser's default black instead of the
        # dark-mode --ink-primary override.
        html = client.get("/").text
        selector = html.split("--ink-primary:")[0].rsplit("}", 1)[-1].split("{")[0]
        assert "body" in selector

    def test_no_unescaped_script_from_span_names(self, tmp_path, monkeypatch):
        """Span/model names ultimately come from code we control, but the
        renderer still must not blindly interpolate raw strings into HTML --
        guards against a future span name containing markup."""
        monkeypatch.setattr(settings, "traces_dir", tmp_path)
        trace_path = tmp_path / "t2.jsonl"
        span = {
            "trace_id": "t2", "span_id": "s1", "parent_span_id": None,
            "name": "<script>alert(1)</script>", "start_time": 1.0, "end_time": 1.01,
            "duration_ms": 10.0, "status": "ok", "attributes": {},
        }  # fmt: skip
        trace_path.write_text(json.dumps(span) + "\n", encoding="utf-8")
        html = TestClient(app).get("/").text
        assert "<script>alert(1)</script>" not in html


class TestEmptyTraceDir:
    def test_metrics_endpoint_works_with_no_traces(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "traces_dir", tmp_path)
        resp = TestClient(app).get("/metrics")
        assert resp.status_code == 200
        assert resp.json()["total_spans"] == 0

    def test_dashboard_renders_with_no_traces(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "traces_dir", tmp_path)
        resp = TestClient(app).get("/")
        assert resp.status_code == 200
        assert "No spans recorded yet" in resp.text
