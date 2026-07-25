"""Integration tests for the dashboard: real ingestion, real delta compute,
real chromadb+reranker -- but the final LLM call is faked by patching
LLMClient at the class level (answer_question()'s `llm or LLMClient()`
default picks it up transparently), so this suite makes no live network
call, consistent with tests/test_chat_answer.py's own FakeLLM/ExplodingLLM
pattern. The real end-to-end LLM path was verified live via manual
smoke-testing (see the phase summary), not exercised here.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import _safe_path_component, app
from src.dashboard.state import reset_sessions

PAIR1_A = "data/samples/originals/lift_gas_compressor_26-KA-901.pdf"
PAIR1_B = "data/samples/pair1/B.pdf"


class _FakeLLMClient:
    """Stands in for src.chat.answer.LLMClient -- patched at class level so
    answer_question()'s internal `llm or LLMClient()` picks it up without
    any dashboard route code needing to support injection."""

    def complete(self, system: str, user: str, max_tokens: int = 1024, purpose: str = "") -> str:
        return "NOT_GROUNDED: test fixture never answers for real."


@pytest.fixture(autouse=True)
def _clean_sessions():
    reset_sessions()
    yield
    reset_sessions()


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch):
    monkeypatch.setattr("src.chat.answer.LLMClient", _FakeLLMClient)


@pytest.fixture
def client():
    return TestClient(app)


class TestSafePathComponent:
    def test_plain_pid_passes_through(self):
        assert _safe_path_component("26-KA-901_A") == "26-KA-901_A"

    def test_script_tag_is_sanitized(self):
        # The exact input that broke report writing on Windows before this
        # was fixed: OSError from a literal "<script>alert(1)</script>"
        # path segment.
        result = _safe_path_component("<script>alert(1)</script>")
        assert "<" not in result
        assert ">" not in result
        assert "/" not in result

    def test_path_traversal_attempt_is_neutralized(self):
        result = _safe_path_component("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_empty_input_uses_fallback(self):
        assert _safe_path_component("", fallback="x") == "x"

    def test_only_unsafe_chars_uses_fallback(self):
        assert _safe_path_component("///???", fallback="x") == "x"

    def test_long_input_is_capped(self):
        assert len(_safe_path_component("a" * 500)) <= 80


def _submit_sample(client: TestClient, pair_id: str) -> str:
    r = client.post(f"/submit/sample/{pair_id}", follow_redirects=False)
    assert r.status_code == 303
    return r.headers["location"]


class TestIndex:
    def test_home_page_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Submit a pair" in r.text

    def test_shows_quick_start_sample_buttons(self, client):
        r = client.get("/")
        for pid in ["pair1", "pair2", "pair3", "pair4"]:
            assert f"/submit/sample/{pid}" in r.text

    def test_empty_session_list_shows_placeholder(self, client):
        r = client.get("/")
        assert "No sessions yet" in r.text


class TestSubmitSample:
    def test_loads_pair3_and_redirects_to_session(self, client):
        location = _submit_sample(client, "pair3")
        assert location.startswith("/session/")

    def test_unknown_sample_id_404s(self, client):
        r = client.post("/submit/sample/does-not-exist")
        assert r.status_code == 404

    def test_session_appears_in_home_list_afterward(self, client):
        location = _submit_sample(client, "pair4")
        r = client.get("/")
        assert location in r.text


class TestSessionView:
    def test_session_page_shows_stats(self, client):
        location = _submit_sample(client, "pair3")
        r = client.get(location)
        assert r.status_code == 200
        assert "Added" in r.text
        assert "Modified" in r.text

    def test_unknown_session_404s(self, client):
        r = client.get("/session/does-not-exist")
        assert r.status_code == 404

    def test_report_link_serves_real_delta_report_html(self, client):
        location = _submit_sample(client, "pair3")
        report = client.get(f"{location}/report")
        assert report.status_code == 200
        assert "Delta Report" in report.text
        assert "<svg" in report.text

    def test_report_404s_for_unknown_session(self, client):
        assert client.get("/session/nope/report").status_code == 404

    def test_download_json_is_valid_delta_report(self, client):
        location = _submit_sample(client, "pair1")
        dl = client.get(f"{location}/download/json")
        assert dl.status_code == 200
        payload = json.loads(dl.text)
        assert payload["pid_a"] and payload["pid_b"]

    def test_download_unknown_kind_404s(self, client):
        location = _submit_sample(client, "pair1")
        assert client.get(f"{location}/download/nonsense").status_code == 404

    def test_dwg_pair_offers_markup_download_links(self, client):
        location = _submit_sample(client, "pair3")
        r = client.get(location)
        assert f"{location}/download/markup_a" in r.text
        assert f"{location}/download/markup_b" in r.text


class TestSubmitUpload:
    def test_missing_files_shows_friendly_error(self, client):
        r = client.post(
            "/submit",
            data={"pid_a": "a", "pid_b": "b", "format_a": "pdf_native", "format_b": "pdf_native"},
        )
        assert r.status_code == 400
        assert "required" in r.text.lower()

    def test_real_file_upload_end_to_end(self, client):
        with open(PAIR1_A, "rb") as file_a, open(PAIR1_B, "rb") as file_b:
            r = client.post(
                "/submit",
                data={
                    "pid_a": "upload_A", "pid_b": "upload_B",
                    "format_a": "pdf_native", "format_b": "pdf_native",
                    "revision_label_a": "as-provided", "revision_label_b": "rev B",
                },  # fmt: skip
                files={
                    "doc_a_file": ("a.pdf", file_a, "application/pdf"),
                    "doc_b_file": ("b.pdf", file_b, "application/pdf"),
                },
                follow_redirects=False,
            )
        assert r.status_code == 303
        assert r.headers["location"].startswith("/session/")

    def test_mismatched_format_shows_friendly_error_not_500(self, client):
        # Claims a PDF is a DWG -- ezdxf.readfile() must fail cleanly and
        # the dashboard must show an error page, never a raw 500.
        with open(PAIR1_A, "rb") as file_a, open(PAIR1_B, "rb") as file_b:
            r = client.post(
                "/submit",
                data={
                    "pid_a": "bad_a", "pid_b": "bad_b",
                    "format_a": "dwg", "format_b": "pdf_native",
                },  # fmt: skip
                files={
                    "doc_a_file": ("a.pdf", file_a, "application/pdf"),
                    "doc_b_file": ("b.pdf", file_b, "application/pdf"),
                },
            )
        assert r.status_code == 400
        assert "Could not process" in r.text


class TestChat:
    def test_chat_page_loads_for_valid_session(self, client):
        location = _submit_sample(client, "pair3")
        r = client.get(f"{location}/chat")
        assert r.status_code == 200
        assert "Ask a question" in r.text

    def test_question_produces_a_turn_with_the_fake_llm(self, client):
        location = _submit_sample(client, "pair3")
        r = client.post(
            f"{location}/chat",
            data={"question": "What is the set pressure of PSV-9066A?"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        chat = client.get(f"{location}/chat")
        assert "What is the set pressure of PSV-9066A?" in chat.text
        assert "test fixture never answers for real" in chat.text

    def test_empty_question_is_ignored(self, client):
        location = _submit_sample(client, "pair3")
        client.post(f"{location}/chat", data={"question": "   "})
        chat = client.get(f"{location}/chat")
        assert "No questions asked yet" in chat.text

    def test_chat_404s_for_unknown_session(self, client):
        assert client.get("/session/nope/chat").status_code == 404

    def test_chat_post_404s_for_unknown_session(self, client):
        r = client.post("/session/nope/chat", data={"question": "hi"})
        assert r.status_code == 404


class TestMetricsIntegration:
    def test_metrics_json_endpoint_reachable(self, client):
        r = client.get("/metrics/metrics")
        assert r.status_code == 200
        assert "trace_files_scanned" in r.json()

    def test_metrics_html_dashboard_reachable(self, client):
        r = client.get("/metrics/")
        assert r.status_code == 200


class TestXssSafety:
    def test_pair_label_is_escaped_in_session_page(self, client):
        # pid fields are user-controlled on upload -- must never be
        # injected unescaped into the rendered session page.
        with open(PAIR1_A, "rb") as file_a, open(PAIR1_B, "rb") as file_b:
            r = client.post(
                "/submit",
                data={
                    "pid_a": "<script>alert(1)</script>", "pid_b": "b",
                    "format_a": "pdf_native", "format_b": "pdf_native",
                },  # fmt: skip
                files={
                    "doc_a_file": ("a.pdf", file_a, "application/pdf"),
                    "doc_b_file": ("b.pdf", file_b, "application/pdf"),
                },
                follow_redirects=True,
            )
        assert "<script>alert(1)</script>" not in r.text
        assert "&lt;script&gt;" in r.text
