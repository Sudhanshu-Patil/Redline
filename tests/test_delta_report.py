"""Tests for the delta report renderer: pure aggregation helpers, and the
JSON/Markdown/HTML renderers over a real DeltaReport produced by the engine
(hand-built elements + a fake embedder, same pattern as test_delta_engine.py).
"""

import json

import pytest

from src.canonical.model import CanonicalDocument
from src.delta.engine import compute_delta
from src.delta.report import (
    counts_by_change_type,
    counts_by_element_type,
    counts_by_page,
    render_html,
    render_json,
    render_markdown,
    write_report,
)
from tests.conftest import FakeEmbedder
from tests.conftest import make_positioned_element as el


def doc(pid: str, elements, fmt="pdf_native", revision_label=None) -> CanonicalDocument:
    return CanonicalDocument(
        pid=pid, format=fmt, revision_label=revision_label, page_count=1,
        elements=elements, raw_source_path=f"{pid}.pdf",
    )  # fmt: skip


@pytest.fixture
def sample_report():
    a = [
        el("26BL9072", element_type="valve", x0=0.05, y0=0.05, x1=0.10, y1=0.06),  # unchanged
        el("OLD-TAG", element_type="tag", x0=0.05, y0=0.20, x1=0.10, y1=0.21),  # removed
        # modified
        el("SP = 257 bar (g)", element_type="setpoint", x0=0.10, y0=0.10, x1=0.20, y1=0.11),
        el(
            "<script>alert(1)</script>", element_type="text_block",
            x0=0.30, y0=0.30, x1=0.40, y1=0.31,
        ),
    ]  # fmt: skip
    b = [
        el("26BL9072", element_type="valve", x0=0.05, y0=0.05, x1=0.10, y1=0.06),
        el("NEW-TAG", element_type="tag", x0=0.85, y0=0.85, x1=0.90, y1=0.86),  # added
        el("SP = 260 bar (g)", element_type="setpoint", x0=0.10, y0=0.10, x1=0.20, y1=0.11),
        # added, unsafe chars
        el("&fin<de>", element_type="text_block", x0=0.85, y0=0.10, x1=0.95, y1=0.11),
    ]
    embedder = FakeEmbedder(
        {
            "SP = 257 bar (g)": [1.0, 0.0, 0.0],
            "SP = 260 bar (g)": [1.0, 0.0, 0.0],
            "<script>alert(1)</script>": [0.0, 1.0, 0.0],
            "OLD-TAG": [0.0, 0.0, 1.0],
            "NEW-TAG": [0.0, 1.0, 0.0],
            "&fin<de>": [0.0, 0.0, 0.0],
        }
    )
    doc_a = doc("pair_A", a, revision_label="Rev A")
    doc_b = doc("pair_B", b, revision_label="Rev B")
    report = compute_delta(doc_a, doc_b, embedder=embedder)
    return report, doc_a, doc_b


class TestAggregationHelpers:
    def test_counts_by_change_type(self, sample_report):
        # removed=2: OLD-TAG and the <script> element (no B-side counterpart
        # for either); added=2: NEW-TAG and &fin<de>; modified=1: setpoint.
        report, _, _ = sample_report
        counts = counts_by_change_type(report.deltas)
        assert counts["removed"] == 2
        assert counts["modified"] == 1
        assert counts["added"] == 2
        assert sum(counts.values()) == len(report.deltas)

    def test_counts_by_page_all_on_page_zero(self, sample_report):
        report, _, _ = sample_report
        by_page = counts_by_page(report.deltas)
        assert set(by_page.keys()) == {0}
        assert sum(by_page[0].values()) == len(report.deltas)

    def test_counts_by_element_type(self, sample_report):
        report, _, _ = sample_report
        by_type = counts_by_element_type(report.deltas)
        assert by_type["tag"] == {"added": 1, "modified": 0, "removed": 1}
        assert by_type["setpoint"]["modified"] == 1


class TestRenderJson:
    def test_round_trips_to_valid_report(self, sample_report):
        from src.delta.engine import DeltaReport

        report, _, _ = sample_report
        text = render_json(report)
        restored = DeltaReport.model_validate_json(text)
        assert restored == report

    def test_is_valid_json(self, sample_report):
        report, _, _ = sample_report
        parsed = json.loads(render_json(report))
        assert parsed["pid_a"] == "pair_A"
        assert parsed["stats"]["added"] == 2


class TestRenderMarkdown:
    def test_contains_summary_and_doc_lines(self, sample_report):
        report, doc_a, doc_b = sample_report
        md = render_markdown(report, doc_a, doc_b)
        assert "pair_A" in md and "pair_B" in md
        assert "Rev A" in md and "Rev B" in md
        assert "| Added | 2 |" in md
        assert "| Removed | 2 |" in md
        assert "| Modified | 1 |" in md

    def test_includes_each_change_section(self, sample_report):
        report, doc_a, doc_b = sample_report
        md = render_markdown(report, doc_a, doc_b)
        assert "## Added (2)" in md
        assert "## Removed (2)" in md
        assert "## Modified (1)" in md

    def test_pipe_in_text_is_escaped_for_table_safety(self):
        from src.delta.engine import Delta, DeltaReport, DeltaStats

        report = DeltaReport(
            pid_a="A", pid_b="B",
            deltas=[
                Delta(
                    change_type="added", element_type="note", new_text="a | b",
                    confidence=1.0,
                )
            ],
            stats=DeltaStats(
                total_a=0, total_b=1, matched=0, unchanged=0, added=1, removed=0, modified=0,
                matched_by_tier={}, alignment_rate=0.0, exact_key_rate=0.0,
            ),
        )  # fmt: skip
        md = render_markdown(report)
        assert "a \\| b" in md

    def test_no_warnings_section_when_clean(self, sample_report):
        report, doc_a, doc_b = sample_report
        md = render_markdown(report, doc_a, doc_b)
        assert "⚠️" not in md


class TestRenderHtml:
    def test_contains_svg_charts(self, sample_report):
        report, doc_a, doc_b = sample_report
        html = render_html(report, doc_a, doc_b)
        assert html.count("<svg") == 2  # by-page and by-type charts

    def test_stat_tiles_show_correct_counts(self, sample_report):
        report, doc_a, doc_b = sample_report
        html = render_html(report, doc_a, doc_b)
        assert '<div class="value">2</div>' in html  # added / removed
        assert '<div class="value">1</div>' in html  # modified

    def test_script_tag_in_content_is_escaped_not_executable(self, sample_report):
        """Delta text comes from parsed PDF content -- if a drawing literally
        contained the string '<script>...', it must never become live HTML."""
        report, doc_a, doc_b = sample_report
        html = render_html(report, doc_a, doc_b)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_ampersand_and_angle_brackets_escaped(self, sample_report):
        report, doc_a, doc_b = sample_report
        html = render_html(report, doc_a, doc_b)
        assert "&fin<de>" not in html
        assert "&amp;fin&lt;de&gt;" in html

    def test_is_self_contained_no_external_resources(self, sample_report):
        report, doc_a, doc_b = sample_report
        html = render_html(report, doc_a, doc_b)
        assert "http://" not in html and "https://" not in html
        assert "<script" not in html.lower().replace("&lt;script", "")

    def test_dark_mode_media_query_present(self, sample_report):
        report, doc_a, doc_b = sample_report
        html = render_html(report, doc_a, doc_b)
        assert "prefers-color-scheme: dark" in html

    def test_warning_banner_rendered_when_present(self):
        from src.delta.engine import DeltaReport, DeltaStats

        report = DeltaReport(
            pid_a="A", pid_b="B", deltas=[],
            stats=DeltaStats(
                total_a=0, total_b=0, matched=0, unchanged=0, added=0, removed=0, modified=0,
                matched_by_tier={}, alignment_rate=0.0, exact_key_rate=0.0,
            ),
            warnings=["Low exact-key alignment (10.0%) between A and B"],
        )  # fmt: skip
        html = render_html(report)
        assert "warning-banner" in html
        assert "Low exact-key alignment" in html

    def test_single_character_deltas_excluded_from_charts_but_not_tables(self):
        """A dwarfing volume of below-threshold single-char noise (see
        PROVENANCE.md's Phase 5 findings) must not bury the real signal in
        the summary charts -- but every delta still appears in the tables."""
        from src.delta.engine import Delta, DeltaReport, DeltaStats

        deltas = [
            Delta(change_type="added", element_type="text_block", new_text="U", confidence=1.0),
            Delta(change_type="removed", element_type="text_block", old_text="C", confidence=1.0),
            Delta(
                change_type="modified", element_type="setpoint",
                old_text="SP = 257 bar (g)", new_text="SP = 260 bar (g)", confidence=1.0,
            ),
        ]  # fmt: skip
        report = DeltaReport(
            pid_a="A", pid_b="B", deltas=deltas,
            stats=DeltaStats(
                total_a=3, total_b=3, matched=1, unchanged=0, added=1, removed=1, modified=1,
                matched_by_tier={}, alignment_rate=0.33, exact_key_rate=0.0,
            ),
        )  # fmt: skip
        html = render_html(report)
        assert "1 single-character fragment" not in html  # 2 excluded, not 1
        assert "2 single-character fragments omitted" in html
        # the real signal (setpoint) still appears in the by-type chart
        assert "setpoint" in html
        # but the noise is still fully listed in the tables
        assert ">U<" in html
        assert ">C<" in html

    def test_no_chart_note_when_nothing_excluded(self, sample_report):
        report, doc_a, doc_b = sample_report
        html = render_html(report, doc_a, doc_b)
        assert "omitted from the charts" not in html

    def test_empty_deltas_render_without_crashing(self):
        from src.delta.engine import DeltaReport, DeltaStats

        report = DeltaReport(
            pid_a="A", pid_b="B", deltas=[],
            stats=DeltaStats(
                total_a=5, total_b=5, matched=5, unchanged=5, added=0, removed=0, modified=0,
                matched_by_tier={}, alignment_rate=1.0, exact_key_rate=1.0,
            ),
        )  # fmt: skip
        html = render_html(report)
        assert "No changes." in html
        assert "<svg" not in html  # no deltas -> no chart data


class TestWriteReport:
    def test_writes_all_three_files(self, sample_report, tmp_path):
        report, doc_a, doc_b = sample_report
        paths = write_report(report, tmp_path, doc_a, doc_b, basename="pairX")
        assert set(paths.keys()) == {"json", "markdown", "html"}
        assert paths["json"] == tmp_path / "pairX.json"
        assert paths["json"].exists()
        assert paths["markdown"].exists()
        assert paths["html"].exists()

    def test_written_json_is_the_same_as_render_json(self, sample_report, tmp_path):
        report, doc_a, doc_b = sample_report
        paths = write_report(report, tmp_path, doc_a, doc_b)
        assert paths["json"].read_text(encoding="utf-8") == render_json(report)

    def test_creates_output_directory_if_missing(self, sample_report, tmp_path):
        report, doc_a, doc_b = sample_report
        out_dir = tmp_path / "nested" / "reports"
        paths = write_report(report, out_dir, doc_a, doc_b)
        assert paths["html"].exists()
