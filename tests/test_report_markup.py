"""Tests for delta/report.py's markup-overlay integration specifically:
real Pair 3 DXF files (small, fast) exercised through render_html()/
write_report() end to end, plus the graceful-degradation contract when a
source file is missing (test_delta_report.py's fake-doc tests already prove
this implicitly; this makes it explicit and asserts the log warning).
"""

import logging
from pathlib import Path

import pytest

from src.delta.engine import compute_delta
from src.delta.report import render_html, write_report
from src.ingest.dwg import DwgAdapter

PAIR3_A = Path("data/samples/pair3/A.dxf")
PAIR3_B = Path("data/samples/pair3/B.dxf")


@pytest.fixture(scope="module")
def dxf_pair():
    doc_a = DwgAdapter().ingest(PAIR3_A, pid="rm_pair3_A", revision_label="rev A")
    doc_b = DwgAdapter().ingest(PAIR3_B, pid="rm_pair3_B", revision_label="rev B")
    report = compute_delta(doc_a, doc_b)
    return report, doc_a, doc_b


class TestMarkupSectionInHtml:
    def test_section_present_with_real_files(self, dxf_pair):
        report, doc_a, doc_b = dxf_pair
        html = render_html(report, doc_a, doc_b)
        assert "Markup overlay" in html
        assert html.count('<img class="markup-img"') == 2

    def test_legend_present(self, dxf_pair):
        report, doc_a, doc_b = dxf_pair
        html = render_html(report, doc_a, doc_b)
        assert "markup-legend" in html
        assert "Added" in html and "Removed" in html and "Modified" in html

    def test_still_self_contained_no_external_resources(self, dxf_pair):
        report, doc_a, doc_b = dxf_pair
        html = render_html(report, doc_a, doc_b)
        assert "http://" not in html
        assert "https://" not in html

    def test_no_section_without_both_docs(self, dxf_pair):
        report, doc_a, _doc_b = dxf_pair
        html = render_html(report, doc_a, None)
        assert "Markup overlay" not in html

    def test_download_links_included_when_markup_links_given(self, dxf_pair, tmp_path):
        report, doc_a, doc_b = dxf_pair
        links = {"a": tmp_path / "t_A.png", "b": tmp_path / "t_B.png"}
        html = render_html(report, doc_a, doc_b, markup_links=links)
        assert 'href="t_A.png"' in html
        assert 'href="t_B.png"' in html

    def test_missing_source_file_degrades_gracefully(self, dxf_pair, caplog):
        report, doc_a, doc_b = dxf_pair
        broken_doc_a = doc_a.model_copy(update={"raw_source_path": "no/such/file.dxf"})
        with caplog.at_level(logging.WARNING):
            html = render_html(report, broken_doc_a, doc_b)
        assert "Markup overlay" not in html
        # the rest of the report must still be intact
        assert "Delta Report" in html
        assert "<table>" in html


class TestWriteReportProducesMarkupFiles:
    def test_markup_files_written_and_returned(self, dxf_pair, tmp_path):
        report, doc_a, doc_b = dxf_pair
        paths = write_report(report, tmp_path, doc_a, doc_b, basename="pair3")
        assert "markup_a" in paths
        assert "markup_b" in paths
        assert paths["markup_a"].exists()
        assert paths["markup_b"].exists()
        assert paths["markup_a"].suffix == ".png"

    def test_html_file_links_to_markup_files_by_relative_name(self, dxf_pair, tmp_path):
        report, doc_a, doc_b = dxf_pair
        paths = write_report(report, tmp_path, doc_a, doc_b, basename="pair3")
        html = paths["html"].read_text(encoding="utf-8")
        assert f'href="{paths["markup_a"].name}"' in html
        assert f'href="{paths["markup_b"].name}"' in html

    def test_missing_source_file_does_not_fail_write_report(self, dxf_pair, tmp_path):
        report, doc_a, doc_b = dxf_pair
        broken_doc_a = doc_a.model_copy(update={"raw_source_path": "no/such/file.dxf"})
        paths = write_report(report, tmp_path, broken_doc_a, doc_b, basename="pair3")
        assert set(paths.keys()) == {"json", "markdown", "html"}
        assert paths["html"].exists()
