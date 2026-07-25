"""Delta report renderer: JSON / Markdown / HTML views over a DeltaReport
(plan §5), plus two small inline-SVG bar charts for the HTML view --
changes by page and changes by element type. Self-contained: no JS
framework, no external assets.

Color: added/modified/removed map onto the dataviz skill's fixed status
palette (good/warning/critical) -- validated for contrast and CVD-safety,
and the same green/amber/red convention the Phase 9 markup overlay uses, so
a reviewer sees one consistent color language across the report and the
annotated drawing.
"""

import base64
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from src.canonical.model import CanonicalDocument
from src.config import settings
from src.delta.colors import CHANGE_ORDER, STATUS_COLORS, STATUS_LABELS
from src.delta.engine import Delta, DeltaReport
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_SURFACE = "#fcfcfb"

_INK_PRIMARY_DARK = "#ffffff"
_INK_SECONDARY_DARK = "#c3c2b7"
_INK_MUTED_DARK = "#898781"
_GRIDLINE_DARK = "#2c2c2a"
_SURFACE_DARK = "#1a1a19"


def _delta_page(d: Delta) -> int | None:
    bbox = d.new_bbox or d.old_bbox
    return bbox.page if bbox else None


def _empty_change_counts() -> dict[str, int]:
    return dict.fromkeys(CHANGE_ORDER, 0)


def counts_by_change_type(deltas: list[Delta]) -> dict[str, int]:
    counts = _empty_change_counts()
    for d in deltas:
        counts[d.change_type] += 1
    return counts


def counts_by_page(deltas: list[Delta]) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for d in deltas:
        page = _delta_page(d)
        if page is None:
            continue
        result.setdefault(page, _empty_change_counts())[d.change_type] += 1
    return result


def counts_by_element_type(deltas: list[Delta]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for d in deltas:
        result.setdefault(d.element_type, _empty_change_counts())[d.change_type] += 1
    return result


def _is_chart_worthy(d: Delta) -> bool:
    """Excludes single-character fragments from the *chart* rows only (the
    full tables below always list every delta -- nothing is hidden from the
    report as a whole). Below `alignment_min_embed_text_len`, the engine
    itself never attempts to match this text (see delta/align.py), so these
    show up as symmetric added/removed noise -- on Pair 1 that's 236 single-
    character flags vs. 3 real setpoint edits, enough to bury the actual
    story in the by-type chart. Same threshold as the engine, for the same
    reason: below it there's no real signal to visualize.
    """
    text = d.new_text if d.new_text is not None else d.old_text
    if not text:
        return False
    return len(text.strip()) >= settings.alignment_min_embed_text_len


# --- JSON ---


def render_json(report: DeltaReport) -> str:
    return report.model_dump_json(indent=2)


# --- Markdown ---


def _doc_summary_line(pid: str, doc: CanonicalDocument | None) -> str:
    if doc is None:
        return f"- **{pid}**"
    rev = doc.revision_label or "(no revision label)"
    return (
        f"- **{pid}** — {doc.format}, {rev}, {doc.page_count} page(s), {len(doc.elements)} elements"
    )


def render_markdown(
    report: DeltaReport,
    doc_a: CanonicalDocument | None = None,
    doc_b: CanonicalDocument | None = None,
) -> str:
    s = report.stats
    lines = [
        f"# Delta Report: {report.pid_a} → {report.pid_b}",
        "",
        _doc_summary_line(report.pid_a, doc_a),
        _doc_summary_line(report.pid_b, doc_b),
        "",
    ]

    if report.warnings:
        lines.append("> ⚠️ " + "  \n> ".join(report.warnings))
        lines.append("")

    lines += [
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Added | {s.added} |",
        f"| Removed | {s.removed} |",
        f"| Modified | {s.modified} |",
        f"| Unchanged (matched, no diff) | {s.unchanged} |",
        f"| Alignment rate | {s.alignment_rate:.1%} |",
        f"| Exact-key rate | {s.exact_key_rate:.1%} |",
        f"| Matched by tier | exact_key={s.matched_by_tier.get('exact_key', 0)}, "
        f"geometry={s.matched_by_tier.get('geometry', 0)}, "
        f"embedding_proximity={s.matched_by_tier.get('embedding_proximity', 0)} |",
        "",
    ]

    for change_type in CHANGE_ORDER:
        group = [d for d in report.deltas if d.change_type == change_type]
        if not group:
            continue
        lines.append(f"## {STATUS_LABELS[change_type]} ({len(group)})")
        lines.append("")
        lines.append("| Type | Old | New | Confidence | Tier |")
        lines.append("|---|---|---|---|---|")
        for d in group:
            old = (d.old_text or "").replace("|", "\\|") or "—"
            new = (d.new_text or "").replace("|", "\\|") or "—"
            tier = d.match_tier or "—"
            lines.append(f"| {d.element_type} | {old} | {new} | {d.confidence:.2f} | {tier} |")
        lines.append("")

    return "\n".join(lines)


# --- HTML: inline SVG stacked bar chart ---

_BAR_HEIGHT = 20  # <=24px per mark spec
_ROW_GAP = 10
_SEGMENT_GAP = 2  # surface-color gap between stacked segments
_CHART_LEFT_PAD = 130  # room for row labels
_CHART_WIDTH = 480
_CHART_RIGHT_PAD = 50  # room for the total-value end label


def _estimate_text_width(text: str, font_size: float = 11) -> float:
    return len(text) * font_size * 0.55


def _svg_stacked_bar_chart(
    title: str, rows: list[tuple[str, dict[str, int]]], chart_id: str
) -> str:
    """rows: list of (row_label, {change_type: count}). One horizontal
    stacked bar per row; segments in CHANGE_ORDER; legend once at the top.
    """
    if not rows:
        return ""
    max_total = max(sum(counts.values()) for _, counts in rows) or 1
    plot_width = _CHART_WIDTH - _CHART_LEFT_PAD - _CHART_RIGHT_PAD
    row_height = _BAR_HEIGHT + _ROW_GAP
    chart_height = len(rows) * row_height + 40  # + legend/title header room
    baseline_x = _CHART_LEFT_PAD

    svg_parts = [
        f'<svg class="viz-root" viewBox="0 0 {_CHART_WIDTH} {chart_height}" '
        f'width="100%" height="{chart_height}" role="img" aria-labelledby="{chart_id}-title">',
        f'<title id="{chart_id}-title">{xml_escape(title)}</title>',
    ]

    # Legend
    legend_x: float = baseline_x
    legend_y = 14
    for change_type in CHANGE_ORDER:
        svg_parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" rx="2" '
            f'fill="{STATUS_COLORS[change_type]}"/>'
        )
        label = STATUS_LABELS[change_type]
        svg_parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y}" class="viz-legend">{label}</text>'
        )
        legend_x += 14 + _estimate_text_width(label, 11) + 16

    y = 36
    for row_label, counts in rows:
        total = sum(counts.values())
        svg_parts.append(
            f'<text x="{baseline_x - 8}" y="{y + _BAR_HEIGHT / 2 + 4}" '
            f'text-anchor="end" class="viz-row-label">{xml_escape(row_label)}</text>'
        )
        x: float = baseline_x
        for i, change_type in enumerate(CHANGE_ORDER):
            count = counts.get(change_type, 0)
            if count == 0:
                continue
            seg_width = (count / max_total) * plot_width
            gap = _SEGMENT_GAP if i > 0 and x > baseline_x else 0
            x += gap
            seg_width = max(seg_width - gap, 1)
            tooltip = f"{STATUS_LABELS[change_type]}: {count}"
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{seg_width:.1f}" height="{_BAR_HEIGHT}" '
                f'rx="4" fill="{STATUS_COLORS[change_type]}">'
                f"<title>{xml_escape(tooltip)}</title></rect>"
            )
            if _estimate_text_width(str(count)) <= seg_width - 6:
                svg_parts.append(
                    f'<text x="{x + seg_width / 2:.1f}" y="{y + _BAR_HEIGHT / 2 + 4}" '
                    f'text-anchor="middle" class="viz-seg-label">{count}</text>'
                )
            x += seg_width
        # total value at the tip
        total_y = y + _BAR_HEIGHT / 2 + 4
        svg_parts.append(
            f'<text x="{x + 8:.1f}" y="{total_y}" class="viz-row-total">{total}</text>'
        )
        y += row_height

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


_HTML_STYLE = """
<style>
  body, .viz-root, .report-root {
    color-scheme: light;
    --surface: #fcfcfb; --page: #f9f9f7;
    --ink-primary: #0b0b0b; --ink-secondary: #52514e; --ink-muted: #898781;
    --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
    --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    body, .viz-root, .report-root {
      color-scheme: dark;
      --surface: #1a1a19; --page: #0d0d0d;
      --ink-primary: #ffffff; --ink-secondary: #c3c2b7; --ink-muted: #898781;
      --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
    }
  }
  /* Tokens must be defined ON body (not just its .report-root child) --
     the next rule reads var(--page)/var(--ink-primary) on body itself, and
     CSS custom properties are only visible to the element they're declared
     on and its descendants, never an ancestor. Getting this backwards was a
     real bug: it left body's own background/color unresolved, so dark-mode
     users saw default-black text on a dark-but-not-explicitly-set canvas
     inside every box that inherited color from body instead of a
     .report-root descendant that had a color of its own. */
  body { margin: 0; background: var(--page); color: var(--ink-primary);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .report-root { max-width: 900px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 32px 0 12px; color: var(--ink-primary); }
  .subtitle { color: var(--ink-secondary); font-size: 13px; margin-bottom: 24px; }
  .doc-line { color: var(--ink-secondary); font-size: 13px; margin: 2px 0; }
  .warning-banner { background: var(--warning); color: #1a1a19; padding: 10px 14px;
                     border-radius: 6px; font-size: 13px; margin: 16px 0; }
  .stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 8px; }
  .stat-tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
               padding: 12px 16px; min-width: 110px; }
  .stat-tile .label { font-size: 12px; color: var(--ink-secondary); }
  .stat-tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
  .stat-tile.added .value { color: var(--good); }
  .stat-tile.modified .value { color: #a06700; }
  .stat-tile.removed .value { color: var(--critical); }
  .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
                padding: 16px; margin: 12px 0; }
  .viz-legend, .viz-row-label, .viz-row-total { fill: var(--ink-secondary); font-size: 11px; }
  .viz-seg-label { fill: #ffffff; font-size: 10px; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 20px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--gridline); }
  th { color: var(--ink-secondary); font-weight: 600; font-size: 12px; }
  td.mono { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 12px; }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px;
           font-weight: 600; color: #ffffff; }
  .badge.added { background: var(--good); }
  .badge.modified { background: #a06700; }
  .badge.removed { background: var(--critical); }
  .conf { color: var(--ink-muted); font-size: 12px; }
  .markup-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0 20px; }
  .markup-col { flex: 1 1 320px; min-width: 280px; }
  .markup-caption { font-size: 12px; color: var(--ink-secondary); margin-bottom: 6px; }
  .markup-img { width: 100%; border: 1px solid var(--border); border-radius: 8px;
                background: var(--surface); }
  .markup-legend { display: flex; gap: 14px; font-size: 12px; color: var(--ink-secondary);
                    margin: 8px 0 4px; }
  .markup-legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                            margin-right: 4px; vertical-align: middle; }
</style>
"""


def _render_deltas_table(deltas: list[Delta]) -> str:
    if not deltas:
        return "<p class='conf'>No changes.</p>"
    rows = []
    for d in deltas:
        badge = f'<span class="badge {d.change_type}">{STATUS_LABELS[d.change_type]}</span>'
        old = xml_escape(d.old_text or "—")
        new = xml_escape(d.new_text or "—")
        rows.append(
            "<tr>"
            f"<td>{badge}</td>"
            f"<td class='mono'>{d.element_type}</td>"
            f"<td>{old}</td>"
            f"<td>{new}</td>"
            f"<td class='conf'>{d.confidence:.2f}</td>"
            f"<td class='conf'>{d.match_tier or '—'}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Change</th><th>Type</th><th>Old</th><th>New</th>"
        "<th>Confidence</th><th>Tier</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _markup_section(
    report: DeltaReport,
    doc_a: CanonicalDocument | None,
    doc_b: CanonicalDocument | None,
    markup_links: dict[str, Path] | None = None,
) -> str:
    """Embeds a before/after raster preview of the delta markup overlay
    (plan §8) with real per-box hover tooltips, so a reviewer sees WHERE a
    change is without leaving the HTML report. Local import: overlay.py
    pulls in fitz/ezdxf/matplotlib, which JSON/Markdown-only callers of this
    module shouldn't pay the import cost for.

    `markup_links`, when given (write_report() passes the paths
    write_markup() just wrote to the same out_dir), adds a download link to
    the real exported PDF/PNG alongside each preview -- the PDF export in
    particular carries live popup tooltips and vector fidelity the raster
    preview doesn't.
    """
    if doc_a is None or doc_b is None or not report.deltas:
        return ""
    from src.markup.overlay import render_markup_png

    try:
        png_a = render_markup_png(doc_a, report.deltas, side="a")
        png_b = render_markup_png(doc_b, report.deltas, side="b")
    except Exception as exc:
        # Optional visual extra over a source file the report doesn't
        # strictly need (missing/corrupt file, no ODA converter for a raw
        # .dwg, ...) -- the report's real content (stats/tables) must not
        # fail just because the source drawing isn't available to re-render.
        log.warning(
            "markup overlay unavailable for HTML embed",
            extra={
                "extra_fields": {
                    "pid_a": doc_a.pid, "pid_b": doc_b.pid, "error": str(exc),
                }
            },
        )  # fmt: skip
        return ""

    b64_a = base64.b64encode(png_a).decode("ascii")
    b64_b = base64.b64encode(png_b).decode("ascii")
    legend = "".join(
        f'<span><span class="swatch" style="background:{STATUS_COLORS[k]}"></span>'
        f"{STATUS_LABELS[k]}</span>"
        for k in CHANGE_ORDER
    )
    links = markup_links or {}
    link_a = (
        f' · <a href="{xml_escape(links["a"].name)}">download annotated {links["a"].suffix[1:]}</a>'
        if "a" in links
        else ""
    )
    link_b = (
        f' · <a href="{xml_escape(links["b"].name)}">download annotated {links["b"].suffix[1:]}</a>'
        if "b" in links
        else ""
    )
    return f"""
  <h2>Markup overlay</h2>
  <div class="markup-legend">{legend}</div>
  <div class="markup-row">
    <div class="markup-col">
      <div class="markup-caption">{xml_escape(report.pid_a)} (before){link_a}</div>
      <img class="markup-img" src="data:image/png;base64,{b64_a}"
           alt="Delta markup on {xml_escape(report.pid_a)}">
    </div>
    <div class="markup-col">
      <div class="markup-caption">{xml_escape(report.pid_b)} (after){link_b}</div>
      <img class="markup-img" src="data:image/png;base64,{b64_b}"
           alt="Delta markup on {xml_escape(report.pid_b)}">
    </div>
  </div>
"""


def render_html(
    report: DeltaReport,
    doc_a: CanonicalDocument | None = None,
    doc_b: CanonicalDocument | None = None,
    markup_links: dict[str, Path] | None = None,
) -> str:
    s = report.stats

    chart_deltas = [d for d in report.deltas if _is_chart_worthy(d)]
    excluded_count = len(report.deltas) - len(chart_deltas)

    by_page = counts_by_page(chart_deltas)
    page_rows = [(f"Page {p}", counts) for p, counts in sorted(by_page.items())]
    page_chart = _svg_stacked_bar_chart("Changes by page", page_rows, "chart-page")

    by_type = counts_by_element_type(chart_deltas)
    type_rows = sorted(by_type.items(), key=lambda kv: -sum(kv[1].values()))
    type_chart = _svg_stacked_bar_chart(
        "Changes by element type", [(t, c) for t, c in type_rows], "chart-type"
    )
    chart_note = ""
    if excluded_count:
        chart_note = (
            f'<p class="conf">{excluded_count} single-character fragment'
            f"{'s' if excluded_count != 1 else ''} omitted from the charts above "
            "(below the matching-confidence floor — see full tables below).</p>"
        )

    warning_html = ""
    if report.warnings:
        warning_html = "".join(
            f'<div class="warning-banner">⚠️ {xml_escape(w)}</div>' for w in report.warnings
        )

    doc_a_line = _doc_summary_line(report.pid_a, doc_a).lstrip("- ")
    doc_b_line = _doc_summary_line(report.pid_b, doc_b).lstrip("- ")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Delta Report: {xml_escape(report.pid_a)} vs {xml_escape(report.pid_b)}</title>
{_HTML_STYLE}
</head>
<body>
<div class="report-root">
  <h1>Delta Report</h1>
  <div class="subtitle">{xml_escape(report.pid_a)} → {xml_escape(report.pid_b)}</div>
  <div class="doc-line">{xml_escape(doc_a_line)}</div>
  <div class="doc-line">{xml_escape(doc_b_line)}</div>

  {warning_html}

  <div class="stat-row">
    <div class="stat-tile added">
      <div class="label">Added</div><div class="value">{s.added}</div>
    </div>
    <div class="stat-tile modified">
      <div class="label">Modified</div><div class="value">{s.modified}</div>
    </div>
    <div class="stat-tile removed">
      <div class="label">Removed</div><div class="value">{s.removed}</div>
    </div>
    <div class="stat-tile">
      <div class="label">Unchanged</div><div class="value">{s.unchanged}</div>
    </div>
    <div class="stat-tile">
      <div class="label">Alignment rate</div><div class="value">{s.alignment_rate:.0%}</div>
    </div>
  </div>
  {_markup_section(report, doc_a, doc_b, markup_links)}
  <h2>Changes by page</h2>
  <div class="chart-card">{page_chart}</div>

  <h2>Changes by element type</h2>
  <div class="chart-card">{type_chart}</div>
  {chart_note}

  <h2>Added ({s.added})</h2>
  {_render_deltas_table([d for d in report.deltas if d.change_type == "added"])}

  <h2>Removed ({s.removed})</h2>
  {_render_deltas_table([d for d in report.deltas if d.change_type == "removed"])}

  <h2>Modified ({s.modified})</h2>
  {_render_deltas_table([d for d in report.deltas if d.change_type == "modified"])}
</div>
</body>
</html>
"""


# --- write all three ---


def write_report(
    report: DeltaReport,
    out_dir: Path,
    doc_a: CanonicalDocument | None = None,
    doc_b: CanonicalDocument | None = None,
    basename: str = "delta_report",
) -> dict[str, Path]:
    with tracing.span("delta.report.write", pid_a=report.pid_a, pid_b=report.pid_b) as sp:
        out_dir.mkdir(parents=True, exist_ok=True)

        markup_paths: dict[str, Path] = {}
        if doc_a is not None and doc_b is not None and report.deltas:
            from src.markup.overlay import write_markup

            try:
                markup_paths = write_markup(doc_a, doc_b, report.deltas, out_dir, basename)
            except Exception as exc:
                log.warning(
                    "markup export failed; report will still be written without it",
                    extra={
                        "extra_fields": {
                            "pid_a": doc_a.pid, "pid_b": doc_b.pid, "error": str(exc),
                        }
                    },
                )  # fmt: skip

        paths = {
            "json": out_dir / f"{basename}.json",
            "markdown": out_dir / f"{basename}.md",
            "html": out_dir / f"{basename}.html",
        }
        paths["json"].write_text(render_json(report), encoding="utf-8")
        paths["markdown"].write_text(render_markdown(report, doc_a, doc_b), encoding="utf-8")
        html = render_html(report, doc_a, doc_b, markup_links=markup_paths or None)
        paths["html"].write_text(html, encoding="utf-8")
        paths.update({f"markup_{side}": p for side, p in markup_paths.items()})
        sp["files"] = list(paths.keys())
        return paths


if __name__ == "__main__":
    import argparse

    from eval.schema import load_manifest
    from src.delta.engine import compute_delta
    from src.ingest.dwg import DwgAdapter
    from src.ingest.pdf_native import PdfNativeAdapter
    from src.ingest.pdf_scanned import PdfScannedAdapter

    _ADAPTERS = {
        "pdf_native": PdfNativeAdapter,
        "pdf_scanned": PdfScannedAdapter,
        "dwg": DwgAdapter,
    }

    parser = argparse.ArgumentParser(description="Compute and render a delta report for a pair.")
    parser.add_argument("manifest", type=Path, help="path to a pair's manifest.json")
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    doc_a = _ADAPTERS[manifest.doc_a.format]().ingest(
        Path(manifest.doc_a.path),
        pid=manifest.doc_a.pid,
        revision_label=manifest.doc_a.revision_label,
    )
    doc_b = _ADAPTERS[manifest.doc_b.format]().ingest(
        Path(manifest.doc_b.path),
        pid=manifest.doc_b.pid,
        revision_label=manifest.doc_b.revision_label,
    )
    delta_report = compute_delta(doc_a, doc_b)
    paths = write_report(delta_report, args.out, doc_a, doc_b, basename=manifest.pair_id)
    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")
