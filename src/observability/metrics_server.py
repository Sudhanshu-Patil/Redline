"""Serves the metrics snapshot (plan §9 / §12 Phase 7): GET /metrics (JSON)
and GET / (an HTML dashboard view). Built as an APIRouter rather than a
bare FastAPI app so Phase 11's fuller dashboard can `include_router()` this
instead of re-implementing metrics rendering.
"""

from xml.sax.saxutils import escape as xml_escape

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

from src.observability import tracing
from src.observability.metrics import (
    DeltaTotals,
    LLMTelemetry,
    MetricsSnapshot,
    compute_metrics_snapshot,
)

router = APIRouter()

_HTML_STYLE = """
<style>
  .metrics-root {
    color-scheme: light;
    --surface: #fcfcfb; --page: #f9f9f7;
    --ink-primary: #0b0b0b; --ink-secondary: #52514e; --ink-muted: #898781;
    --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
    --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    .metrics-root {
      color-scheme: dark;
      --surface: #1a1a19; --page: #0d0d0d;
      --ink-primary: #ffffff; --ink-secondary: #c3c2b7; --ink-muted: #898781;
      --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
    }
  }
  body { margin: 0; background: var(--page); color: var(--ink-primary);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .metrics-root { max-width: 900px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 32px 0 12px; }
  .subtitle { color: var(--ink-secondary); font-size: 13px; margin-bottom: 24px; }
  .stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 8px; }
  .stat-tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
               padding: 12px 16px; min-width: 110px; }
  .stat-tile .label { font-size: 12px; color: var(--ink-secondary); }
  .stat-tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
  .stat-tile.error .value { color: var(--critical); }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 20px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--gridline); }
  th { color: var(--ink-secondary); font-weight: 600; font-size: 12px; }
  td.mono, th.mono { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 12px; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .empty { color: var(--ink-muted); font-size: 13px; }
</style>
"""


def _latency_table(snapshot: MetricsSnapshot) -> str:
    if not snapshot.latency_by_stage:
        return "<p class='empty'>No spans recorded yet.</p>"
    rows = []
    for name, stats in sorted(snapshot.latency_by_stage.items()):
        rows.append(
            "<tr>"
            f"<td class='mono'>{xml_escape(name)}</td>"
            f"<td class='num'>{stats.count}</td>"
            f"<td class='num'>{stats.mean_ms:.1f}</td>"
            f"<td class='num'>{stats.p50_ms:.1f}</td>"
            f"<td class='num'>{stats.p95_ms:.1f}</td>"
            f"<td class='num'>{stats.p99_ms:.1f}</td>"
            f"<td class='num'>{stats.max_ms:.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Stage</th><th>Count</th><th>Mean (ms)</th>"
        "<th>p50</th><th>p95</th><th>p99</th><th>Max</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _llm_table(llm: LLMTelemetry) -> str:
    if not llm.by_model:
        return "<p class='empty'>No LLM calls recorded yet.</p>"
    rows = []
    for model, m in sorted(llm.by_model.items()):
        rows.append(
            "<tr>"
            f"<td class='mono'>{xml_escape(model)}</td>"
            f"<td class='num'>{m.calls}</td>"
            f"<td class='num'>{m.input_tokens:,}</td>"
            f"<td class='num'>{m.output_tokens:,}</td>"
            f"<td class='num'>${m.cost_usd:.4f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Model</th><th>Calls</th><th>Input tok</th>"
        "<th>Output tok</th><th>Cost (USD)</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _stat_tile(label: str, value: str, css_class: str = "stat-tile") -> str:
    return (
        f"<div class='{css_class}'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div></div>"
    )


def _delta_summary(delta: DeltaTotals) -> str:
    if delta.runs == 0:
        return "<p class='empty'>No delta runs recorded yet.</p>"
    align = f"{delta.avg_alignment_rate:.1%}" if delta.avg_alignment_rate is not None else "—"
    exact = f"{delta.avg_exact_key_rate:.1%}" if delta.avg_exact_key_rate is not None else "—"
    tiles = [
        _stat_tile("Runs", str(delta.runs)),
        _stat_tile("Added", str(delta.total_added)),
        _stat_tile("Removed", str(delta.total_removed)),
        _stat_tile("Modified", str(delta.total_modified)),
        _stat_tile("Avg alignment", align),
        _stat_tile("Avg exact-key", exact),
    ]
    return "<div class='stat-row'>" + "".join(tiles) + "</div>"


def _retrieval_summary(snapshot: MetricsSnapshot) -> str:
    r = snapshot.retrieval
    if r is None:
        return "<p class='empty'>No retrieval queries recorded yet (Phase 8 not run).</p>"
    recall = f"{r.recall_at_k:.1%}" if r.recall_at_k is not None else "—"
    mrr = f"{r.mean_reciprocal_rank:.3f}" if r.mean_reciprocal_rank is not None else "—"
    tiles = [
        _stat_tile("Queries", str(r.queries)),
        _stat_tile("Recall@k", recall),
        _stat_tile("MRR", mrr),
    ]
    return "<div class='stat-row'>" + "".join(tiles) + "</div>"


def render_metrics_html(snapshot: MetricsSnapshot) -> str:
    error_class = "stat-tile error" if snapshot.error_rate > 0 else "stat-tile"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Document Delta &amp; Grounded Chat — Metrics</title>
{_HTML_STYLE}
</head>
<body>
<div class="metrics-root">
  <h1>Metrics</h1>
  <div class="subtitle">Generated {xml_escape(snapshot.generated_at.isoformat())} — live snapshot
    over {snapshot.trace_files_scanned} trace file(s), no separately maintained store.</div>

  <div class="stat-row">
    {_stat_tile("Trace files", str(snapshot.trace_files_scanned))}
    {_stat_tile("Total spans", str(snapshot.total_spans))}
    {_stat_tile("Error rate", f"{snapshot.error_rate:.1%}", error_class)}
  </div>

  <h2>Latency by stage</h2>
  {_latency_table(snapshot)}

  <h2>LLM usage</h2>
  {_llm_table(snapshot.llm)}

  <h2>Delta engine</h2>
  {_delta_summary(snapshot.delta)}

  <h2>Retrieval</h2>
  {_retrieval_summary(snapshot)}
</div>
</body>
</html>
"""


@router.get("/metrics", response_model=MetricsSnapshot)
def get_metrics() -> MetricsSnapshot:
    with tracing.span("metrics.snapshot", format="json"):
        return compute_metrics_snapshot()


@router.get("/", response_class=HTMLResponse)
def get_dashboard() -> str:
    with tracing.span("metrics.snapshot", format="html"):
        snapshot = compute_metrics_snapshot()
    return render_metrics_html(snapshot)


def create_app() -> FastAPI:
    app = FastAPI(title="Document Delta & Grounded Chat — Metrics")
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    from src.config import settings

    uvicorn.run(app, host=settings.metrics_host, port=settings.metrics_port)
