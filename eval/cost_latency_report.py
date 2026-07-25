"""Cost / latency budget analysis (plan §10): a short written analysis
generated from real trace data -- cost per delta run, cost per chat turn,
where latency actually goes, and what it'd look like at 10x/100x document
volume. Numbers are computed live from src.observability.metrics (the same
snapshot /metrics serves); only the narrative framing around them is
authored judgment, not a separately hand-maintained figure.
"""

from src.observability.metrics import LatencyStats, MetricsSnapshot, compute_metrics_snapshot

# Stage-name -> human label, grouped by where in the pipeline it sits.
# Stages not present in a given snapshot (e.g. no scanned-PDF runs yet) are
# simply omitted from the rendered table, not shown as zero.
_INGEST_STAGES = {
    "pdf_native.ingest": "PDF native: full ingest",
    "pdf_scanned.ingest": "PDF scanned: full ingest",
    "pdf_scanned.tesseract": "PDF scanned: tesseract OCR pass",
    "pdf_scanned.render": "PDF scanned: page rasterize",
    "pdf_scanned.vision_fallback": "PDF scanned: vision-LLM fallback (per region)",
    "dwg.ingest": "DWG/DXF: full ingest",
}
_DELTA_STAGES = {
    "delta.load_embedding_model": "Delta: embedding model load (one-time/process)",
    "delta.align.exact_key_match": "Delta: tier 1 (exact key)",
    "delta.align.geometry_match": "Delta: tier 2 (geometry)",
    "delta.align.embedding_proximity_match": "Delta: tier 3 (embedding proximity)",
    "delta.compute": "Delta: full compute (all tiers + classify)",
}
_CHAT_STAGES = {
    "chat.retrieve.exact": "Chat: exact tag lookup",
    "chat.retrieve.vector": "Chat: vector search",
    "chat.retrieve.hybrid": "Chat: hybrid retrieval (full)",
    "chat.rerank": "Chat: cross-encoder rerank",
    "chat.answer": "Chat: full turn (retrieve+rerank+LLM)",
}
_LLM_STAGES = {
    "llm.complete": "LLM: text completion call",
    "llm.read_image_text": "LLM: vision OCR-fallback call",
}


def _table(snapshot: MetricsSnapshot, stage_labels: dict[str, str]) -> list[str]:
    rows = []
    for name, label in stage_labels.items():
        stats = snapshot.latency_by_stage.get(name)
        if stats is None:
            continue
        rows.append(_row(label, stats))
    return rows


def _row(label: str, stats: LatencyStats) -> str:
    return (
        f"| {label} | {stats.count} | {stats.mean_ms:,.1f} | {stats.p50_ms:,.1f} | "
        f"{stats.p95_ms:,.1f} | {stats.max_ms:,.1f} |"
    )


_TABLE_HEADER = "| Stage | Count | Mean (ms) | p50 | p95 | Max |\n|---|---|---|---|---|---|"


def _cost_section(snapshot: MetricsSnapshot) -> str:
    llm = snapshot.llm
    lines = [
        "## Cost",
        "",
        f"Total spend across {llm.total_calls} LLM call(s) in the scanned trace history: "
        f"**${llm.total_cost_usd:.4f}**.",
        "",
    ]
    chat = llm.by_purpose.get("chat_answer")
    judge = llm.by_purpose.get("eval_judge")
    delta_runs = snapshot.delta.runs
    lines.append(
        f"- **Cost per delta run:** $0.0000 (deterministic, {delta_runs} run(s) measured -- "
        "the delta engine never calls an LLM, by design; see the LLM-never-called test in "
        "tests/test_delta_engine.py)."
    )
    if chat:
        per_turn = chat.cost_usd / chat.calls if chat.calls else 0.0
        lines.append(
            f"- **Cost per chat turn:** ${per_turn:.6f} average ({chat.calls} turn(s), "
            f"${chat.cost_usd:.4f} total, {chat.input_tokens:,} input / "
            f"{chat.output_tokens:,} output tokens)."
        )
    else:
        lines.append("- **Cost per chat turn:** no chat turns recorded yet in this trace history.")
    if judge:
        per_call = judge.cost_usd / judge.calls if judge.calls else 0.0
        lines.append(
            f"- **Cost per eval-judge call:** ${per_call:.6f} average ({judge.calls} call(s), "
            f"kept separate from chat-turn cost via the purpose tag -- otherwise judge "
            "eval spend would silently inflate the reported per-turn figure)."
        )
    if llm.total_cost_usd == 0.0 and llm.total_calls > 0:
        lines.append(
            "\n*(All $0.00 above: LLM_COST_PER_MTOK_INPUT/OUTPUT are configured as 0 for this "
            "project's free-tier provider. Set them to a provider's real sticker price to see "
            "actual dollar figures -- the token counts themselves are real and unaffected.)*"
        )
    return "\n".join(lines)


def _latency_section(snapshot: MetricsSnapshot) -> str:
    lines = ["## Where the latency goes", ""]
    for title, stages in (
        ("Ingest", _INGEST_STAGES),
        ("Delta engine", _DELTA_STAGES),
        ("Chat", _CHAT_STAGES),
        ("Raw LLM calls", _LLM_STAGES),
    ):
        rows = _table(snapshot, stages)
        if not rows:
            continue
        lines += [f"### {title}", "", _TABLE_HEADER, *rows, ""]
    return "\n".join(lines)


def _scaling_section(snapshot: MetricsSnapshot) -> str:
    ingest = snapshot.latency_by_stage.get("pdf_native.ingest")
    scanned = snapshot.latency_by_stage.get("pdf_scanned.ingest")
    embed_load = snapshot.latency_by_stage.get("delta.load_embedding_model")
    lines = ["## At 10x / 100x document volume", ""]

    if ingest:
        lines.append(
            f"- **Native PDF ingest** measured at {ingest.mean_ms:,.0f}ms/doc average "
            f"(p95 {ingest.p95_ms:,.0f}ms). At 10x (tens of documents) this is still trivially "
            "single-machine; at 100x (hundreds), ingest itself is not the bottleneck -- it's "
            "embarrassingly parallel across documents with no shared state, so it scales by "
            "adding workers, not by algorithmic changes."
        )
    if scanned:
        lines.append(
            f"- **Scanned PDF ingest is the real bottleneck**: measured at "
            f"{scanned.mean_ms:,.0f}ms/doc average (p95 {scanned.p95_ms:,.0f}ms) -- tesseract "
            "plus any vision-LLM fallback calls dominate. At 100x volume this stage alone "
            "would need a job queue (not synchronous request/response) and fallback-call "
            "batching or a stricter per-document fallback-region cap "
            "(VISION_FALLBACK_MAX_REGIONS already exists as the knob) to keep worst-case "
            "documents from starving the queue."
        )
    lines.append(
        "- **Delta alignment is O(n x m) per tier over the leftover pool after coarser tiers "
        "thin it** (documented in src/delta/align.py) -- fine at hundreds of elements per sheet, "
        "but a 500-sheet set doing full pairwise scans per pair would want a spatial index "
        "(KD-tree/grid) instead, since the leftover pool after exact-key matching won't shrink "
        "proportionally on much denser or noisier sheets."
    )
    if embed_load:
        lines.append(
            f"- **Embedding/reranker model load is a one-time process cost** "
            f"({embed_load.mean_ms:,.0f}ms measured), amortized by the class-level caching "
            "already in place (SentenceTransformerEmbedder, CrossEncoderReranker) -- at higher "
            "volume this cost is paid once per worker process, not once per document, so it "
            "matters for cold-start latency but not steady-state throughput."
        )
    lines.append(
        "- **Retrieval indexing is currently one chromadb collection per chat session** "
        "(src/chat/index.py) -- at 100x document volume this should shard by document/pair "
        "rather than growing one global collection, both to bound per-query candidate-pool size "
        "and so re-indexing one changed document doesn't touch unrelated ones (incremental "
        "re-index, not full reindex)."
    )
    return "\n".join(lines)


def generate_cost_latency_report(snapshot: MetricsSnapshot | None = None) -> str:
    snapshot = snapshot or compute_metrics_snapshot()
    return "\n\n".join(
        [
            "# Cost / Latency Budget Analysis",
            f"Generated {snapshot.generated_at.isoformat()} from "
            f"{snapshot.trace_files_scanned} trace file(s) / {snapshot.total_spans} span(s).",
            _cost_section(snapshot),
            _latency_section(snapshot),
            _scaling_section(snapshot),
        ]
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate the cost/latency budget analysis.")
    parser.add_argument(
        "--out", type=Path, default=None, help="write to this file instead of stdout"
    )
    args = parser.parse_args()

    report_text = generate_cost_latency_report()
    if args.out:
        args.out.write_text(report_text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report_text)
