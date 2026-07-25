# Document Delta & Grounded Chat

Takes two revisions of an engineering drawing (native PDF, scanned PDF, or DWG/DXF) and tells
you, deterministically, what actually changed — then lets you ask questions about either
revision and get cited answers instead of guesses.

Built for an Applied AI Engineer take-home. Full design rationale in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md), the execution contract in
[`BRIEF.md`](BRIEF.md), a live-verified demo script in [`DEMO.md`](DEMO.md), the candid failure
table in [`FAILURE_MODES.md`](FAILURE_MODES.md), and interview prep with real evidence in
[`INTERVIEW_PREP.md`](INTERVIEW_PREP.md). Every provenance claim in this repo — every sample
edit, every measured number, every bug found and fixed — is written up in
[`data/samples/PROVENANCE.md`](data/samples/PROVENANCE.md), which is the single largest and most
honest document here.

## What it actually does

1. **Ingest** three formats into one canonical schema: native PDF (`pymupdf` text extraction),
   scanned PDF (`tesseract` OCR + a vision-LLM second read for low-confidence regions), and
   DWG/DXF (`ezdxf`, real entity parsing — layers, blocks, ATTRIBs, dimensions).
2. **Diff** two revisions with a three-tier deterministic alignment engine — exact tag/instrument
   key match, then DWG geometry (layer + entity type + position), then embedding + bbox
   proximity for untagged text — and classify every result as added / removed / modified with a
   confidence score. **No LLM in this path, at all** — see
   [`tests/test_delta_engine.py::TestLLMNeverCalled`](tests/test_delta_engine.py).
3. **Report** the delta as JSON, Markdown, and an HTML report with charts, plus the delta drawn
   back onto the source document as colored bbox overlays (green/red/amber for
   added/removed/modified).
4. **Answer questions** about either revision with hybrid retrieval (deterministic exact-tag
   lookup unioned with vector search) + cross-encoder reranking + a citation-or-refuse system
   prompt, so an unanswerable question gets "I don't have grounding for that," not a guess.
5. **Observe** all of it — every stage emits an OTel-shaped trace span and a structured JSON log
   line from the first phase onward, aggregated live at `/metrics` with no separate metrics
   store.
6. **Evaluate** all of the above against 4 labeled real-edit pairs (delta P/R/F1), a 15-question
   labeled QA set (chat correctness/groundedness via an LLM judge, itself validated against
   independent human labels on every one of those 15 questions — not a small sample), and
   retrieval recall@k/MRR — with `make eval-diff` to catch a regression before it ships.

## Quickstart

```bash
uv sync --all-groups
cp .env.example .env   # then paste a free-tier key -- see .env's own inline instructions
make check              # ruff + mypy + full test suite (deterministic paths need no key)
make dashboard           # http://127.0.0.1:8001 -- submit a pair, view the report, chat about it
```

No API key is required for delta computation, ingestion, or the test suite — the determinism
boundary in point 2 above is load-bearing, not just a design note. A key is only needed for
scanned-PDF vision fallback and for grounded chat.

## Architecture

```
                    ┌─────────────┐
  PDF (native) ────▶│             │
  PDF (scanned)────▶│   ingest    │──▶ CanonicalDocument (pydantic v2, one schema for all 3 formats)
  DWG / DXF ───────▶│             │         │
                    └─────────────┘         │
                                             ▼
                              ┌───────────────────────────┐
                              │   delta/align.py          │  tier 1: exact key
                              │   (deterministic,         │  tier 2: DWG geometry
                              │    no LLM)                │  tier 3: embedding + bbox proximity
                              └───────────────────────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                   delta/report.py    markup/overlay.py   eval/metrics.py
                   JSON/MD/HTML       colored bbox         P/R/F1 vs.
                   + charts           overlay export       ground truth
                                             
                              ┌───────────────────────────┐
  question ───────────────────▶   chat/index.py            │
                              │   exact lookup + vector    │
                              │   search (chromadb)        │
                              └──────────────┬─────────────┘
                                             ▼
                              chat/rerank.py (cross-encoder)
                                             ▼
                              chat/answer.py (LLM: cite-or-refuse)
                                             ▼
                                     cited answer or refusal

  Every stage above ──▶ observability/tracing.py + logging.py ──▶ traces/{trace_id}.jsonl
                                                                          │
                                                                          ▼
                                                        observability/metrics.py (live aggregation)
                                                                          │
                                                                          ▼
                                                                    served at /metrics
```

`src/dashboard/app.py` is a thin FastAPI layer over all of the above — no new logic, just
routing and session bookkeeping (plain HTML forms, no client-side JS framework).

## Determinism, deliberately

The brief calls out a specific bar: delta alignment and add/remove/modify classification must
never depend on an LLM. This isn't just documented — it's proven by a test that patches the LLM
client to explode if called and asserts the classifier path never touches it
(`tests/test_delta_engine.py::TestLLMNeverCalled::test_classifier_never_invokes_llm_client`). The
LLM only ever writes prose (chat answers, the eval judge) or reads a hard-to-OCR image crop —
never decides what changed.

## Testing & eval

```bash
make check                                    # ruff + mypy + pytest (527 tests, 96%+ coverage)
uv run python -m eval.run_eval --delta-only --out eval/scorecard_ci.json   # what CI runs on every push
make eval                                     # full suite: needs LLM_API_KEY, several minutes, real cost
make eval-diff OLD=eval/scorecard.json NEW=<new run>   # flags any regression, non-zero exit
make metrics                                   # serve /metrics standalone
```

CI (`.github/workflows/ci.yml`) runs lint, type-check, the full unit suite, and the
`--delta-only` eval scorecard on every push — fast, free, deterministic, no secrets required. The
full LLM-judge suite is deliberately not run on every push (cost, time, free-tier quota
fragility); it's a real, on-demand `make eval` against the same 4 pairs and 15 questions, with
the last real run's numbers committed at `eval/scorecard.json`:

| Metric | Value |
|---|---|
| Delta recall (aggregate, 4 pairs) | 0.82 |
| Chat avg correctness / groundedness | 3.93 / 4.73 (of 5) |
| Chat refusal accuracy (unanswerable Qs correctly declined) | 1.00 |
| Judge/human agreement | 15/15 exact, on both dimensions |
| Retrieval recall@k / MRR | 0.45 / 0.45 |

Delta *precision* is not in that table on purpose — see [`FAILURE_MODES.md`](FAILURE_MODES.md)
for why, and what actually drives it.

## Repo layout

```
src/
├─ ingest/       pdf_native.py, pdf_scanned.py, dwg.py, registry.py
├─ canonical/    model.py -- the one schema every adapter produces
├─ delta/        align.py (3 tiers), engine.py, report.py, colors.py
├─ chat/         index.py, rerank.py, llm.py, answer.py
├─ markup/       overlay.py
├─ observability/ tracing.py, logging.py, metrics.py, metrics_server.py
└─ dashboard/    app.py, state.py, templates/
eval/            metrics.py, judge.py, retrieval_eval.py, cost_latency_report.py, run_eval.py
data/samples/    4 labeled pairs + a 5th stress/scaling set + PROVENANCE.md (source, edits,
                 and every real finding)
tests/           ~50 files, one per module, integration tests against real sample data
```

## Known limitations

Stated plainly rather than buried: scanned-PDF delta precision is genuinely weak (see the table
above and [`FAILURE_MODES.md`](FAILURE_MODES.md) for the root cause and what was and wasn't
fixed), DWG-binary conversion has never run against a real `.dwg` file in this environment, and
classification is grounded in one client's tag-numbering convention, not a general P&ID
standard. None of this is hidden — it's measured, root-caused, and documented, which is a
different thing from "working."
