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
4. **Answer questions** about either revision, or about what changed between them, with hybrid
   retrieval (deterministic exact-tag lookup unioned with vector search over both revisions'
   content *and* the delta report itself) + cross-encoder reranking + a citation-or-refuse system
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

## Design decisions & trade-offs

### Determinism in the delta path

The brief calls out a specific bar: delta alignment and add/remove/modify classification must
never depend on an LLM. This isn't just documented — it's proven by a test that patches the LLM
client to explode if called and asserts the classifier path never touches it
(`tests/test_delta_engine.py::TestLLMNeverCalled::test_classifier_never_invokes_llm_client`). The
LLM only ever writes prose (chat answers, the eval judge) or reads a hard-to-OCR image crop —
never decides what changed. Trade-off: this rules out using an LLM to fuzzy-match content the
deterministic tiers miss — e.g. a different tag-numbering convention (`FAILURE_MODES.md` §3) —
accepted deliberately, because a delta that isn't reproducible on the same two inputs isn't
trustworthy at any scale.

### One canonical schema, not three per-format models

`src/canonical/model.py`'s `Element`/`CanonicalDocument` is the only shape `delta/align.py`,
`chat/index.py`, `markup/overlay.py`, and the eval harness ever see — native PDF, scanned PDF,
and DWG/DXF each normalize into it in their own adapter and nowhere else. Trade-off: DWG's actual
richness (layers, blocks, ATTRIBs) gets flattened into a generic `attributes` dict rather than
kept as a first-class DWG object, so a DWG-specific downstream feature would have to reach past
the schema — but it's exactly what makes a 4th format a new adapter, not a new code path threaded
through every downstream module.

### Exact key, then geometry, then embeddings — never the reverse

The three alignment tiers in `src/delta/align.py` are ordered cheapest-and-most-certain first on
purpose: exact tag/instrument-key match, then DWG geometry, then embedding + bbox proximity only
for what's left over. Trade-off: content the classifiers don't recognize as a key falls through
every tier to the most expensive, least certain one — the exact mechanism behind the scanned-PDF
precision problem (`FAILURE_MODES.md` §1) — but it keeps the common case fast, cheap, and
provably deterministic instead of guessing first and correcting later.

### Hybrid retrieval, and exact matches always win the rerank budget

`chat/index.py` unions a deterministic exact-tag lookup with chromadb vector search, so a tag
literally named in the question is never left to embedding similarity alone. `chat/answer.py`
then reranks only the vector-only half of the candidate set within
`remaining_budget = rerank_top_k - len(exact matches)` — every exact match survives, full stop.
Trade-off: a deliberate precision bias on tag lookups over the reranker's own judgment, accepted
because a wrong citation is worse than a slightly fuller context window.

### The delta report is retrievable too, not just the two documents

`chat_index.index_delta()` (`src/chat/index.py`) indexes each delta entry as its own chunk,
phrased as a sentence ("CHANGED ... from '257 bar (g)' to '260 bar (g)'") — because a raw element
from either revision can only ever ground "what is X," never "what changed about X"; only the
delta itself knows the relationship between the two revisions. Verified live, not just unit
tested: asking a running session "What changed about the PSV setpoints between the two
revisions?" returns an answer citing both a normal document element *and* a delta chunk
(`[delta:modified:pair1_A:pdf_native:00141:pair1_B:pdf_native:00822]`) in the same response.
Retrieval-quality regression-checked against the existing labeled QA set before and after (both
0.4545 recall@k / MRR — unchanged). Trade-off, found the same day this shipped: a delta whose
underlying text is itself low-information (pair1's "added" note is a bare `"NOTE 29"`
cross-reference marker, not the note's actual content — a pre-existing data characteristic, not
new) can still lose the ranking race against dozens of near-identical bare references already in
the corpus. Vector search only helps when the underlying text has enough signal to embed well;
it doesn't fix a genuinely uninformative source string.

### Cite-or-refuse, not best-effort

The chat system prompt is told explicitly to answer only from retrieved chunks and say
`NOT_GROUNDED: ...` otherwise — the 1.00 refusal accuracy below is a direct consequence of that
instruction, not a lucky outcome. Trade-off: a reasonable question whose answer sits just outside
the retrieved top-k gets an honest refusal instead of a plausible-sounding guess. Intentional,
given the brief asks for *grounded* chat, not a helpful chatbot.

## What we deliberately cut

Not gaps found after the fact — things considered and consciously left out, with the reasoning
that drove each call:

- **Spatial-context enrichment at retrieval index time.** A bare value like `HH: 250` sitting
  near an instrument tag could be auto-attached to its nearest tag at index time. Rejected: on a
  dense layout a *different* tag can sit closer to a stray value than the tag it actually
  belongs to (measured: `TIT-9064` sits ~0.006 normalized-bbox-units closer to a nearby value
  than the tag that value actually describes) — a positional heuristic risks a confidently wrong
  attachment, worse than the honest retrieval gap it would "fix." Full reasoning:
  `FAILURE_MODES.md` §2.
- **Gating CI on the full LLM-judge eval suite.** `make eval`'s chat correctness/groundedness
  judge costs real money, takes several minutes, and depends on a free-tier key that can run dry
  mid-run (lived this first-hand — see `FAILURE_MODES.md` §1). CI instead runs `--delta-only` —
  fast, free, deterministic — on every push; the full suite stays a deliberate on-demand
  `make eval`, its last real output committed at `eval/scorecard.json`.
- **Dashboard authentication and a persistent session store.** In-memory, single-process, no
  login (`src/dashboard/state.py`'s own module docstring says so). Fine for `make dashboard` as a
  local demo tool for this submission; a real scope boundary, not an oversight — see
  `FAILURE_MODES.md` §7.
- **Multi-turn conversation memory in chat.** `answer_question()` takes one question at a time —
  no prior-turn context is threaded in from the dashboard or the eval harness
  (`src/chat/answer.py:110`). A follow-up like "what about its low limit?" is answered as a
  standalone question, not resolved against the previous exchange. Cut for scope, not because it
  would be hard to add — retrieval and reranking wouldn't need to change, only prompt assembly.
- **Switching OCR/vision providers.** Spent part of a day live-testing tesseract against two
  free-tier vision-model alternatives (Gemini, then OpenRouter). Both hit hard quota walls before
  enough specimens completed to draw a real conclusion, so the call was to keep tesseract plus
  the OCR-tolerant classification fix that's actually verified, rather than swap engines on
  inconclusive same-day data. See `FAILURE_MODES.md` §1 for the real, if incomplete, numbers.

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

## What's next

With more time, in priority order:

1. **Close the two still-open OCR corruption mechanisms** — digit-field corruption with lost
   punctuation, and tesseract's block segmentation genuinely differing from pymupdf's (it
   truncates multi-line notes). Neither is fixed by the character-confusion tolerance already
   shipped; both need their own approach. `FAILURE_MODES.md` §1 has the exact repro for each.
2. **Re-run and republish the full eval scorecard** after the OCR-tolerant classification fix —
   it's verified standalone (3 elements rescued, 819→816 false positives on the targeted bug
   class) but not yet folded into a fresh `eval/scorecard.json`, so the committed aggregate above
   still reflects the pre-fix number.
3. **Finish the OCR-provider comparison properly** — either a paid tier or a longer test window
   than one free-tier afternoon allows, to turn today's inconclusive data into an actual answer
   on whether a better OCR/vision model measurably improves scanned-PDF precision.
4. **Job-queue scanned-PDF ingest and vision fallback** instead of synchronous request/response —
   already the measured bottleneck at current scale (`eval/cost_latency_report.py`), and the
   first thing that breaks at 500-sheet volume (`INTERVIEW_PREP.md`'s scaling section).
5. **Shard chromadb retrieval by document/pair** instead of one collection per chat session, and
   give the dashboard a real session store — both documented, deliberate scope boundaries for
   this submission, not things to carry forward silently into a real deployment.
