# Demo script

A live walkthrough, ~3-4 minutes, hitting every core + bonus item in one pass: all three
ingestion formats, the delta engine, markup overlay, grounded chat with citations and refusal,
and observability. Everything below uses the 5 bundled sample pairs — no file prep needed.

## Setup (before the room fills up)

```bash
uv sync --all-groups
cp .env.example .env   # paste a free-tier key (Groq recommended -- see .env's inline instructions)
make dashboard          # http://127.0.0.1:8001
```

Confirm it's live: the home page shows "Submit a pair" plus 4 quick-start sample buttons
(pair1–pair4) and an empty session list.

## 1. Native PDF delta — the primary, most-scrutinized pair (45s)

Click **pair1** on the home page. This is a real P&ID
(`26-KA-901`, a lift-gas compressor) with 6 hand-made edits: two relief-valve setpoints uprated,
an instrument trip limit changed, a note deleted, a note added, a flow-rate value changed on a
table far from any tag key.

Land on the session page. Point at the stats: **added / removed / modified** counts, and the
**alignment rate** — this is the fraction of all elements the deterministic aligner successfully
paired across revisions, computed with zero LLM calls.

Click **View report**. Scroll to the delta table and find the two identical `SP = 257 bar (g)` →
`SP = 260 bar (g)` edits, correctly disambiguated to two different valves by bbox proximity
alone — this is the aligner's own stated hard case, solved, not avoided.

## 2. DWG/DXF delta — the format most take-homes skip (30s)

Back to home, click **pair3**. Same delta pipeline, different adapter: `ezdxf` reads real layers,
blocks, and ATTRIBs. Point out the `matched_by_tier` breakdown includes **geometry** matches
(same layer + entity type + position) — a match strategy that only makes sense for DWG's richer
structure, and that PDF pairs never use.

Click through to the report and show the **DWG markup download** — the delta rasterized and
redrawn on top of the actual drawing geometry, not a generic overlay.

## 3. Scanned PDF — OCR, live, honestly (30s)

Click **pair2**. This is the *same* PSV/PIT/flow-rate edits as pair1, but revision B was
rasterized and OCR'd instead of read natively. Point at the lower alignment rate versus pair1 —
say plainly that this is the weakest link in the system (see `FAILURE_MODES.md` for exactly why,
with real numbers), not something to talk around.

## 4. Negative control — proving the aligner isn't gullible (20s)

Click **pair4**: two genuinely unrelated compressor P&IDs, not a revision pair. Point at the
**low exact-key alignment warning** on the session page — the system telling you, unprompted,
"these don't look like the same document," instead of confidently dumping hundreds of spurious
edits.

## 5. Grounded chat — citations and refusal (60s)

From any session, scroll to the "Grounded chat" card and click **Open chat**. Ask a real question:

> What is the set pressure of PSV-9066A?

Answer comes back with an inline citation like `[pair1_B:pdf_native:00824]` — click through the
report to find that exact element if you want to prove it isn't fabricated.

Now ask something genuinely not in the document:

> What is the warranty period for the PSV valves?

Answer: `NOT_GROUNDED: ...` — an explicit refusal, not a guess. This is the system prompt's
cite-or-refuse rule, tested in `eval/run_eval.py`'s refusal-accuracy metric (currently 1.00 on
the labeled set — every unanswerable question correctly declined).

## 6. Observability (20s)

Navigate to `/metrics/`. Live-aggregated from the trace files on disk, not a separate store:
latency percentiles per pipeline stage, LLM token/cost totals broken down by purpose (chat
answer vs. eval judge — never conflated), delta counts, retrieval hit rate. Point out the
scanned-PDF and vision-fallback rows are the visibly slowest — matches what was just said in
step 3, backed by numbers instead of a claim.

## If asked to go deeper

- **"What's the failure table look like?"** → `FAILURE_MODES.md` — root-caused, not hand-waved,
  with a real example of a corrected model (the OCR-tolerant reclassification fix) alongside an
  honestly unresolved one.
- **"How would this scale to 500 sheets?"** → `INTERVIEW_PREP.md`'s scaling section, backed by
  Pair 5's real measured timing curve (`data/samples/PROVENANCE.md`).
- **"Show me a trace."** → any file under `traces/`; the scanned-PDF or vision-fallback ones are
  the most interesting (multi-second LLM calls sitting inside an otherwise sub-second pipeline).
