# Interview prep

Five questions any reviewer would ask about a system like this, each answered with a pointer at
a real repo artifact — not a rehearsed line. If a claim below can't be traced to a file, a test,
or a measured number, it isn't in here.

## "Walk me through how you align content between two revisions. Where does it break?"

Three tiers, run in order, in `src/delta/align.py`, each handing its leftovers to the next:

1. **Exact key match** — literal text equality on tags, instrument loops, valve numbers, line
   numbers (`element_key()`). Deterministic, highest confidence, and — because two identical
   tags can appear on one sheet (e.g. two PSVs sharing a setpoint string) — disambiguated by
   nearest position, not first-match. Scoped per-page (`bbox.page` is part of the candidate key
   in all three tiers), which matters: see below.
2. **Geometry match** (DWG only) — same layer + entity type + bbox proximity, for content with no
   text key at all (a moved line, an unlabeled block).
3. **Embedding + bbox proximity** — for everything left over. Position within a tight radius is
   decisive on its own; outside that, embedding similarity has to clear a threshold too. A
   same-type gate rejects cross-type coincidences (see the negative-control story below).

**Where it genuinely breaks:** two near-identical, un-keyed notes disambiguated only by embedding
+ position — if both signals are close for two different candidates, the greedy assignment can
pick the wrong one. This was the anticipated hard case going in, not a surprise found later —
checked directly in `tests/test_delta_align.py::TestEmbeddingProximityMatch` and
`TestNoteSplitIntoTwo`. Full writeup: `FAILURE_MODES.md` §5.

**A real bug I found and fixed, not just anticipated:** every sample pair through the first
eleven phases was single-page, so normalized (page-*relative*) bbox distance being used as if it
were absolute was never exercised. Two same-position elements on *different* pages were
indistinguishable by distance and got resolved by an arbitrary id-string tie-break — confirmed
with a minimal repro (`DUPTAG` on page 0 and page 1) that reproduced the engine reporting the
*wrong page's* element as removed. Fixed by scoping all three tiers to `bbox.page`
(`tests/test_delta_align.py::TestPageScoping`). Second-order effect: this also converts
alignment from an all-pages-vs-all-pages O(n·m·N²) scan into N independent same-page O(n·m·N)
sub-problems — confirmed by Pair 5's own near-linear scaling curve (§ scaling, below).

## "Where is the LLM in your delta path, and why there and not elsewhere?"

It isn't. `tests/test_delta_engine.py::TestLLMNeverCalled::test_classifier_never_invokes_llm_client`
patches the LLM client to raise if called at all, then runs the full delta pipeline and asserts
it never does. Alignment, classification (added/removed/modified), and confidence scoring are
pure Python + a frozen local embedding model (`SentenceTransformerEmbedder`, config default
`all-MiniLM-L6-v2`) — never a network call.

**Why:** a delta report has to be the same every time you run it on the same two inputs. An LLM
call doesn't guarantee that, is slower, costs money per run, and can't be meaningfully unit
tested (you can't assert a specific alignment decision against a model that might phrase things
differently next time). The LLM only ever does two things in this system: write prose it isn't
asked to make factual judgments in (the chat answer, told explicitly to cite-or-refuse) and read
a hard-to-OCR image crop (vision fallback) — never decide *what changed*.

## "Show me a trace for a slow or failed request."

`traces/{trace_id}.jsonl`, one JSON line per span, OTel-shaped (`trace_id`, `span_id`,
`parent_span_id`, `duration_ms`, `status`, `attributes`). Two real ones worth having open:

- **Slow but successful:** any `pdf_scanned.ingest` span from a Pair 2 run — tesseract's OCR pass
  alone measured ~20s, and `pdf_scanned.vision_fallback` (the LLM re-read of low-confidence
  regions) dominates on top of that. This is the system's own honestly-reported bottleneck, not
  a cherry-picked slow case — `eval/cost_latency_report.py`'s own generated analysis says so in
  the same words.
- **Failed, with a real root cause:** any `llm.read_image_text` span with `status: "error"` from
  today's live OCR-provider comparison — Gemini's free tier returned a 429 with `"quotaValue":
  '20'` (a **daily** cap, already spent) disguised behind a deceptively short "retry in 22s"
  message, which is exactly the kind of signal that looks like an ordinary rate limit but isn't
  one. Real, dated example of reading an error message carefully instead of trusting its shape.

## "How would your eval catch a regression you introduced tomorrow?"

```bash
make eval-diff OLD=eval/scorecard.json NEW=<new_run>.json
```

`eval/eval_diff.py` compares every tracked metric (delta P/R/F1, chat correctness/groundedness,
refusal accuracy, over-refusal rate, retrieval recall@k/MRR) and prints each one's before/after
with a `REGRESSED` flag and non-zero exit if anything got worse. This isn't hypothetical — it's
the exact tool used today to verify each hardening fix actually helped rather than just
"should help": e.g. Pair 1's delta precision going 0.021 → 0.625 after the tight-band matching
fix showed up as `improved`, immediately, in real output, not as a claim.

CI runs a fast, free, deterministic slice of this on every push (`--delta-only`, no LLM, no
secrets — see `.github/workflows/ci.yml`); the full LLM-judge suite is a deliberate `make eval`
on demand, because it costs real money and real time against a real rate limit.

## "What would you do differently for a 500-sheet P&ID set?"

**Alignment itself already scales close to linearly**, not because it was designed to from day
one but because fixing the page-scoping bug above happened to also fix an O(N²) scaling cliff —
confirmed on real data: Pair 5 (the same content duplicated across 1/2/5/10 pages) measured
~1.6-1.8s/page from N=2 onward, not the quadratic blowup the pre-fix version would have hit.

**What genuinely wouldn't scale as-is, in priority order:**

1. **Scanned-PDF ingest and vision fallback.** Already the measured bottleneck at *current*
   scale (`eval/cost_latency_report.py`'s own "10x/100x" section). At 500 sheets this needs a job
   queue, not synchronous request/response, plus fallback-call batching or a stricter per-document
   region cap (`VISION_FALLBACK_MAX_REGIONS` already exists as that knob).
2. **Free-tier LLM rate limits are a real wall, not a theoretical one — I hit this myself today.**
   Testing two different free vision-model tiers (Gemini, then OpenRouter) for a same-day OCR
   comparison, I ran into a **daily** 20-request cap on one model and a ~50% upstream-rate-limit
   failure rate on the other, inside a single afternoon of testing 13 crops. At 500-sheet volume
   with hundreds of low-confidence regions per document, a free tier doesn't degrade gracefully,
   it stops outright. Production scale needs a paid tier with real throughput guarantees, request
   batching, and a queue that backs off on 429 without burning wall-clock time the way a naive
   per-call retry does (`src/chat/llm.py::_with_rate_limit_wait` already fails fast when a
   *stated* wait exceeds budget — the gap found today is that a *daily* cap can report a
   deceptively short retry window, so the fail-fast check doesn't catch it; a real fix needs to
   distinguish "wait this long" from "come back tomorrow" from the error body, not just the
   number).
3. **Retrieval indexing is one chromadb collection per chat session.** Fine for a handful of
   concurrent demo sessions; at scale this should shard by document/pair, both to bound
   per-query candidate-pool size and so re-indexing one changed document doesn't touch unrelated
   ones.
4. **The dashboard is single-process and in-memory.** A real deployment needs a real session
   store and horizontal workers — a documented, deliberate scope boundary for a take-home demo,
   not something carried forward silently.
