# Sample data provenance

Tracks the origin and exact edits for every ingestion pair, per `IMPLEMENTATION_PLAN.md` §6.
Pairs 1, 2 and 4 are generated **reproducibly** by `scripts/synthesize_pairs.py` (`make data`);
the script asserts every edit hit its target and re-ingests the result through the Phase 1
adapter before writing ground truth — a failed edit fails the build, it can't silently produce
a wrong ground truth. Machine-readable ground truth lives in each pair's `ground_truth.json`
(schema: `eval/schema.py`); this file is the human-readable why.

## Originals (as provided with the assignment)

| File | Document | Role |
|---|---|---|
| `originals/lift_gas_compressor_26-KA-901.pdf` | Lift Gas Compressor P&ID, 26-KA-901 | Base for Pairs 1/2; A-side of Pair 4 |
| `originals/export_gas_compressor_26-KA-902.pdf` | Export Gas Compressor P&ID, 26-KA-902 | B-side of Pair 4 |

These are **companion documents, not a revision pair** — different equipment, different trains,
different design data (see plan §0). They serve as (a) structural validation for the native-PDF
adapter and (b) the negative-control Pair 4.

## Pair 1 — primary synthetic revision (native PDF vs native PDF)

- **A:** `originals/lift_gas_compressor_26-KA-901.pdf`, unmodified.
- **B:** `pair1/B.pdf` — A with six MOC-style edits applied via pymupdf redact-and-reinsert
  (text-only redactions; images/line art explicitly preserved). Replacement text is Helvetica
  5.5pt standing in for the source's Calibri 5.5pt — visually near-identical, structurally
  irrelevant to the delta engine.

| GT id | Edit | Old → New | Why this edit |
|---|---|---|---|
| GT-SP-9066A | PSV-9066A set pressure label | `SP = 257 bar (g)` → `SP = 260 bar (g)` | Two *identical* SP strings exist on the sheet; forces duplicate-text disambiguation by location |
| GT-SP-9066B | PSV-9066B set pressure label | `SP = 257 bar (g)` → `SP = 260 bar (g)` | Second half of the duplicate pair |
| GT-HH-9062 | PIT-9062 HH trip limit | `HH: 245` → `HH: 250` | Small numeric change on an instrument annotation — the classic easy-to-miss revision |
| GT-FLOW | Design table FLOW RATE (kg/h) | `19057` → `20500` | Value far from any tag key; exercises proximity/embedding matching, not exact-key matching |
| GT-NOTE30-DEL | Note 30 definition | removed (was `30. SAFETY CRITICAL HEAT TRACING - HYDRATE MITIGATION (25°C)`) | Whole-element removal; the three on-drawing `NOTE 30` callouts **intentionally remain** as a documented dangling-reference case |
| GT-NOTE37-ADD | New note 37 | added: `37. PSV 9066A/B SET PRESSURE REVISED TO 260 BAR(G).` | Whole-element addition, self-documenting MOC trail for the PSV change |

Deliberate scope notes: the title-block revision was *not* bumped (the edit list above is the
complete, literal ground truth), and the dangling NOTE 30 callouts are *not* ground-truth
changes. Verified crops of every edited region were reviewed during synthesis.

## Pair 2 — scanned revision (native A vs image-only B)

- **A:** same original as Pair 1 A (native text layer).
- **B:** `pair2/B_scanned.pdf` — Pair 1's B rasterized at 300 dpi (config `OCR_DPI`) and
  re-embedded as an image-only PDF. Verified to contain **no text layer**; page size preserved.
- **Ground truth:** identical deltas to Pair 1 (same revision, different carrier format). The
  B side must come through OCR, so extracted text will carry confidence scores and noise.
- Rasterization is clean (no synthetic skew/noise) — 5.5pt source text at 300 dpi is already a
  genuinely hard OCR case; degrading it further would test tesseract's floor, not our fallback
  logic.

## Pair 4 — negative control (two different documents)

- **A:** 26-KA-901 (Lift Gas), **B:** 26-KA-902 (Export Gas), both as provided.
- `expected_deltas: []`, `negative_control: true`. Expected engine behaviour: a
  **low-alignment-rate warning**, not a dump of hundreds of spurious adds/removes. Eval scores
  this pair on that behaviour alone.

## Pair 3 — DWG/DXF (authored schematic vs edited revision)

- **A:** `pair3/A.dxf`, **B:** `pair3/B.dxf` — both authored programmatically by
  `scripts/synthesize_pairs.py::_author_pair3_drawing` (ezdxf, R2018).
- **Why authored rather than converted:** no PDF→DXF converter preserves the text layer
  (text becomes vector outlines), which would destroy the exact-match keys the delta engine
  depends on; and neither ODA File Converter (DWG↔DXF only) nor any other conversion tooling
  ships on a stock machine. An authored DXF gives real files, a real ezdxf parse, exact
  provenance, and an entity mix chosen to exercise what makes DWG *different*: layers, block
  references, block ATTRIBs (tagged instrument bubbles), and dimension entities. The plan's
  alternative ("source a public-domain sample DWG") gives weaker provenance, not stronger.
  Binary `.dwg` ingest is still supported via the ODA auto-conversion hook when a converter
  is installed.
- **Drawing content (A):** layers PIPING/INSTRUMENTS/TEXT/NOTES/EQUIPMENT/DIMS; VALVE_GATE
  block inserted twice with tag labels (26BL9072, 26BL9075); INSTR_BUBBLE block with TAG
  ATTRIBs (PIT-9062, PSV-9066A); setpoint texts (`SP = 257 bar (g)`, `HH: 245`); a line
  number; an MTEXT notes block; a linear dimension with text override `600`; a drain stub
  LINE.

| GT id | Edit | Old → New | Why |
|---|---|---|---|
| GT3-SP | Setpoint TEXT | `SP = 257 bar (g)` → `SP = 260 bar (g)` | Cross-format echo of Pair 1's key edit |
| GT3-MOVE | VALVE_GATE INSERT (26BL9075) | (120, 38.5) → (128, 44.5) | Moved block, same layer/block: the §4.2 geometry-match rule's showcase; its label moves too |
| GT3-DIM | Dimension override | `600` → `750` | Dimension-entity change, DWG-specific |
| GT3-ADD-VALVE | New VALVE_GATE + label + tie-in LINE | added `43BL9020` | Added tagged geometry |
| GT3-DEL-DRAIN | Drain stub LINE | removed | Pure geometry removal — no text key; only the geometry rule can catch it |

## Pair 5 — stress set (Phase 12)

Not a checked-in manifest/ground-truth pair like 1-4 (deliberate scope boundary: plan §10's
"4-5 labeled pairs" is already satisfied by the 4 eval-harness pairs, and Pair 5's actual job —
"approximate a larger sheet set, for the scaling note" (§6/§14) — is a *measurement*, not
another P/R/F1 fixture). Built and run by `scripts/stress_test_pair5.py`
(`uv run python scripts/stress_test_pair5.py`): Pair 1's real, already-verified A/B documents
(six real edits) are duplicated across N synthetic pages — bit-for-bit the same real content
per page, just relabeled onto a different `bbox.page`, so ground truth is exactly "N × Pair 1's
six edits," no new synthesis risk. The two large per-scale `A.canonical.json`/`B.canonical.json`
dumps (~4MB each, N=10) are gitignored as deterministically regenerable; the actual measurements
(`data/samples/pair5/stress_results.json`) are committed as the durable evidence.

**Real measured results** (native PDF, `compute_delta()` wall time, same machine as every other
timing in this repo):

| Pages | Elements/side | Seconds | Added | Removed | Modified |
|---|---|---|---|---|---|
| 1 | 825 | 42.1 *(one-time embedding-model warmup, see below)* | 119 | 119 | 4 |
| 2 | 1,650 | 3.2 | 238 | 238 | 8 |
| 5 | 4,125 | 8.4 | 595 | 595 | 20 |
| 10 | 8,250 | 17.7 | 1,190 | 1,190 | 40 |

Two things worth reading precisely, not just the raw numbers:

1. **Added/removed/modified scale exactly N×** (119→238→595→1190 is exactly 119×1/2/5/10; same
   for modified at 4×). That's the correctness signal: no cross-page contamination, each page's
   real edits detected independently and completely.
2. **Time scales close to linearly with page count from N=2 onward** (≈1.6–1.8s/page: 3.2s/2,
   8.4s/5, 17.7s/10) -- N=1's 42.1s is a one-time cost (SentenceTransformerEmbedder's first
   real inference in the process, not just model load — the class-level cache means every
   later call in the same run is warm) and not representative of steady-state per-page cost;
   reported as measured rather than discarded, since a real deployment pays this exact cost on
   its first request too.

**This stress test is what actually found the Phase 12 page-scoping bug** (see the delta-engine
findings section below) — every one of Pairs 1-4 is single-page, so nothing before Pair 5 ever
exercised two elements sharing identical *normalized* bbox coordinates while sitting on
different pages.

## Delta engine findings from real data (Phase 5)

Three real bugs were found and fixed by running the engine against these pairs rather than
only hand-built fixtures (see `src/delta/align.py` module docstring for full detail):

1. **Tight-proximity ties.** Collapsing every within-tolerance candidate to an identical score
   let an arbitrary id tie-break win over the genuinely closer match — `HH: 245` matched an
   unrelated `LL: 120` (distance 0.017) instead of its true partner `HH: 250` (distance 0.003).
   Fixed by ranking within the tight band by actual closeness instead of a flat score.
2. **Template-coincidence noise (Pair 4).** Two unrelated documents sharing a P&ID template
   coincidentally place *some* element near *some* other at nearly every position, which
   tier 3 picked up regardless of content (`"NOTE 26"` ↔ `"N3601"` at score 0.99). Neither a
   same-type gate nor a similarity floor fully separates this from genuine edits — some
   coincidental pairs (`26GT9175` ↔ `26GT9134`) score *higher* similarity than real edits.
   The negative-control warning was switched from the (inflatable) overall alignment rate to
   the exact-key rate specifically, which isn't fooled by position: 0.99 on the genuine
   revision vs. 0.24 on the negative control, a clean separation.
3. **Unnamed geometry force-pairing (Pair 3).** A removed drain-stub `LINE` (no `block_name`,
   so no identity beyond position) matched an unrelated added tie-in `LINE` at distance 0.16,
   while the genuinely moved named block (`VALVE_GATE`) was only 0.09 away — too close a margin
   for one threshold to separate. Named block references now get a generous move tolerance;
   bare primitives get a much tighter one, since they have no identity beyond position to fall
   back on.

**Known, accepted limitation:** single-character instrument flags (`U`/`C`/`P`/`S`/`D`) and
bare digit/letter grid cells fall below `ALIGNMENT_MIN_EMBED_TEXT_LEN` and are never matched
across revisions — they show up as symmetric added/removed noise (119/119 on Pair 1) rather
than risking a confident-looking wrong pairing on content with no real signal. This is a
deliberate precision-over-recall tradeoff, not a bug; see `IMPLEMENTATION_PLAN.md` §14 and the
Phase 13 failure table.

## Markup overlay findings from real data (Phase 9)

Rendering the delta overlay onto Pair 3's DXF surfaced one non-obvious registration detail,
verified by rasterizing the output and sampling actual pixel colors rather than eyeballing it:
`ezdxf`'s `Frontend.draw_layout()` auto-fits and re-pads the matplotlib axes during drawing (a
requested `xlim` of `(10, 150)` came out as `(3, 157)` after drawing), so placing overlay boxes
by normalized fractional coordinates against the *requested* extents would have been measurably
off. The fix — recover true DXF model-space coordinates (`canonical_bbox + min_x/min_y`, reversing
`src/ingest/dwg.py`'s own normalization offset) and add matplotlib patches directly in *data*
coordinates — sidesteps the padding entirely, since matplotlib places data-coordinate patches
correctly regardless of what the final axes limits end up being. Confirmed correct via pixel
sampling: the removed drain-stub `LINE` (a degenerate zero-width bbox) renders as red pixels
`(208, 59, 59)` immediately straddling its native cyan `PIPING`-layer line, exactly where the
recovered coordinates place it.

Also worth recording so a future reader isn't confused by the rendered output: Pair 3's
`EQUIPMENT` layer (the "26-KA-901 GAS LIFT SKID" outline box) is authored with DXF color index 1,
which is red — the same color this project's convention uses for "removed". The two are
unrelated (confirmed by pixel-sampling all four edges of the outline as uniform red in *both*
A's and B's renderings, i.e. present unchanged on both sides, not a delta), but it's a
coincidental clash worth knowing about before assuming every red line in a Pair 3 render is a
flagged deletion.

## Eval harness findings from real data (Phase 10)

Running `eval/metrics.py`'s ground-truth matcher against all four labeled pairs (not just
hand-built fixtures) surfaced two more real issues, both fixed here:

1. **Two stale ground-truth entries in Pair 3.** `GT3-MOVE` listed the moved valve's *label*
   text (`26BL9075`) as the changed element, but a position-only text move with unchanged
   content correctly aligns as unchanged (by design, see §"Delta engine findings" above) — the
   engine reports the *geometry INSERT itself* as modified (text = its block name,
   `VALVE_GATE`), which is what the entry now says. `GT3-ADD-VALVE` listed `element_type:
   geometry` for the new valve, but the engine correctly classifies the labelled valve as
   `element_type: valve` via tier-1 exact-key matching (its own rationale text already said
   "counted as the labelled valve" — the type field just hadn't been updated to match). Both
   were authoring bugs from Phase 2/4, caught for the first time by an eval harness that
   actually cross-checks ground truth text/type against real engine output field-by-field
   instead of only checking element counts.

2. **Pair 2's raw precision is very low (0.005) even after excluding the known single-character
   noise category (§ above) — and this is a different, new kind of noise, not a bug.**
   Investigated by sampling the actual false-positive text: of 861 raw false positives, 656 are
   ≥2 characters (so the existing noise filter doesn't catch them), with a median length of 5
   characters and real examples like `'ennnnNenAI OOS'`, `'aw,'`, `'Wn'`, `'RDS'` — tesseract
   OCR misreads on a dense, small-font P&ID, not genuine content differences. This is
   qualitatively different from Pair 1's single-character flag noise (a length-based
   near-nothing signal) — OCR noise is *word-shaped garbage*, so a length threshold calibrated
   for the native-PDF case doesn't and shouldn't catch it. No fix attempted: building a
   garbage-text classifier (dictionary lookup, vowel-ratio heuristic, or similar) is a
   meaningfully different feature than what `ALIGNMENT_MIN_EMBED_TEXT_LEN` was calibrated for,
   and is exactly the kind of OCR-on-dense-content failure mode the plan's failure table is
   meant to hold, not something to quietly engineer away under eval-phase time pressure.

**Retrieval-quality note carried over from Phase 8, now measured, not just anecdotal:** the QA
dataset (`eval/datasets/qa_pair1.json`) deliberately includes `QA-11` ("What is the HH trip
limit for PIT-9062?"), the exact known-hard case documented in `src/chat/index.py`. The final
live run measured `recall@k = 0.36`, `MRR = 0.36` over the 11 answerable questions, with
`over_refusal_rate = 54.5%` (6 of 11 answerable questions incorrectly refused). Investigating
individual cases (not just trusting the aggregate) surfaced two distinct, differently-actionable
failure modes, both real:

- **Most are genuine retrieval misses**: the expected answer text never reaches the reranked
  top-k handed to the LLM. Same root cause as the documented `src/chat/index.py` limitation —
  short, self-contained note/value text competing against more prominent nearby content in the
  cross-encoder rerank. `QA-11` reproduces this every run (verified across all live runs).
- **At least one is a pure generation failure despite correct retrieval**, confirmed by direct
  inspection: for "What is note 22 about the design pressure?", the reranked context handed to
  the LLM was verified (by calling `hybrid_search`/`rerank_chunks` directly and printing the
  result) to contain `'22. DESIGN PRESSURE IN EXTERNAL SYSTEM DOWNSTREAM COMPRESSOR 257 BARG.'`
  as the #1 and #2 ranked passages — yet in that run the model still replied `NOT_GROUNDED: There
  is no note 22 in the provided context.` The correct answer was directly in front of it. Chat
  completions aren't fully deterministic across runs (observed: which specific questions get
  incorrectly refused varies run-to-run, e.g. this exact question answered correctly in a later
  run), so this is reported as a confirmed *failure mode* — the model can decline even when
  correctly-retrieved grounding is right there in context — rather than a claim that any one
  specific question always fails this way. A free-tier-model instruction-following limitation,
  not a retrieval or prompt-construction bug — no fix attempted (see below on scope).

**A related false alarm, resolved by checking the actual cited element rather than trusting the
recall@k number alone**: `QA-01`/`QA-02` (PSV-9066A/B set pressure) both scored `recall@k = 0.0`
against the dataset's single expected citation text (`"SP = 260 bar (g)"`), yet the live chat
answered both correctly and cited a *different*, equally valid source — note 37
(`pair1_B:pdf_native:00824`, verified: `'37. PSV 9066A/B SET PRESSURE REVISED TO 260 BAR(G).'`).
The retrieval eval is strictly correct by its own narrow definition (that specific text wasn't
in the top-k), but the QA dataset's `expected_citation_texts` only anticipated one valid
grounding source per question — a real limitation of the eval design, not the system under test.
Left as-is rather than broadening the expected-citations list after seeing the result, which
would be a step toward eval-gaming even with a legitimate justification; noted here instead so
`recall@k = 0.36` is read with the right caveat: retrieval quality is somewhat understated,
not overstated, by this number.

**A genuine judge-miscalibration bug, caught by the "validate the judge" step, fixed, and
re-validated.** The first live run scored `avg_correctness = 5.0` and `avg_groundedness = 5.0`
across all 15 items — including all six false-refusal cases above. The judge/human agreement
check (comparing against `eval/datasets/human_labels_pair1.json`'s hand-scored held-out subset)
caught this immediately: 60% exact agreement, mean absolute difference 1.6 on correctness — the
two held-out false-refusal items (QA-09, QA-11) were hand-scored `correctness=1` (zero
information delivered) but judged `5`. Root cause: the original judge prompt's "if the reference
summary says unanswerable and the assistant declined, score 5" rule was being over-applied by the
judge model to *any* decline, not conditioned on whether the question was actually answerable.
Fixed by rewriting `eval/judge.py`'s system prompt as an explicit four-branch decision procedure
keyed off the `answerable` ground-truth field (passed to the judge as data, not left to its own
inference) — branch 3 states plainly that declining an answerable question is CORRECTNESS=1
regardless of how reasonable the refusal sounds. Re-run after the fix: `avg_correctness` dropped
to `3.4` (now correctly penalizing the six false refusals) and `avg_groundedness` to `4.6`, and
**judge/human agreement on the held-out set reached 100% exact agreement on both dimensions**
(mean absolute difference `0.0`). This is exactly the scenario the brief's "validate the judge"
requirement exists to catch: an unvalidated judge here would have shipped a scorecard claiming
perfect groundedness while silently missing over half of answerable questions.

**A second, independent bug found and fixed along the way: rate-limit wait parsing.** Re-running
the full eval after the judge fix stalled for close to an hour on Pair 2's scanned-PDF ingest.
Root cause, confirmed via `src/chat/llm.py`'s own trace logs: Groq's daily-quota 429 errors use
a compound `"37m25.536s"` format (minutes *and* seconds), which the existing
`parse_suggested_wait` regex silently failed to parse (`\b` never matches between the `m` unit
and the immediately-following digit) — it fell back to a hardcoded 15s guess, which then retried
~8 times and burned the *entire* configured wait budget (120s) before giving up, on every single
vision-fallback region that needed it. Fixed in two parts: (1) the regex now parses the compound
format directly, and (2) `_with_rate_limit_wait` fails immediately, without sleeping at all, when
the provider's stated wait already exceeds the remaining budget — since no amount of waiting
within budget can succeed, waiting anyway is pure cost with zero chance of success. Both fixes
are covered by new unit tests (`tests/test_llm.py`) using the exact error strings observed live.
This is unrelated to the take-home's own subject matter but a real, measured operational finding
about running against a free-tier provider under sustained load — worth keeping for the
interview-prep "what would you do differently at scale" discussion (§14).

**Scope note:** the retrieval-side findings (short-fragment rerank loss; the generation-side
false-refusal-despite-correct-retrieval case) are left undressed here, same reasoning as
Phase 8 — real, measured, honestly reported, and exactly the kind of material the Phase 13
failure table and interview prep are meant to hold, not something to patch under eval-phase time
pressure. The judge-prompt and rate-limit fixes were different in kind: both corrected bugs in
code this project owns (the validator's own logic; this project's own retry/backoff behavior),
not a capability limit of the free-tier LLM being evaluated, so both were made directly rather
than deferred.

## Delta engine findings from real data (Phase 12): cross-page matching

Pair 5 (above) surfaced a fourth real alignment bug — a serious one, and unlike the Phase 5
findings, this one was a plain correctness bug rather than a calibration/threshold judgment
call. **Every one of Pairs 1-4 is single-page**, so nothing before Pair 5 ever built a document
with two elements sharing identical *normalized* (page-relative) bbox coordinates while sitting
on different pages — a completely ordinary situation for any real multi-sheet P&ID set (the same
tag/layout convention repeated per sheet).

**The bug.** All three alignment tiers (`src/delta/align.py`) computed candidate bbox distance
using only `BBox.normalized` (fractional 0-1 position *within* an element's own page) — `page`
itself was never part of the matching key or the candidate filter. Two same-text elements on
different pages, sitting at the same fractional position, are therefore indistinguishable by
distance (both score exactly 0.0 against a same-position candidate on either page), so the
greedy assignment's deterministic tie-break (`a.id`, `b.id` string order) — not actual page
correspondence — decided which page's element "won" a match.

**Confirmed with a minimal repro before touching Pair 5's full scale:** doc A has the same tag
`DUPTAG` on page 0 and page 1; doc B keeps only page 1's copy (page 0's was genuinely deleted).
Correct behavior: report page 0's `DUPTAG` as removed, leave page 1's matched/unchanged. Actual
(pre-fix) behavior: the engine matched page *0*'s `DUPTAG` to B's surviving copy and reported
page *1*'s as removed — the exact opposite of reality. In a real multi-page review, this is a
report that confidently, wrongly tells a reviewer *the wrong sheet* changed, and the Phase 9
markup overlay would have highlighted the wrong page's box as "removed."

**Fix:** page is now part of the candidate-generation key/filter in all three tiers —
`exact_key_match` groups by `(element_key, bbox.page)` instead of `element_key` alone,
`geometry_match`'s coarse key gains `bbox.page` as a fourth component, and
`embedding_proximity_match` gained an explicit `a.bbox.page != b.bbox.page: continue` gate
alongside its existing same-type gate. Zero regressions across the full pre-existing test suite
(463 tests) — expected, since every prior fixture defaults to `page=0` on both sides, making the
new page-equality constraint a no-op for all of them. Four new regression tests
(`TestPageScoping` in `tests/test_delta_align.py`) pin down the exact repro above for all three
tiers, plus a same-non-zero-page sanity check.

**A second-order benefit, not just a correctness fix: this also fixes a latent performance
cliff.** Before the fix, tier 3's candidate generation was effectively O(n·m) over the *entire*
flattened element pool regardless of page — an N-page duplicate set would have been O(n·m·N²)
(all pages compared against all pages). Scoping to same-page pairs makes it N independent
same-size sub-problems, O(n·m·N) — confirmed by Pair 5's own measurement above showing
close-to-linear (not quadratic) time growth from 2 to 10 pages. A genuine 500-sheet set would
have made the pre-fix quadratic behavior actively unusable, not just wrong.

## Real-world hardening pass (post-Phase-12): four genuine bugs found by distrusting the test suite

Every phase up to this point was checkpointed on "tests pass, live-verified once." Before Phase
13, we deliberately went back and asked a harder question: not "does it pass," but "does it
actually work" — by reading the *committed* `eval/scorecard.json` line by line against its own
questions and ground truth, instead of trusting the aggregate numbers. That distrust turned up
two significant, previously-undetected bugs (the delta noise and the chat retrieval mismatch
below), which in turn led to two more once fixing them changed what was actually being exercised.
None of these were caught by 494 passing unit tests, because the tests were unit tests — this
required reading real model output against real ground truth.

### Bug 1: 235 of 237 Pair 1 delta "changes" were single-character noise, not edits

The committed scorecard's delta P/R/F1 looked catastrophic on inspection — Pair 1 aggregate
precision of **0.0207** (tp=5, fp=237, fn=1) despite Pair 1 being the best-understood, most
heavily-tested sample pair in the whole project. Recall (0.83) was fine; something was burying 6
real edits under 237 false positives.

Direct inspection of `compute_delta()`'s raw output (not just the P/R/F1 summary) showed the false
positives were overwhelmingly `('removed', 'text_block')` / `('added', 'text_block')` pairs whose
text was a single character: `'U'`, `'C'`, `'P'`, `'*'`, `'S'`, `'D'`... These are a real,
deliberately classified category (`classify_block_lines`'s `kind="flag"` — valve position letters
scattered around the P&ID), not extraction garbage, and the vast majority are genuinely unchanged
between revision A and B.

**Root cause** (`src/delta/align.py::embedding_proximity_match`): candidate generation itself was
filtered by `alignment_min_embed_text_len` (2) before the tight/loose distance bands were ever
evaluated — so a single-character element could never reach the *tight* band, even though the
tight band is **position-decisive and never uses the embedding at all** (`dist <= tight` skips
straight to a score; only the *loose* band needs semantic similarity). Two identical, identically
positioned `'U'` flags in revision A and B were therefore structurally unmatchable by any tier:
too short for tier 1 (no stable key — many identical single letters exist), filtered out of tier
3's candidate pool entirely regardless of position.

**Fix:** candidate generation is no longer filtered by `min_len` — only *embedding computation* is
(a dict keyed by element id, populated only for elements meeting `min_len`, looked up on the loose
band; the tight band never touches it). Verified directly against real Pair 1 data:

| | Before | After |
|---|---|---|
| Pair 1 total deltas | 242 | **8** |
| Pair 1 alignment_rate | 0.8558 | **0.9976** |
| Pair 1 precision / recall / F1 | 0.0207 / 0.8333 / 0.0403 | **0.625 / 0.8333 / 0.7143** |

The remaining 8 deltas map onto the pair's actual intentional edits (PSV setpoints, PIT HH limit,
note add/remove, flow rate). Pair 3 (DXF) was already fine (0.71 precision, unaffected — its
elements don't include single-character flags). **Pair 2 (scanned/OCR) is NOT fixed by this** —
its false positives (819, barely moved from 861) are a different, OCR-text-variance root cause,
not the min_len bug; see "what's still honestly broken" below.

Two existing unit tests encoded the *old, incorrect* behavior as correct
(`test_short_text_below_min_length_is_never_embedded` asserted two identical same-position `'*'`
elements do *not* match) — rewritten into two tests that separate the two real properties: short
text still matches via the embedding-free tight band at the same position, but still correctly
fails to match via the loose band (where no embedding exists to compare).

### Bug 2: chat retrieval had no way to answer "what does note N say?"

Reading the scorecard's actual answer text against its own questions (not just the aggregate
correctness/groundedness averages) showed a stark pattern: of 11 answerable QA-pair1 questions,
**6 got a false `NOT_GROUNDED` refusal** — and 5 of those 6 were exactly the phrasing "what does
note N say" / "what is note N about."

**Root cause** (`src/chat/index.py::exact_lookup`): the deterministic exact-match path extracts
tag-shaped tokens from the query (`_extract_tag_tokens`, requires a digit and length ≥ 3) and
matches them against an element's *entire* text. A bare note number like `"37"` is filtered out
for being too short, and even if it weren't, it would never equal a whole note's full-sentence
text via substring match. So every "note N" question fell through to pure vector search alone —
which is exactly why some notes happened to retrieve (note 8, note 22) and others didn't (note
37, 16, 33, 19, 11): it was never a deterministic guarantee, just semantic-similarity luck.

**Fix:** `note_number` (already extracted at ingest time by `classify_block_lines`, previously
only used internally by the delta engine's tier-1 key) is now also stored in chromadb's indexed
metadata, and a new `_extract_note_numbers` pattern (`\bnote\s*(?:number\s*)?#?\s*(\d{1,3})\b`)
lets `exact_lookup` match a query's note-number reference against it directly — deterministic, no
LLM, same "short high-signal identifier deserves exact matching" philosophy as the existing tag
lookup.

### Bug 3: a fixed exact match could still be reranked out of the LLM's context

Fixing bug 2 alone was not sufficient — direct testing after the fix showed `exact_lookup("What
is note 16 about?")` correctly found the right passage, yet the live scorecard still refused.
`answer_question` (`src/chat/answer.py`) reranked *everything* `hybrid_search` returned uniformly,
including exact-source hits, down to `rerank_top_k` (5) slots — so a deterministic, guaranteed-
relevant exact match could still lose to the cross-encoder's approximate judgment on a crowded
query. This is the same crowding-out mechanism already documented in `chat/index.py`'s own
docstring for the HH:250/PIT-9062 case, just newly reachable for exact hits too, which by
construction should never lose that competition.

**Fix:** exact-source chunks now always survive into the final context; only the remaining budget
(`rerank_top_k - len(exact)`) is spent reranking the vector-sourced candidates.

### Bug 4: a correct passage in context wasn't enough — the model still refused

With bugs 2 and 3 both fixed, three of the five "note N" false refusals resolved on the next live
run, but two (note 16, note 33) still refused. Reproducing the *exact* context that would be sent
to the LLM and calling it directly (bypassing retrieval/reranking entirely) confirmed the correct
passage — `"16. PRIMARY SEAL GAS IS TAKEN DOWNSTREAM..."` — was sitting right there as the first
context line, and the model *still* said the content wasn't provided, consistently across 3
attempts. Stripping the context down to just that one passage made the model answer correctly and
cleanly every time; the difference was the presence of a second, separate chunk that vector search
also (correctly, on its own terms) retrieves: a bare `"NOTE 16"` cross-reference fragment
(`kind="reference"`, a different element than the note's own definition — a literal callout
elsewhere on the drawing pointing at note 16, carrying no content of its own). The model appears
to read multiple fragmentary mentions of "NOTE 16" as evidence the information is scattered/
incomplete, rather than recognizing the first, fuller passage as the complete answer.

**Fix:** added one clarifying rule to `_SYSTEM_PROMPT` (`src/chat/answer.py`) explicitly telling
the model that a numbered passage like `"16. TEXT..."` is the note's full content, and a bare
`"NOTE 16"` fragment with nothing else is a cross-reference, not content, to be ignored when a
fuller passage with the same number is present. Verified directly: 3/3 consistent correct answers
against the exact real (noisy) context that previously refused 3/3, with no retrieval or indexing
changes at all.

### Measured before/after (real, live, Groq/llama-3.3-70b-versatile — not simulated)

Three complete live full-suite runs were captured while iterating (bugs 1–3 fixed, prompt fix
added after); a fourth full run to re-validate bug 4 in the complete suite hit Groq's free-tier
**daily** token quota (100k TPD, exhausted by the day's testing) and correctly fail-fast'ed rather
than hang — see `src/chat/llm.py`'s rate-limit handling. Bug 4's fix is verified directly (above,
3/3) rather than via a fourth full scorecard; the committed `eval/scorecard.json` reflects bugs
1–3 fixed (the last complete run) and predates bug 4's fix. A full re-run once the quota resets is
straightforward (`make eval`) but not required to trust the fix, given the isolated reproduction.

| Metric | Before | After (bugs 1–3) |
|---|---|---|
| Delta F1 (aggregate) | 0.0248 | 0.0327 |
| Delta precision (aggregate) | 0.0126 | 0.0167 |
| Chat avg correctness | 3.40 | 3.93 |
| Chat avg groundedness | 4.60 | 4.73 |
| Chat over-refusal rate | 0.5455 (6/11) | 0.3636 (4/11) |
| Retrieval recall@k / MRR | 0.3636 | 0.4545 |

The aggregate delta F1 barely moves despite Pair 1 individually going from 0.04 to 0.71 F1,
because Pair 2's still-unfixed OCR noise (819 false positives) dominates the aggregate sum — an
honest, visible illustration of why per-pair breakdowns matter more than a single blended number.

### Judge validation, expanded from n=5 to n=15

The original judge/human agreement check (Phase 10) hand-scored 5 of 15 QA-pair1 items — legally
a "held-out validation set," but a thin statistical basis for the claim "the judge is trustworthy"
(100% agreement at n=5 has enormous variance). Since the real live scorecard now contains the
actual model output for all 15 items (not just the 5 originally hand-checked), every item was
independently re-read against its expected answer/citations from `eval/datasets/qa_pair1.json`
and hand-scored the same way as the original 5 (`eval/datasets/human_labels_pair1.json`, all 15
`held_out: true` now). Result: **15/15 exact agreement on both correctness and groundedness, mean
absolute difference 0.0** — the same clean numbers as before, now on the full dataset rather than
a third of it. One item (QA-08, note 19's alarm) is flagged in its own hand-label notes as a
genuine borderline call (the model's answer addresses the literal question asked but omits a
related detail from the same note) rather than papered over as an unambiguous agreement.

### Other real gaps found and fixed

- **Dashboard DWG upload could never trigger DWG→DXF conversion.** `_UPLOAD_EXTENSION` saved every
  `format="dwg"` upload as `doc_a.dxf` regardless of what was actually uploaded, so
  `DwgAdapter.ingest()`'s `path.suffix.lower() == ".dwg"` check (which gates the ODA File
  Converter call) could never see a `.dwg` suffix through the dashboard, even with a real `.dwg`
  file and a working converter — it would always be hard-parsed as DXF and fail with a confusing
  low-level `ezdxf` error. Fixed (`src/dashboard/app.py::_upload_suffix`) by reading the suffix off
  the *client-supplied filename* (safe here specifically because it only selects between two
  hardcoded extensions, never becomes a path component — the existing "never trust a display
  string as a path" rule for pids is unaffected).
- **No dashboard upload size limit.** `UploadFile.file.read()` with no argument reads the entire
  body into memory in one shot before writing it to disk — unbounded for a server with no auth in
  front of it. Fixed with a configurable `DASHBOARD_MAX_UPLOAD_MB` (default 50) enforced via a
  bounded `.read(limit+1)` rather than reading the whole file to check its size after the fact.
- **The low-alignment-rate warning could go silent exactly when it's needed most.** It fires on
  low *exact-key match rate*, but only once both sides already produce ≥20 keyed elements — a
  document whose tag-numbering convention isn't recognized at all by `src/ingest/pdf_native.py`'s
  classifiers (openly documented there as grounded in one client's convention, not a general P&ID
  standard) produces very few keyed elements on *either* side, so the existing warning never gets
  enough signal to fire. Added a complementary `low_keyed_fraction_threshold` warning
  (`src/delta/engine.py`) that fires when the *fraction* of elements classified as any recognized
  tag/instrument/valve/line-number type is implausibly low, independent of match rate.
- Added a regression test confirming a genuinely corrupt (empty/garbage-bytes) upload with the
  *correct* declared format fails cleanly (400, not a crash) — the existing test only covered a
  wrong-format upload, a different failure mode.

### What's still honestly broken or unverified (not silently hidden)

- **DWG-binary support (as opposed to DXF) has never been exercised end-to-end.** No real `.dwg`
  file exists anywhere in this repo (Pair 3 is hand-authored DXF), and the ODA File Converter
  isn't installed in this development environment, so `convert_dwg_to_dxf`'s actual subprocess
  invocation has only ever been unit-tested against a mocked converter. The code path is
  reasonable (documented CLI args, output-file verification, a clear error when the tool is
  missing) but unverified against a real conversion — exactly the risk plan §13 anticipated and
  pre-authorized falling back on, not a hidden gap.
- **Classification is genuinely convention-specific**, not a general P&ID/engineering-drawing
  parser — a document using a different tag-numbering scheme than the two real sample P&IDs this
  was built against will mostly fall through to generic `text_block`, and while the new low-keyed-
  fraction warning now surfaces this at runtime, the underlying limitation (recognizing an
  arbitrary drafting standard) is out of scope for this engagement, not something this pass
  attempted to solve.
- **Pair 2 (scanned/OCR) delta precision** — see the dedicated deep-dive below. Root-caused
  precisely, one contributing cause fixed, the deeper one deliberately not patched without a
  larger calibration sample (explained below, not just asserted).

## Second hardening pass: precise root cause for Pair 2, and what's genuinely hard vs. fixable

The first hardening pass above found and fixed four real bugs but left Pair 2's delta precision
at 0.0049 (819 false positives) with only "OCR text variance" as an explanation — not good enough
given the stated bar ("if it's not near-perfect, there's no difference from manual review"). This
section is a genuine root-cause dig, not a restatement of the same hand-wave.

### Pair 2's 819 false positives: two distinct, now-precise mechanisms

**Mechanism A — OCR corruption breaks classification, and the delta engine's own type-safety gate
then blocks the rescue.** Sampling actual removed/added pairs at near-identical positions found
concrete corruption, not vague "noise":

| Removed (native A) | "Added" (OCR'd B) | What happened |
|---|---|---|
| `LL :50` (setpoint) | `LLC5O` (text_block) | tesseract read `:` as `C` and `0` as `O` |
| `1"-AI-63-9006-AS20-00` (line_number) | `1"-Al-63-9006-AS20-00` (text_block) | tesseract read capital `I` as lowercase `l` |

Both pairs sit at essentially the same position (bbox center distance a few tenths of a point on a
~700-point-wide page) — a human would call these the same element, unedited. But
`classify_block_lines`'s regexes are exact-anchored (`^...$`); `LLC5O` doesn't match
`_SETPOINT_LIMIT_RE` and `1"-Al-...` doesn't match `_LINE_NUMBER_RE` (uppercase-only service
code), so both get reclassified as generic `text_block`. The delta engine's tier-3 same-type gate
(`src/delta/align.py`, added earlier specifically to stop coincidental cross-type matches on
Pair 4's unrelated documents) then refuses to match a `setpoint`/`line_number` against a
`text_block` counterpart *even at near-zero distance*, so each pair becomes a spurious
remove-from-A + add-to-B instead of one correctly-matched (or correctly-flagged-modified) element.

**Mechanism B — the vision-LLM quality rescue barely reaches the problem.** tesseract flagged 408
of Pair 2 B's 916 elements (44.5%) as low-confidence, but `vision_fallback_max_regions` capped the
LLM re-read at 12 — worst-12-first, leaving the other 396 (97% of flagged regions) with raw,
sometimes-corrupted tesseract text exactly like the two examples above. This cap exists for a real
reason (each region is a paid LLM call against a rate-limited budget), but 12 was never actually
calibrated against how many low-confidence regions a real, dense P&ID produces — it was just
"small enough to not blow up an eval run."

### What was fixed, and the real (modest) measured impact

**Fixed (mechanism B): `vision_fallback_max_regions` raised from 12 to 40** (`src/config.py`) —
attacks the OCR-quality root directly (feeds cleaner text into the same classifier) rather than
patching around already-bad text downstream. A real, deliberately partial improvement (12→40 is
~3x the coverage, not exhaustive — 40 still isn't 408), bounded on purpose: every additional
region is a real LLM call against the same free-tier daily quota this investigation's own testing
fully exhausted (both the text and vision models, confirmed below), so uncapped coverage would
trade OCR quality for making the ingest step itself unreliable under rate limits. **Not
re-verified against a live Pair 2 run** — the vision model's quota was gone by the time this was
implemented; sound by construction, not re-measured end to end.

**Fixed (mechanism A): OCR-tolerant reclassification** (`src/ingest/pdf_native.py`,
`classify_block_lines(..., ocr_tolerant=True)`, wired only into the scanned-PDF adapter). The
first design attempt was a blind whole-string character substitution (map every `O`→`0`, every
`1`→`I`, etc. across the entire string) and it was *wrong* — caught before shipping by testing it
against the real motivating example: blindly substituting `1`→`I` also corrupted the line number's
own leading size digit (`1"` → `I"`), breaking the very match it was meant to fix, because a blind
substitution can't tell a digit-context position from a letter-context one in a string that mixes
both (`1"-AI-63-9006-AS20-00` has digits *and* letters). The shipped version is position-aware
instead: parallel "tolerant" regexes for the letter-only fields specifically (service code, tag
prefix, instrument function code), built by widening just those `[A-Z]` character classes to admit
tesseract's known digit-lookalikes, while every digit field stays strict `\d`. For the one pattern
that extracts a semantic value from the match (`instrument_loop`'s function code, checked against
`INSTRUMENT_FUNCTION_CODES`), the captured text is normalized back to a clean letter code *only*
after the tolerant regex confirms the structure and *before* the vocabulary check — a corrupted
string can never equal a known code, so without this the vocabulary gate would silently reject
every real match. An element's stored `text` is never rewritten either way, only its classified
`type`/`attributes` — matching what the rest of this file already promises (never trust a
normalized guess as ground truth about what the drawing says).

Digit-field corruption (the `LL:50`→`LLC5O` example) is deliberately still not covered: the same
tesseract pass that mis-read `0` as `O` also dropped the `:` separator entirely, and character
substitution alone can't recover a deleted character — there was no working example to justify
guessing at a punctuation-recovery rule, so digit fields stay strict rather than being loosened
without evidence.

**Measured directly against real Pair 2 data** (no LLM involved, fully deterministic): 3 of Pair 2
B's 916 elements were reclassified out of the box —

| Original (native A) | OCR'd (scanned B), now reclassified | Reclassified type |
|---|---|---|
| `1"-AI-63-9006-AS20-00` | `1"-Al-63-9006-AS20-00` | `line_number` |
| `1"-AI-63-9007-AS20-00` | `1"-Al-63-9007-AS20-00` | `line_number` |
| `40GT9309` | `40G6T9309` | `valve` |

Traced one of these (`40G6T9309`) through the full pipeline: it now sits at normalized bbox
distance 0.00041 from `40GT9309` in revision A and correctly resolves to a single `modified valve`
delta (`40GT9309` → `40G6T9309`, tier `embedding_proximity`) instead of a spurious
remove-from-A + add-to-B pair. A batch safety check against 13 plausible generic drawing words
(`VENDOR`, `GAS`, `COMPRESSOR`, ...) confirmed none were falsely reclassified.

Pair 2's aggregate P/R/F1, recomputed for real: `fp` 819 → **816**, `tp` unchanged at 4, precision
still rounds to 0.0049. **Honest conclusion: correct, safe, verified improvement — and a small
one.** This specific, narrowly-scoped fix (letter-position character-confusion tolerance) resolves
only the subset of Pair 2's noise that is exactly that failure mode. The other ~816 false
positives are dominated by different mechanisms this pass did not individually root-cause for
every case — digit-field corruption combined with lost punctuation (confirmed present, not fixed),
and very likely broader tesseract-vs-pymupdf line/block segmentation differences that were not
exhaustively catalogued. Pair 2 is measurably, verifiably *less wrong* than before this
investigation, and still far from the "near-perfect" bar for scanned-source documents specifically
— stated plainly rather than rounded up.

### Pair 1 and Pair 3's remaining "false positives" are not matching bugs

Worth stating plainly since it changes the read on those two pairs' precision numbers:

- **Pair 1's 2 non-ground-truth deltas** (`added note 'NOTE 29'`, and the ground-truth flow-rate
  edit's `old_text` coming out as `'19057 NOTE 29'` instead of `'19057'`) trace to one cause:
  pymupdf grouped the flow-rate value and an unrelated nearby `NOTE 29` reference into the same
  extracted text *line* in revision A but not in revision B (a rendering-level line-grouping
  artifact, not a bug in this project's own code — `classify_block_lines` never sees the two
  pieces as separable once pymupdf has already joined their spans). This is real and worth knowing
  about — text-block segmentation boundaries are not perfectly stable across independently-
  rendered PDF revisions in general — but it is not a delta-engine defect: the flow-rate edit
  *was* correctly detected as `modified`, just with extra baggage text that then fails the eval
  harness's own fuzzy-match threshold against a strict ground-truth string (confirmed directly:
  `SequenceMatcher('19057', '19057 NOTE 29').ratio() == 0.556`, below the 0.85 bar) and reads as
  a false-positive-plus-false-negative pair in the scored metric without being a wrong answer in
  substance.
- **Pair 3's 2 non-ground-truth deltas** are both structural, not wrong: the "added valve" edit
  produces two canonical elements by design (`geometry` for the INSERT block reference + `valve`
  for its ATTRIB tag — DWG's richer structure "earning its keep," per plan §3), but the ground
  truth only itemizes the tag; and the "removed line" edit sits among several anonymous, textless
  `geometry` primitives on the same layer, where position is the *only* available signal for two
  candidates that are genuinely, information-theoretically indistinguishable beyond it. Neither is
  a matching-algorithm defect — the DWG format faithfully has fewer human-readable "keys" than
  P&ID tags do, and any content that generates truly zero distinguishing signal cannot be
  disambiguated with more/better code.

Net effect: Pair 1 and Pair 3's real precision (once these are understood, not just counted) is
much closer to "correct, with cosmetic scoring artifacts" than the raw 0.625/0.714 numbers alone
suggest. Pair 2 is the one pair with a genuine, still-partially-open accuracy problem.

### Chat: the three remaining "note N" refusals are very likely already fixed, unverified only because of quota

Deterministic retrieval inspection (no LLM call needed) of the three still-refusing questions
(note 16, note 33, note 11 — QA-05/07/09) shows an *identical* pattern: the correct definition is
retrieved as an exact match in the top 2 slots, alongside several bare `"NOTE N"` cross-reference
fragments from vector search — exactly the shape that bug 4 above (system-prompt fix) was verified
against directly and fixed 3/3. Note 33 in particular has *more* bare-reference fragments (4–6) in
its retrieved set than note 16 did, so it is, if anything, a stronger instance of the same pattern,
not a different one.

The fourth refusal, `PIT-9062`'s HH trip limit (QA-11), is structurally different and confirmed
still genuinely hard: retrieval finds the `PIT-9062` tag correctly but **no value-bearing fragment
(`"HH: 250"` or similar) appears anywhere in the top 8 hybrid-search results at all** — this isn't
a prompt-following problem, the relevant text simply isn't in the candidate pool being reranked.
This matches the already-documented, deliberately-not-fixed limitation in `chat/index.py`'s own
docstring (no textual/positional link between a bare value annotation and the instrument tag it
belongs to).

Net read: the real current chat over-refusal rate, with all four committed fixes in place, is very
likely close to 1/11 (only the PIT-9062 case), not the 4/11 measured in the last complete run —
but this is inference from strong, structurally-matching indirect evidence, not a re-confirmed
live measurement, and is reported as such rather than rounded up to a claim.

### Both Groq daily quotas (text and vision models) are now fully exhausted

Confirmed directly: `llama-3.3-70b-versatile` at 100000/100000 TPD and `qwen/qwen3.6-27b` at
200000/200000 TPD, both reporting the standard "Please try again in `<tomorrow>`" 429 body. This
is why the vision_fallback_max_regions fix above and the note 33/note 11 inference above are
reported as reasoned-but-unverified rather than measured: today's investigation itself consumed
the remaining budget for both models. Nothing here was left untested by choice where a live test
was actually possible.
