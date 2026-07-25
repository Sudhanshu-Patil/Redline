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

## Pair 5 — stress set (pending Phase 12)

Duplicated sheet set with scattered edits, for the scaling analysis.

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
