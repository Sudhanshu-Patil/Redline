# Failure modes — candid, not curated

Every row here is a real, measured finding, not a hypothetical. "Fixed" means verified against
real data with a before/after number, not "should work now." Full mechanism and measurement
detail for anything marked with a section link lives in
[`data/samples/PROVENANCE.md`](data/samples/PROVENANCE.md) — this file is the scannable index,
that one is the primary source.

| # | Failure mode | Severity | Status | Where |
|---|---|---|---|---|
| 1 | Scanned-PDF delta precision is genuinely weak | High | Partially fixed, root-caused | [§1](#1-scanned-pdf-delta-precision) |
| 2 | A bare value with no textual link to its tag can be missed by retrieval | Medium | Open, deliberately not fixed | [§2](#2-the-hh250pit-9062-retrieval-gap) |
| 3 | Classification is grounded in one client's tag convention | Medium | Open, now surfaced at runtime | [§3](#3-classification-is-convention-specific) |
| 4 | DWG-binary conversion has never run against a real `.dwg` file | Medium | Untested, not hidden | [§4](#4-dwg-binary-conversion-is-unverified) |
| 5 | Near-identical notes can be matched by the wrong instance | Low | Open, understood | [§5](#5-embedding-only-near-duplicate-matches) |
| 6 | pymupdf can merge unrelated text into one element across revisions | Low | Understood, not a delta-engine bug | [§6](#6-pdf-line-grouping-artifacts) |
| 7 | Dashboard sessions don't survive a restart; no auth | Low | Documented scope boundary | [§7](#7-dashboard-operational-scope) |

## 1. Scanned-PDF delta precision

**The number:** aggregate delta precision is 0.0167 across the 4 labeled pairs (`eval/scorecard.json`)
— almost entirely driven by Pair 2 (scanned), which alone accounts for 819 of the 824 false
positives in the aggregate (Pair 1: 3, Pair 3: 2). Pair 1 (native PDF) and Pair 3 (DXF) are not
part of this problem — see §6 below for why their own scores also aren't as bad as a raw number
suggests.

**Root cause, precisely, not "OCR is noisy":** tesseract's well-known single-character confusions
(`0`/`O`, `1`/`I`/`l`, `5`/`S`, `8`/`B`, `2`/`Z`, `6`/`G`) break `classify_block_lines`'s
exact-anchored regexes often enough to reclassify a real `valve`/`line_number`/`tag` as a generic
`text_block`. That misclassification then hits the delta engine's same-type matching gate — added
specifically to reject a *different*, real problem (Pair 4's cross-document layout coincidences,
see §5) — which then refuses to match the corrupted element against its correctly-classified
counterpart on the other revision, even at near-zero bbox distance. It becomes a spurious
remove+add instead of one correctly matched element.

**What was fixed, with a number:** `classify_block_lines` gained an `ocr_tolerant` retry mode
(scanned-PDF adapter only — native PDF and DWG extraction is exact, so the same tolerance there
would be pure downside) using *position-aware* tolerant regexes — only the letter-only fields
widen, digit fields stay strict. Verified directly: `40GT9309` was OCR'd as `40G6T9309`, correctly
reclassified as `valve`, and now resolves to one correct `modified` delta instead of a spurious
remove+add pair. Also raised `vision_fallback_max_regions` 12 → 40 (tesseract flagged 408 of 916
elements low-confidence on Pair 2's B side; the old cap covered ~3% of them with the LLM re-read
built specifically to fix bad OCR text). Net effect measured directly against real data: 3
elements rescued, Pair 2's false positives 819 → 816 on the specific bug class this targeted —
a standalone, targeted verification (traced one rescued element end to end:
`40GT9309`, OCR'd as `40G6T9309`, now correctly resolves to one `modified` delta instead of a
spurious remove+add). Not yet folded into a fresh full `eval/scorecard.json` run, so the
committed aggregate above still reflects the pre-fix 819 — the fix is real and verified, the
headline number just hasn't been re-published yet.

**What's still open, honestly:** the OCR-tolerant fix only covers *character-confusion*
corruption. Two other mechanisms are confirmed present and *not* fixed:
- Digit-field corruption combined with lost punctuation (`LL :50` OCR'd as `LLC5O` — the `:`
  didn't just get misread, it disappeared; character substitution alone can't recover a deleted
  character, and there was no second working example to calibrate a safe fix from).
- tesseract's line/block segmentation genuinely differs from pymupdf's, so the same
  multi-line note-accumulation logic produces truncated notes on the OCR side — e.g. note 19's
  first ~100 characters (including its own note number) never reached the element that carries
  the rest of the note, because tesseract split it into a different block boundary than pymupdf
  would have. This is a layout-analysis gap, not a character-recognition one, and a stronger OCR
  *engine* wouldn't automatically fix it — that specifically needs better paragraph/block
  grouping, a distinct capability from character accuracy.

A same-day live test comparing tesseract against two free-tier vision-model alternatives (Google
Gemini, then OpenRouter's `google/gemma-4-31b-it:free`) produced real but inconclusive data: one
exact-correct read on a crop tesseract had badly mangled, but both free tiers hit hard quota
walls before enough specimens completed to draw a real conclusion. Genuinely open, not resolved.

## 2. The HH:250/PIT-9062 retrieval gap

A bare value annotation like `HH: 250` sits ~0.05 normalized-bbox-units from the instrument tag
it belongs to (`PIT-9062`) but carries no textual or structural link to it in the extracted data
— they're two unrelated `Element`s that happen to be near each other. Asking "what is the HH trip
limit for PIT-9062" retrieves the tag correctly, but the value-bearing fragment can lose the
cross-encoder rerank to more semantically-titled competitors, or (confirmed via direct inspection)
simply never make it into the top-k candidate pool at all.

**Deliberately not fixed:** spatial-context enrichment at index time was considered and rejected —
on a dense layout, a different tag can sit *closer* to a stray value than the tag it actually
belongs to (measured: `TIT-9064` sits ~0.006 units closer to a nearby value than the tag that
value actually describes), so a positional heuristic risks confidently attaching a value to the
wrong instrument, which is worse than an honest refusal. See `src/chat/index.py`'s own docstring
for the full reasoning.

## 3. Classification is convention-specific

`src/ingest/pdf_native.py`'s regex classifiers (tag/valve/instrument-loop/line-number patterns)
are grounded in the two real sample P&IDs this project was built against — openly documented in
that module's own docstring, not discovered after the fact. A drawing using a different
tag-numbering convention will mostly fall through to generic `text_block`.

**Partial mitigation:** the delta engine's existing low-alignment warning needs both sides to
already produce plenty of keyed elements before it can compare their match rate — so it stayed
silent in exactly the case a user most needs to be told about. A complementary
`low_keyed_fraction_threshold` warning now fires when the *fraction* of recognized tag-like
elements is implausibly low, independent of match rate (`src/delta/engine.py`). The underlying
limitation — recognizing an arbitrary drafting standard — is out of scope; the warning just makes
sure nobody trusts a bad delta silently.

## 4. DWG-binary conversion is unverified

`src/ingest/dwg.py::convert_dwg_to_dxf` shells out to the free ODA File Converter for genuine
binary `.dwg` input. The code path is reasonable (documented CLI invocation, output-file
verification, a clear error when the tool is missing) and unit-tested against a mocked converter
— but no real `.dwg` file exists anywhere in this repo (the DWG sample pair is hand-authored DXF)
and the ODA tool isn't installed in this development environment, so the actual subprocess call
has never run for real. This is the risk plan §13 explicitly anticipated and pre-authorized a
fallback for, not a gap introduced quietly.

## 5. Embedding-only near-duplicate matches

The honest, designed-for break case: two near-identical notes with no exact key are
disambiguated only by embedding similarity and bbox proximity. When both signals are close for
two different candidates, the aligner's greedy assignment can pick the wrong one. This was the
anticipated hard case going in, checked directly in
`tests/test_delta_align.py::TestEmbeddingProximityMatch::test_global_greedy_prefers_exact_self_match_over_ambiguous_candidate`
(an unchanged element must claim its own perfect match first, so it can't be mistakenly stolen as
a nearby edited value's partner) and in the dedicated `TestNoteSplitIntoTwo` /
`TestMovedAndRenamedTag` classes in the same file — and it's the honest answer to "where does
your alignment break?"

## 6. PDF line-grouping artifacts

Two of Pair 1's non-ground-truth deltas trace to one cause: pymupdf grouped a table value
(`19057`, the flow-rate edit) and an unrelated adjacent `NOTE 29` reference into the same
extracted *line* in revision A but not revision B — a rendering-level span-grouping decision
upstream of anything this project's own code controls. The edit itself was still correctly
detected as `modified`; the merged text just fails the eval harness's strict fuzzy-match against
a clean ground-truth string (measured: `SequenceMatcher('19057', '19057 NOTE 29').ratio() ==
0.556`, below the 0.85 bar), reading as a scoring miss without being a wrong answer. Not a delta
engine defect — text-block boundaries are not perfectly stable across independently-rendered PDF
revisions in general, which is worth knowing about any PDF-text-extraction pipeline, not just
this one.

## 7. Dashboard operational scope

In-memory session registry, single process, no authentication — a documented scope boundary from
the dashboard's own design (`src/dashboard/state.py`'s module docstring), not an oversight.
Report/markup *files* survive a restart on disk; the in-memory session list and chat history do
not. Fine for `make dashboard` as a local demo tool; not a deployment target as-is.
