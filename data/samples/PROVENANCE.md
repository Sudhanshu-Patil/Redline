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

## Pair 3 — DWG (pending Phase 4)

Will be added by the DWG phase: DXF conversion of a P&ID sheet + one synthetic edit, with
conversion provenance documented here.

## Pair 5 — stress set (pending Phase 12)

Duplicated sheet set with scattered edits, for the scaling analysis.
