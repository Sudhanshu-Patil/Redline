# Sample data provenance

Tracks the origin and exact edits for every ingestion pair used in this project, per
`IMPLEMENTATION_PLAN.md` §6. Filled in as each pair is created (Phase 2 for Pairs 1/2/4,
Phase 4 for Pair 3, Phase 12 for Pair 5).

## Originals

- `originals/lift_gas_compressor_26-KA-901.pdf` — Lift Gas Compressor P&ID, as provided.
- `originals/export_gas_compressor_26-KA-902.pdf` — Export Gas Compressor P&ID, as provided.

These are **companion documents, not a revision pair** (different equipment, different
trains, different design data) — see `IMPLEMENTATION_PLAN.md` §0 and §6. They're used as:
(a) structural validation input for the native-PDF adapter, and (b) the source for Pair 4,
an intentional negative-control pair for the delta engine.

## Pairs

| Pair | Format | Status | Description |
|---|---|---|---|
| 1 | pdf_native | Pending Phase 2 | Primary synthetic edits on Lift Gas Compressor P&ID |
| 2 | pdf_scanned | Pending Phase 2 | Rasterized + OCR'd version of Pair 1's revised PDF |
| 3 | dwg | Pending Phase 4 | DXF conversion + synthetic edit |
| 4 | pdf_native (negative control) | Pending Phase 2 | 26-KA-901 vs 26-KA-902 as given |
| 5 | pdf_native (stress) | Pending Phase 12 | Duplicated/scattered-edit stress set |

Each pair's entry below will document: source file(s), exact edits made (field, old value,
new value, location), and why that edit was chosen as a meaningful test case.
