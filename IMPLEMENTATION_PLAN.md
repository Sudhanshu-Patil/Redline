# Document Delta & Grounded Chat — Implementation Plan (v2, 40-hour scope)

Applied AI Engineer take-home. Revised for an extended ~40-hour window — no bonus cuts, all three ingestion formats implemented for real, full observability + eval depth.

---

## 0. Two notes on the source material (carried over, still true)

**Internal rubric section.** The assignment HTML still contains a section marked "INTERNAL — HIRING PANEL ONLY" whose own footer says it should have been stripped before distribution. I'd still flag it to whoever sent the assignment — costs nothing, and everything useful in it (bonus items, what a strong answer looks like) is already stated in the public sections 03–09 anyway, which is what this plan builds from. I've folded the five likely follow-up questions into §14 as prep, since anticipating interview questions from a public-shaped brief is normal prep either way.

**Your two PDFs are companion documents, not a revision pair.** `26-KA-902` (Export Compressor) and `26-KA-901` (Lift Gas Compressor) are different equipment, different trains, different design data. Real, well-formed native PDFs — good structural samples — but not an A/B pair. Plan still treats them as: (a) structural validation input for the native-PDF adapter, and (b) an intentional negative-control sample for the delta engine (see §6). Primary pairs are synthesized from one of them with documented, traceable edits.

---

## 1. The clock

Extended deadline gives real room — treat that as budget for **depth and coverage**, not padding. All three formats end-to-end, full markup overlay, a served dashboard, and a genuinely rigorous eval harness (retrieval-quality metrics + cost/latency budget analysis) are now core scope, not stretch goals. The one thing that doesn't change: delta quality and eval rigor are still where this gets judged hardest, so they still get the largest single blocks of time and nothing downstream (dashboard, markup polish) should eat into them.

---

## 2. Stack (unchanged core choices, expanded for full scope)

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+, `uv` | fast installs |
| Native PDF | `pymupdf` (fitz) | text + bbox + vector extraction |
| Scanned PDF | `pdf2image` + Tesseract, with a vision-LLM fallback (send low-OCR-confidence regions to the LLM for a second read) | shows judgment on OCR limits per the brief's "smarter than raw diff" bar |
| DWG | `ezdxf` against a DXF export (ODA File Converter CLI or a free sample/converted DWG), real parser — layers, blocks, text entities, dimensions | full format now in scope |
| Canonical model | `pydantic` v2 | validation + serialization |
| Alignment | `sentence-transformers` (`all-MiniLM-L6-v2`) + exact key match | deterministic, fast, local |
| Vector store | `chromadb` | zero infra |
| Reranking | cross-encoder (`ms-marco-MiniLM`) on top-k retrieval | worth it now there's time; strengthens groundedness |
| LLM | Claude (Sonnet) via `anthropic` SDK behind `LLMClient` interface | swappable, key via env var |
| Observability | hand-rolled JSON tracer (OTel-shaped spans) + structured logger + a served `/metrics` page | no infra risk, still fully inspectable |
| Dashboard | small FastAPI + HTMX (or Streamlit) app: submit a pair, view delta report, chat, view traces/metrics | now core bonus, not stretch |
| Testing | `pytest`, `ruff`, `mypy` | worth doing properly with the extra time |
| CI | GitHub Actions: lint + type-check + unit tests + `make eval` on push | cheap, signals engineering maturity |

---

## 3. Canonical representation (unchanged design, same schema)

```python
class BBox(BaseModel):
    page: int
    x0: float; y0: float; x1: float; y1: float

class Element(BaseModel):
    id: str
    type: Literal["tag", "instrument_loop", "valve", "line_number", "note",
                   "dimension", "setpoint", "table_cell", "text_block", "geometry"]
    text: str
    bbox: BBox
    attributes: dict[str, str] = {}
    source_adapter: str
    extraction_confidence: float

class CanonicalDocument(BaseModel):
    pid: str
    format: Literal["pdf_native", "pdf_scanned", "dwg"]
    revision_label: str | None
    page_count: int
    elements: list[Element]
    raw_source_path: str
```

For DWG, `geometry` elements additionally carry `attributes={"layer": ..., "entity_type": "LINE"/"BLOCK"/"MTEXT", "block_name": ...}` so the delta engine can distinguish a moved line from a moved block from an edited text entity — this is where DWG's richer structure actually earns its keep over the PDF adapters.

---

## 4. Delta engine (same alignment strategy, more rigor)

1. **Exact key match** on tag/instrument/line numbers — deterministic, highest confidence.
2. **Geometry-aware match** for DWG: same layer + same entity type + bbox within tolerance → candidate match even without a text key.
3. **Embedding + bbox proximity** for untagged text/notes.
4. Unmatched-A → removed, unmatched-B → added, matched-but-different → modified.

With the extra time: add a **unit test suite for the alignment logic** using hand-built minimal canonical documents (not the full PDFs) so matching edge cases — a moved-and-renamed tag, a note split into two, a duplicate tag number — are covered deterministically and fast to run in CI. This directly answers "where does alignment break?" (see §14) with actual test cases instead of a verbal answer.

LLM stays scoped to change-description and chat, per the determinism requirement — same as before, now with room to write proper tests confirming the classification path never calls the LLM client.

---

## 5. Delta report — unchanged shape, add a small HTML chart

Same JSON/Markdown/HTML output as before. With the extra time, add a small counts-by-page bar and counts-by-type breakdown to the HTML report (plain SVG, no JS framework needed) — cheap, and makes the "human-readable" bar land better in a 2-4 minute demo.

---

## 6. Sample data — expanded to 4-5 pairs

- **Pair 1 (primary synthetic, native PDF):** `Lift_Gas_compressor-P_ID.pdf` as PID A; PID B edits real values — PSV 9066A/B 257→260 bar(g), PIT 9062 HH 245→250, delete Note 30, add a new note, change FLOW RATE 19057→20500 kg/h. Ground truth = your literal edit list.
- **Pair 2 (scanned):** rasterize PID B of Pair 1, OCR it, diff against native PID A — exercises the OCR adapter end-to-end with a real ground truth.
- **Pair 3 (DWG):** convert one PDF page to DXF (ODA File Converter, free) or source a small public-domain sample DWG P&ID; make one synthetic edit (moved block, changed dimension, renamed layer). Document conversion provenance carefully — this is the one most likely to have a rough edge, worth extra buffer time.
- **Pair 4 (negative control):** `26-KA-902` vs `26-KA-901` as given — labeled "not a meaningful revision pair," used to assert the delta engine doesn't just dump noise on mismatched documents, and/or that it surfaces a low-alignment-rate warning. Good, honest talking point for the README.
- **Pair 5 (stress, optional if time allows):** duplicate one document's pages several times with scattered edits to approximate a larger sheet set, for the scaling note in §14.

All five get a `PROVENANCE.md` entry: source, exact edits made, and why.

---

## 7. Grounded chat — add reranking + stricter refusal

Same hybrid retrieval (exact tag lookup + vector search) as before, now with a cross-encoder rerank pass on the top-k before they're handed to the LLM, and an explicit "cite every claim; if nothing retrieved supports the question, say so" system instruction, tested in eval (§10) with a few intentionally unanswerable questions.

---

## 8. Delta markup overlay — now core, not stretch

- Draw the delta back onto PID B (or both) as bounding-box highlights: green for added, red for removed, amber for modified, using `pymupdf`'s annotation API for the PDF cases and a simple SVG/raster overlay for the DWG case (render the DXF to an image first via `ezdxf`'s matplotlib backend, then draw on top).
- Export as an annotated PDF/PNG per pair, linked from the delta report.
- Legend + per-box tooltip text (the change description) if going through the HTML report rather than a flat PDF.

---

## 9. Observability — add a served metrics page

Same tracing/logging/telemetry design as before (§8 of v1: spans per stage, JSON traces, structured logs, LLM token/cost). With the extra time, serve `/metrics` as a small FastAPI endpoint (or a page in the dashboard) showing latency percentiles, token/cost totals, delta counts, retrieval hit rate — pulled live from the trace files, not a separate manually-maintained store.

---

## 10. Eval harness — full depth

- Delta P/R/F1 as before, now against 4-5 labeled pairs instead of 1-2.
- Chat correctness + groundedness via LLM-judge, validated by hand-scoring a held-out sample and reporting judge/human agreement — this is the "validate the judge" bar the brief calls out explicitly.
- **Retrieval-quality evaluation** (bonus item, now core): recall@k and MRR against the labeled Q&A set's expected citations.
- **Cost/latency budget analysis** (bonus item, now core): from the trace/metrics data, a short written analysis — cost per delta run, cost per chat turn, where the latency actually goes (ingest vs. embedding vs. LLM call), and what it'd look like at 10x/100x document volume.
- `make eval` prints a scorecard, writes `scorecard.json`; add a `make eval-diff` that compares two scorecard runs so a regression is visibly flagged, not just eyeballed.
- Candid failure table stays — now with more surface area (DWG conversion edge cases, OCR-on-dense-tables, ambiguous embedding-only matches) to draw honest examples from.

---

## 11. Repo scaffold (adds dashboard + CI)

```
delta-chat/
├─ README.md
├─ DEMO.md
├─ .env.example
├─ Makefile                  # run / chat / eval / eval-diff / metrics / dashboard
├─ pyproject.toml
├─ .github/workflows/ci.yml
├─ src/
│  ├─ ingest/{base.py, pdf_native.py, pdf_scanned.py, dwg.py}
│  ├─ canonical/model.py
│  ├─ delta/{align.py, engine.py, report.py}
│  ├─ chat/{index.py, rerank.py, llm.py, answer.py}
│  ├─ markup/overlay.py
│  ├─ observability/{tracing.py, logging.py, metrics.py}
│  └─ dashboard/                # FastAPI+HTMX or Streamlit app
├─ eval/{datasets/, metrics.py, run_eval.py, retrieval_eval.py, cost_latency_report.py}
├─ data/samples/                # 4-5 pairs + PROVENANCE.md
└─ tests/                       # incl. alignment edge-case unit tests
```

---

## 12. Phased roadmap — ~40 hours, all bonus items in-scope

| Phase | Work | Hours |
|---|---|---|
| 0 | Repo scaffold, canonical model, CI skeleton | 1.0 |
| 1 | Native PDF adapter → canonical elements | 3.0 |
| 2 | Sample data: synthesize Pairs 1, 2, 4 + provenance | 2.0 |
| 3 | Scanned PDF adapter (OCR + vision-LLM fallback) | 2.5 |
| 4 | DWG adapter (real parse via ezdxf/DXF) + Pair 3 | 4.0 |
| 5 | Delta engine: alignment + classification + confidence + unit tests | 5.0 |
| 6 | Delta report renderer (MD/HTML/JSON + chart) | 1.5 |
| 7 | Observability: tracing, logging, telemetry, served `/metrics` | 3.0 |
| 8 | Grounded chat: hybrid retrieval + rerank + cited answers + refusal handling | 4.0 |
| 9 | Delta markup overlay (PDF + DWG-rendered cases) | 3.0 |
| 10 | Eval harness: delta P/R/F1, chat groundedness (validated judge), retrieval eval, cost/latency analysis, `eval-diff` | 4.0 |
| 11 | Dashboard: submit pair, view report, chat, view metrics | 3.0 |
| 12 | Tests, lint/type-check, CI hardening, Pair 5 stress test | 2.0 |
| 13 | README, DEMO.md, failure table, scaling notes, interview prep (§14) | 2.0 |
| **Total** | | **~40h** |

Order still matters even with slack: finish 0–7 (core + observability wired in) before touching markup/dashboard, so the required cross-cutting bars are solid regardless of what happens to the later bonus phases.

---

## 13. Risk buffer (only if something actually blocks you — not a default cut list anymore)

DWG conversion tooling is the most likely place to lose unplanned time (format quirks, missing converter). If it stalls past ~2 extra hours, fall back to a documented real-stub behind the same adapter interface and redirect that time to eval depth or markup polish instead — both still score better than a half-working DWG path.

---

## 14. Interview prep — the follow-up questions worth having sharp answers for

Regardless of where they came from, these are the kind of questions any reviewer would ask about a system like this, so it's worth having a genuine, evidence-backed answer for each, pointed at actual repo artifacts:

- **"Walk me through how you align content between two revisions. Where does it break?"** → point to `delta/align.py` + the unit tests in §4; the honest break case is embedding-only matches on near-duplicate notes.
- **"Where is the LLM in your delta path, and why there and not elsewhere?"** → §4's determinism boundary; show the test that asserts the classifier never calls `LLMClient`.
- **"Show me a trace for a slow or failed request."** → have one bookmarked from `/traces/`, ideally the OCR-fallback or DWG path since those are your naturally slower stages.
- **"How would your eval catch a regression you introduced tomorrow?"** → `make eval-diff` (§10), walk through what a dropped-F1 or dropped-recall@k looks like in the diff output.
- **"What would you do differently for a 500-sheet P&ID set?"** → have Pair 5 (§6) and the cost/latency analysis (§10) ready — batch ingestion, incremental re-indexing instead of full reindex, sharding retrieval by document rather than one global index.

---

## 15. What to hand to Claude Code

See `CLAUDE_CODE_BRIEF.md` — paste as your first message in a fresh session. All phases above, including markup/DWG/dashboard, are in the default sequence now; it still checkpoints after each phase so you stay in control, but it no longer needs to ask permission before starting bonus work.
