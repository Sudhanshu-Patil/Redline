# Brief for Claude Code (v2 — full scope, 40-hour budget)

Paste this whole file as your first message in a fresh Claude Code session, in an empty repo directory. `IMPLEMENTATION_PLAN.md` should sit alongside it in the repo root — read that first, it has the full design detail; this file is the execution contract.

---

We're building the "Document Delta & Grounded Chat" take-home end-to-end: all three ingestion formats (native PDF, scanned PDF, DWG) for real, a rigorous delta engine, a full observability layer, a grounded chat with citations, a delta markup overlay, a small served dashboard, and a deep eval harness (delta metrics, chat groundedness, retrieval-quality metrics, cost/latency analysis). We have a genuinely extended timeline, so **do not silently cut bonus scope** — everything in `IMPLEMENTATION_PLAN.md` §12 is in scope. If something looks like it needs cutting, tell me and let me decide, don't drop it unilaterally.

## Ground rules

1. **Work phase by phase, in the order in `IMPLEMENTATION_PLAN.md` §12** (0 through 13 — this now includes DWG, markup overlay, and the dashboard as ordinary phases, not optional add-ons). After each phase: run the relevant tests/checks, show me a short summary of what was built and any judgment call you made that wasn't already pinned down in the plan, then **stop and wait for me to say "go"** before starting the next phase. Don't batch multiple phases into one turn, even though there's more time available — I want visibility at each checkpoint.
2. **Determinism**: delta alignment and add/remove/modify classification stay deterministic, non-LLM logic (§4 of the plan). Add a test in Phase 5 that actually asserts the classifier path never invokes the LLM client — don't just document this, prove it.
3. **Config over hardcoding**: model names, thresholds, paths in config/env.
4. **No secrets committed.** Env vars only, `.env` gitignored, `.env.example` kept current.
5. **Observability from Phase 1 onward**, not bolted on later — every phase's code should already be emitting trace spans and structured logs per §9 of the plan by the time you show it to me.
6. **Write real tests as you go**, not just at Phase 12 — unit tests for alignment edge cases belong in Phase 5 itself, not deferred.
7. For DWG (Phase 4): if the conversion tooling stalls for more than ~2 hours of wall-clock effort, stop, tell me exactly what's blocking, and propose the fallback from §13 of the plan rather than quietly downgrading to a stub.
8. Commit at the end of each completed phase with a message naming the phase.

## Phase order

0. Repo scaffold + canonical model + CI skeleton
1. Native PDF adapter
2. Sample data: synthesize Pairs 1, 2, 4 + provenance
3. Scanned PDF adapter (OCR + vision-LLM fallback for low-confidence regions)
4. DWG adapter (real parse) + Pair 3
5. Delta engine: alignment + classification + confidence + unit tests
6. Delta report renderer (JSON + Markdown/HTML + chart)
7. Observability: tracing, structured logging, LLM telemetry, served `/metrics`
8. Grounded chat: hybrid retrieval + rerank + cited answers + refusal handling
9. Delta markup overlay (PDF + DWG-rendered)
10. Eval harness: delta P/R/F1, validated chat-groundedness judge, retrieval-quality eval, cost/latency analysis, `eval-diff`
11. Dashboard: submit pair, view report, chat, view metrics
12. Tests, lint/type-check, CI hardening, Pair 5 stress test
13. README, DEMO.md, failure table, scaling notes, interview-prep notes (plan §14)

## First message back to me

Confirm you've read `IMPLEMENTATION_PLAN.md`, flag anything in it that looks wrong or underspecified for Phase 0/1, then start Phase 0.
