# Cost / Latency Budget Analysis

Generated 2026-07-25T09:14:38.208081+00:00 from 4292 trace file(s) / 10845 span(s).

## Cost

Total spend across 415 LLM call(s) in the scanned trace history: **$0.0192**.

- **Cost per delta run:** $0.0000 (deterministic, 806 run(s) measured -- the delta engine never calls an LLM, by design; see the LLM-never-called test in tests/test_delta_engine.py).
- **Cost per chat turn:** $0.000000 average (98 turn(s), $0.0000 total, 37,141 input / 2,370 output tokens).
- **Cost per eval-judge call:** $0.000000 average (90 call(s), kept separate from chat-turn cost via the purpose tag -- otherwise judge eval spend would silently inflate the reported per-turn figure).

## Where the latency goes

### Ingest

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| PDF native: full ingest | 558 | 54.1 | 35.3 | 113.8 | 3,123.1 |
| PDF scanned: full ingest | 55 | 623,140.1 | 20,808.2 | 1,404,844.1 | 26,047,186.1 |
| PDF scanned: tesseract OCR pass | 57 | 20,209.9 | 19,950.1 | 22,254.9 | 24,654.9 |
| PDF scanned: page rasterize | 57 | 248.9 | 251.4 | 310.1 | 367.0 |
| PDF scanned: vision-LLM fallback (per region) | 257 | 128,809.0 | 0.8 | 1,342.3 | 26,024,373.7 |
| DWG/DXF: full ingest | 511 | 109.0 | 23.8 | 594.7 | 3,756.5 |

### Delta engine

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| Delta: embedding model load (one-time/process) | 69 | 5,543.5 | 5,420.6 | 5,964.9 | 8,161.5 |
| Delta: tier 1 (exact key) | 1238 | 0.4 | 0.0 | 2.0 | 24.2 |
| Delta: tier 2 (geometry) | 1239 | 0.1 | 0.0 | 0.3 | 2.9 |
| Delta: tier 3 (embedding proximity) | 1313 | 1,113.1 | 0.0 | 4,368.6 | 43,603.0 |
| Delta: full compute (all tiers + classify) | 1051 | 1,339.5 | 2.2 | 15,075.6 | 43,611.3 |

### Chat

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| Chat: exact tag lookup | 291 | 16.4 | 1.5 | 58.1 | 250.4 |
| Chat: vector search | 270 | 12.0 | 13.1 | 24.2 | 164.2 |
| Chat: hybrid retrieval (full) | 225 | 35.0 | 25.2 | 91.8 | 277.0 |
| Chat: cross-encoder rerank | 411 | 506.0 | 0.0 | 7,318.4 | 31,208.0 |
| Chat: full turn (retrieve+rerank+LLM) | 274 | 777.6 | 0.8 | 2,530.0 | 32,658.6 |

### Raw LLM calls

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| LLM: text completion call | 249 | 707.6 | 299.5 | 3,278.4 | 3,693.8 |
| LLM: vision OCR-fallback call | 309 | 117,694.2 | 39.4 | 122,210.0 | 25,567,099.2 |


## At 10x / 100x document volume

- **Native PDF ingest** measured at 54ms/doc average (p95 114ms). At 10x (tens of documents) this is still trivially single-machine; at 100x (hundreds), ingest itself is not the bottleneck -- it's embarrassingly parallel across documents with no shared state, so it scales by adding workers, not by algorithmic changes.
- **Scanned PDF ingest is the real bottleneck**: measured at 623,140ms/doc average (p95 1,404,844ms) -- tesseract plus any vision-LLM fallback calls dominate. At 100x volume this stage alone would need a job queue (not synchronous request/response) and fallback-call batching or a stricter per-document fallback-region cap (VISION_FALLBACK_MAX_REGIONS already exists as the knob) to keep worst-case documents from starving the queue.
- **Delta alignment is O(n x m) per tier over the leftover pool after coarser tiers thin it** (documented in src/delta/align.py) -- fine at hundreds of elements per sheet, but a 500-sheet set doing full pairwise scans per pair would want a spatial index (KD-tree/grid) instead, since the leftover pool after exact-key matching won't shrink proportionally on much denser or noisier sheets.
- **Embedding/reranker model load is a one-time process cost** (5,544ms measured), amortized by the class-level caching already in place (SentenceTransformerEmbedder, CrossEncoderReranker) -- at higher volume this cost is paid once per worker process, not once per document, so it matters for cold-start latency but not steady-state throughput.
- **Retrieval indexing is currently one chromadb collection per chat session** (src/chat/index.py) -- at 100x document volume this should shard by document/pair rather than growing one global collection, both to bound per-query candidate-pool size and so re-indexing one changed document doesn't touch unrelated ones (incremental re-index, not full reindex).