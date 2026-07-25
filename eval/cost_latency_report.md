# Cost / Latency Budget Analysis

Generated 2026-07-25T06:18:26.714680+00:00 from 2479 trace file(s) / 5533 span(s).

## Cost

Total spend across 285 LLM call(s) in the scanned trace history: **$0.0192**.

- **Cost per delta run:** $0.0000 (deterministic, 351 run(s) measured -- the delta engine never calls an LLM, by design; see the LLM-never-called test in tests/test_delta_engine.py).
- **Cost per chat turn:** $0.000000 average (51 turn(s), $0.0000 total, 19,087 input / 1,208 output tokens).
- **Cost per eval-judge call:** $0.000000 average (45 call(s), kept separate from chat-turn cost via the purpose tag -- otherwise judge eval spend would silently inflate the reported per-turn figure).

## Where the latency goes

### Ingest

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| PDF native: full ingest | 326 | 45.3 | 33.9 | 106.9 | 258.0 |
| PDF scanned: full ingest | 44 | 771,978.9 | 20,713.3 | 1,465,849.3 | 26,047,186.1 |
| PDF scanned: tesseract OCR pass | 46 | 20,171.1 | 19,949.6 | 22,197.8 | 24,654.9 |
| PDF scanned: page rasterize | 46 | 247.3 | 249.4 | 304.1 | 367.0 |
| PDF scanned: vision-LLM fallback (per region) | 204 | 161,890.2 | 0.7 | 1,352.7 | 26,024,373.7 |
| DWG/DXF: full ingest | 259 | 126.5 | 22.1 | 619.0 | 3,756.5 |

### Delta engine

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| Delta: embedding model load (one-time/process) | 46 | 5,588.2 | 5,422.3 | 6,304.0 | 8,161.5 |
| Delta: tier 1 (exact key) | 715 | 0.3 | 0.0 | 1.6 | 24.2 |
| Delta: tier 2 (geometry) | 714 | 0.0 | 0.0 | 0.2 | 1.4 |
| Delta: tier 3 (embedding proximity) | 769 | 1,106.6 | 0.0 | 13,748.5 | 43,603.0 |
| Delta: full compute (all tiers + classify) | 596 | 1,334.3 | 2.0 | 15,116.8 | 43,611.3 |

### Chat

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| Chat: exact tag lookup | 147 | 7.9 | 0.0 | 37.3 | 79.4 |
| Chat: vector search | 139 | 10.1 | 12.6 | 20.0 | 26.7 |
| Chat: hybrid retrieval (full) | 115 | 22.9 | 14.7 | 57.3 | 97.0 |
| Chat: cross-encoder rerank | 197 | 468.8 | 0.0 | 180.8 | 31,208.0 |
| Chat: full turn (retrieve+rerank+LLM) | 126 | 677.1 | 0.8 | 2,418.7 | 32,658.6 |

### Raw LLM calls

| Stage | Count | Mean (ms) | p50 | p95 | Max |
|---|---|---|---|---|---|
| LLM: text completion call | 145 | 559.6 | 248.2 | 2,415.0 | 3,423.7 |
| LLM: vision OCR-fallback call | 225 | 161,286.2 | 927.8 | 123,020.7 | 25,567,099.2 |


## At 10x / 100x document volume

- **Native PDF ingest** measured at 45ms/doc average (p95 107ms). At 10x (tens of documents) this is still trivially single-machine; at 100x (hundreds), ingest itself is not the bottleneck -- it's embarrassingly parallel across documents with no shared state, so it scales by adding workers, not by algorithmic changes.
- **Scanned PDF ingest is the real bottleneck**: measured at 771,979ms/doc average (p95 1,465,849ms) -- tesseract plus any vision-LLM fallback calls dominate. At 100x volume this stage alone would need a job queue (not synchronous request/response) and fallback-call batching or a stricter per-document fallback-region cap (VISION_FALLBACK_MAX_REGIONS already exists as the knob) to keep worst-case documents from starving the queue.
- **Delta alignment is O(n x m) per tier over the leftover pool after coarser tiers thin it** (documented in src/delta/align.py) -- fine at hundreds of elements per sheet, but a 500-sheet set doing full pairwise scans per pair would want a spatial index (KD-tree/grid) instead, since the leftover pool after exact-key matching won't shrink proportionally on much denser or noisier sheets.
- **Embedding/reranker model load is a one-time process cost** (5,588ms measured), amortized by the class-level caching already in place (SentenceTransformerEmbedder, CrossEncoderReranker) -- at higher volume this cost is paid once per worker process, not once per document, so it matters for cold-start latency but not steady-state throughput.
- **Retrieval indexing is currently one chromadb collection per chat session** (src/chat/index.py) -- at 100x document volume this should shard by document/pair rather than growing one global collection, both to bound per-query candidate-pool size and so re-indexing one changed document doesn't touch unrelated ones (incremental re-index, not full reindex).