"""Pair 5: multi-page stress set (IMPLEMENTATION_PLAN.md §6/§12 Phase 12),
built for the scaling narrative (§14: "what would you do differently for a
500-sheet P&ID set?"), not for the delta P/R/F1 eval harness -- 4 labeled
pairs already satisfies plan §10's "4-5 labeled pairs," and forcing this
into the same manifest/ground-truth machinery the eval harness expects
would fit its actual purpose (a real, reproducible scaling measurement)
worse than a dedicated script does.

Pair 1's real, already-verified A/B documents (six real edits, confirmed
correct by synthesize_pairs.py's own assertions) are duplicated across N
synthetic pages -- each page is bit-for-bit the same real content, just
relabeled onto a different `bbox.page`, so the ground truth is exactly
"N times Pair 1's six edits" with no new synthesis risk. This is also what
originally surfaced the Phase 12 page-scoping bug (src/delta/align.py):
duplicate content across pages, sharing identical *normalized* bbox
positions per page, is exactly the case no single-page sample pair (1-4)
ever exercised.

Run: uv run python scripts/stress_test_pair5.py   (takes a few minutes --
embeds and aligns real content at each of several scales)
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.schema import load_manifest  # noqa: E402
from src.canonical.model import CanonicalDocument, Element, make_element_id  # noqa: E402
from src.delta.engine import compute_delta  # noqa: E402
from src.ingest.pdf_native import PdfNativeAdapter  # noqa: E402
from src.observability import tracing  # noqa: E402
from src.observability.logging import get_logger  # noqa: E402

log = get_logger("stress_test_pair5")

SAMPLES = REPO_ROOT / "data" / "samples"
PAIR5_DIR = SAMPLES / "pair5"
PAGE_COUNTS = [1, 2, 5, 10]  # scaling curve, not just one data point


def _duplicate_across_pages(doc: CanonicalDocument, n_pages: int) -> CanonicalDocument:
    """N bit-for-bit copies of doc's elements, one per synthetic page --
    same content, same relative position, different bbox.page and a fresh
    element id per copy (ids must stay unique within the document)."""
    elements: list[Element] = []
    seq = 0
    for page in range(n_pages):
        for el in doc.elements:
            elements.append(
                el.model_copy(
                    update={
                        "id": make_element_id(doc.pid, el.source_adapter, seq),
                        "bbox": el.bbox.model_copy(update={"page": page}),
                    }
                )
            )
            seq += 1
    return doc.model_copy(update={"page_count": n_pages, "elements": elements})


def _run_at_scale(doc_a1: CanonicalDocument, doc_b1: CanonicalDocument, n_pages: int) -> dict:
    doc_a = _duplicate_across_pages(doc_a1, n_pages)
    doc_b = _duplicate_across_pages(doc_b1, n_pages)

    start = time.perf_counter()
    report = compute_delta(doc_a, doc_b)
    elapsed = time.perf_counter() - start

    expected_edits_per_page = 6  # Pair 1's real, verified edit count
    result = {
        "pages": n_pages,
        "elements_per_side": len(doc_a.elements),
        "elapsed_seconds": round(elapsed, 3),
        "added": report.stats.added,
        "removed": report.stats.removed,
        "modified": report.stats.modified,
        "expected_modified_at_least": expected_edits_per_page * n_pages,
        "alignment_rate": report.stats.alignment_rate,
        "exact_key_rate": report.stats.exact_key_rate,
    }
    log.info("stress test scale point", extra={"extra_fields": result})
    return result


def main() -> None:
    with tracing.trace("stress_test_pair5"):
        manifest = load_manifest(SAMPLES / "pair1" / "manifest.json")
        doc_a1 = PdfNativeAdapter().ingest(
            Path(manifest.doc_a.path), pid="pair5_A", revision_label=manifest.doc_a.revision_label
        )
        doc_b1 = PdfNativeAdapter().ingest(
            Path(manifest.doc_b.path), pid="pair5_B", revision_label=manifest.doc_b.revision_label
        )

        results = [_run_at_scale(doc_a1, doc_b1, n) for n in PAGE_COUNTS]

        PAIR5_DIR.mkdir(parents=True, exist_ok=True)
        largest_n = PAGE_COUNTS[-1]
        doc_a_large = _duplicate_across_pages(doc_a1, largest_n)
        doc_b_large = _duplicate_across_pages(doc_b1, largest_n)
        (PAIR5_DIR / "A.canonical.json").write_text(
            doc_a_large.model_dump_json(indent=2), encoding="utf-8"
        )
        (PAIR5_DIR / "B.canonical.json").write_text(
            doc_b_large.model_dump_json(indent=2), encoding="utf-8"
        )
        (PAIR5_DIR / "stress_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        print(
            f"{'pages':>6} {'elements/side':>14} {'seconds':>9} "
            f"{'added':>7} {'removed':>8} {'modified':>9}"
        )
        for r in results:
            print(
                f"{r['pages']:>6} {r['elements_per_side']:>14} {r['elapsed_seconds']:>9.3f} "
                f"{r['added']:>7} {r['removed']:>8} {r['modified']:>9}"
            )
        print(
            f"\nWrote {PAIR5_DIR / 'stress_results.json'} "
            f"and the {largest_n}-page canonical JSON."
        )


if __name__ == "__main__":
    main()
