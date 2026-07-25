"""Format -> adapter registry (plan §3): the one place that knows all three
concrete adapters. New in Phase 11 because the dashboard is the fourth
independent call site that needs this dispatch (after delta/report.py,
markup/overlay.py, eval/run_eval.py's own __main__/module-level copies) --
those three are left as they are (already shipped, already tested; not
worth the risk of touching stable code for a cosmetic DRY win this late),
but a genuinely new consumer should not add a fifth copy.
"""

from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.ingest.base import AdapterFormat, IngestAdapter
from src.ingest.dwg import DwgAdapter
from src.ingest.pdf_native import PdfNativeAdapter
from src.ingest.pdf_scanned import PdfScannedAdapter

ADAPTERS: dict[AdapterFormat, type[IngestAdapter]] = {
    "pdf_native": PdfNativeAdapter,
    "pdf_scanned": PdfScannedAdapter,
    "dwg": DwgAdapter,
}


def ingest_by_format(
    path: Path, doc_format: AdapterFormat, pid: str, revision_label: str | None = None
) -> CanonicalDocument:
    return ADAPTERS[doc_format]().ingest(path, pid=pid, revision_label=revision_label)
