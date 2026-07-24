from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from src.canonical.model import CanonicalDocument

AdapterFormat = Literal["pdf_native", "pdf_scanned", "dwg"]


class IngestAdapter(ABC):
    """Common interface every format adapter (pdf_native, pdf_scanned, dwg) implements."""

    format: AdapterFormat

    @abstractmethod
    def ingest(
        self, path: Path, pid: str, revision_label: str | None = None
    ) -> CanonicalDocument: ...
