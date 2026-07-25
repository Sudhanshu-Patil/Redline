"""Dashboard (plan §11): submit a pair, view its delta report, chat about
it, view metrics -- a thin FastAPI + Jinja2 layer over everything the
earlier phases already built (ingest adapters, delta engine, report/markup
renderers, grounded chat, /metrics). No client-side JS framework: plain
HTML forms and full-page navigation, so the only new surface area here is
routing and session bookkeeping, not a second UI stack to keep correct.

Uploaded files are saved under a fixed name derived from the *selected*
format, never the client-supplied filename, before anything touches the
filesystem. Report/markup output filenames are derived from the
user-supplied pid fields (for readability), so those are separately run
through `_safe_path_component` first -- caught by this phase's own XSS
test: an unsanitized pid of "<script>alert(1)</script>" reached
`Path.write_text` as a literal path segment and raised OSError on Windows
(and would be a path-traversal vector with a pid like "../../etc/passwd"
on POSIX). Never trust a display string as a path component.
"""

import re
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from eval.schema import load_manifest
from src.canonical.model import CanonicalDocument
from src.chat.answer import answer_question
from src.chat.index import ChatIndex
from src.config import settings
from src.dashboard.state import ChatTurn, PairSession, create_session, get_session, list_sessions
from src.delta.engine import compute_delta
from src.delta.report import write_report
from src.ingest.base import AdapterFormat
from src.ingest.registry import ingest_by_format
from src.observability import tracing
from src.observability.logging import get_logger
from src.observability.metrics_server import router as metrics_router

log = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_SAMPLE_PAIRS_DIR = Path("data/samples")
_SAMPLE_PAIR_IDS = ["pair1", "pair2", "pair3", "pair4"]

_UPLOAD_EXTENSION: dict[AdapterFormat, str] = {
    "pdf_native": ".pdf",
    "pdf_scanned": ".pdf",
}


def _upload_suffix(fmt: AdapterFormat, client_filename: str) -> str:
    """The saved filename is always server-controlled (never the raw
    client-supplied name -- see module docstring), but for "dwg" the two
    possible real formats need different handling downstream: DwgAdapter
    auto-converts via the ODA File Converter only when the path's suffix is
    literally ".dwg" (see src/ingest/dwg.py::ingest). A fixed ".dxf" here
    (the original behavior) meant that check could never fire through the
    dashboard, even with a real .dwg upload and a working converter -- the
    file would always be hard-parsed as DXF and fail with a confusing
    ezdxf error instead of converting. Reading the suffix off the client
    filename is safe here specifically because it only selects between two
    hardcoded extensions, never becomes a path component itself.
    """
    if fmt == "dwg":
        return ".dwg" if client_filename.lower().endswith(".dwg") else ".dxf"
    return _UPLOAD_EXTENSION[fmt]


def _read_upload_within_limit(upload: UploadFile) -> bytes:
    """Reads at most (dashboard_max_upload_mb + 1) MB regardless of the
    upload's real size. `.file.read()` with no argument reads the entire
    body into memory in one shot -- an unbounded-memory risk for a server
    with no auth in front of it. Raises ValueError (caught by the same
    friendly-error handler as any other bad upload) if the cap is hit."""
    limit_bytes = settings.dashboard_max_upload_mb * 1024 * 1024
    data = upload.file.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise ValueError(
            f"{upload.filename} exceeds the {settings.dashboard_max_upload_mb}MB upload limit"
        )
    return data


_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_path_component(text: str, fallback: str = "doc") -> str:
    """Collapses any run of non-alphanumeric characters to a single
    underscore and caps length -- makes a user-supplied pid safe to use as
    a filename/path segment (never trust a display string as a path)."""
    cleaned = _UNSAFE_PATH_CHARS.sub("_", text).strip("._-")
    return cleaned[:80] or fallback


def _require_session(session_id: str) -> PairSession:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="No such session (the dashboard's session list does not survive a restart)",
        )
    return session


def _build_session(
    pair_label: str, doc_a: CanonicalDocument, doc_b: CanonicalDocument
) -> PairSession:
    with tracing.span("dashboard.build_session", pair_label=pair_label):
        report = compute_delta(doc_a, doc_b)
        out_dir = settings.dashboard_runs_dir / "reports"
        safe_pid_a = _safe_path_component(doc_a.pid, fallback="a")
        safe_pid_b = _safe_path_component(doc_b.pid, fallback="b")
        basename = f"{safe_pid_a}_{safe_pid_b}_{uuid.uuid4().hex[:8]}"
        report_paths = write_report(report, out_dir, doc_a, doc_b, basename=basename)

        chat_index = ChatIndex(collection_name=f"dashboard_{basename}")
        chat_index.index_document(doc_a)
        chat_index.index_document(doc_b)

        return create_session(pair_label, doc_a, doc_b, report, report_paths, chat_index)


def create_app() -> FastAPI:
    app = FastAPI(title="Document Delta & Grounded Chat")
    app.include_router(metrics_router, prefix="/metrics")

    def _index_with_error(request: Request, message: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"sessions": list_sessions(), "sample_pair_ids": _SAMPLE_PAIR_IDS, "error": message},
            status_code=400,
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"sessions": list_sessions(), "sample_pair_ids": _SAMPLE_PAIR_IDS, "error": None},
        )

    @app.post("/submit", response_model=None)
    def submit(
        request: Request,
        pid_a: Annotated[str, Form()],
        pid_b: Annotated[str, Form()],
        format_a: Annotated[AdapterFormat, Form()],
        format_b: Annotated[AdapterFormat, Form()],
        revision_label_a: Annotated[str, Form()] = "",
        revision_label_b: Annotated[str, Form()] = "",
        doc_a_file: UploadFile | None = None,
        doc_b_file: UploadFile | None = None,
    ) -> HTMLResponse | RedirectResponse:
        missing = (
            doc_a_file is None
            or doc_b_file is None
            or not doc_a_file.filename
            or not doc_b_file.filename
        )
        if missing:
            return _index_with_error(request, "Both documents are required.")
        assert doc_a_file is not None and doc_b_file is not None  # narrowed by `missing` above
        assert doc_a_file.filename and doc_b_file.filename  # narrowed by `missing` above
        try:
            work_dir = settings.dashboard_runs_dir / "uploads" / uuid.uuid4().hex[:12]
            work_dir.mkdir(parents=True, exist_ok=True)
            path_a = work_dir / f"doc_a{_upload_suffix(format_a, doc_a_file.filename)}"
            path_b = work_dir / f"doc_b{_upload_suffix(format_b, doc_b_file.filename)}"
            path_a.write_bytes(_read_upload_within_limit(doc_a_file))
            path_b.write_bytes(_read_upload_within_limit(doc_b_file))

            doc_a = ingest_by_format(
                path_a, format_a, pid=pid_a, revision_label=revision_label_a or None
            )
            doc_b = ingest_by_format(
                path_b, format_b, pid=pid_b, revision_label=revision_label_b or None
            )
        except Exception as exc:
            log.warning("dashboard submission failed", extra={"extra_fields": {"error": str(exc)}})
            return _index_with_error(request, f"Could not process the submitted files: {exc}")

        session = _build_session(f"{pid_a} vs {pid_b}", doc_a, doc_b)
        return RedirectResponse(url=f"/session/{session.session_id}", status_code=303)

    @app.post("/submit/sample/{pair_id}", response_model=None)
    def submit_sample(request: Request, pair_id: str) -> HTMLResponse | RedirectResponse:
        if pair_id not in _SAMPLE_PAIR_IDS:
            raise HTTPException(status_code=404, detail="unknown sample pair")
        try:
            manifest = load_manifest(_SAMPLE_PAIRS_DIR / pair_id / "manifest.json")
            doc_a = ingest_by_format(
                Path(manifest.doc_a.path), manifest.doc_a.format,
                pid=manifest.doc_a.pid, revision_label=manifest.doc_a.revision_label,
            )  # fmt: skip
            doc_b = ingest_by_format(
                Path(manifest.doc_b.path), manifest.doc_b.format,
                pid=manifest.doc_b.pid, revision_label=manifest.doc_b.revision_label,
            )  # fmt: skip
        except Exception as exc:
            log.warning("sample pair load failed", extra={"extra_fields": {"error": str(exc)}})
            return _index_with_error(request, f"Could not load sample pair {pair_id}: {exc}")

        session = _build_session(manifest.description or pair_id, doc_a, doc_b)
        return RedirectResponse(url=f"/session/{session.session_id}", status_code=303)

    @app.get("/session/{session_id}", response_class=HTMLResponse)
    def session_view(request: Request, session_id: str) -> HTMLResponse:
        session = _require_session(session_id)
        return templates.TemplateResponse(
            request, "session.html", {"session": session, "stats": session.report.stats}
        )

    @app.get("/session/{session_id}/report", response_class=HTMLResponse)
    def session_report(session_id: str) -> HTMLResponse:
        session = _require_session(session_id)
        html_path = session.report_paths.get("html")
        if html_path is None or not html_path.exists():
            raise HTTPException(status_code=404, detail="report not available")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/session/{session_id}/download/{kind}")
    def session_download(session_id: str, kind: str) -> FileResponse:
        session = _require_session(session_id)
        path = session.report_paths.get(kind)
        if path is None or not path.exists():
            detail = f"'{kind}' is not available for this session"
            raise HTTPException(status_code=404, detail=detail)
        return FileResponse(path, filename=path.name)

    @app.get("/session/{session_id}/chat", response_class=HTMLResponse)
    def chat_view(request: Request, session_id: str) -> HTMLResponse:
        session = _require_session(session_id)
        return templates.TemplateResponse(request, "chat.html", {"session": session})

    @app.post("/session/{session_id}/chat")
    def chat_ask(session_id: str, question: Annotated[str, Form()]) -> RedirectResponse:
        session = _require_session(session_id)
        question = question.strip()
        if question:
            answer = answer_question(session.chat_index, question)
            session.chat_history.append(ChatTurn(question=question, answer=answer))
        return RedirectResponse(url=f"/session/{session_id}/chat", status_code=303)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.dashboard_host, port=settings.dashboard_port)
