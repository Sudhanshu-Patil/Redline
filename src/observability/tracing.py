"""Hand-rolled, OTel-shaped tracer. No collector, no infra: every span is one
JSON line appended to `traces/{trace_id}.jsonl`. The dashboard's /metrics
page (plan §9) reads these files directly rather than a separate store.
"""

import contextvars
import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import settings

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "span_id", default=None
)


def current_trace_id() -> str | None:
    return _current_trace_id.get()


def current_span_id() -> str | None:
    return _current_span_id.get()


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def _trace_file(trace_id: str) -> Path:
    settings.traces_dir.mkdir(parents=True, exist_ok=True)
    return settings.traces_dir / f"{trace_id}.jsonl"


def _write_span(record: SpanRecord) -> None:
    with _trace_file(record.trace_id).open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


@contextmanager
def trace(name: str, **attributes: Any) -> Iterator[str]:
    """Start a new top-level trace: one ingest call, one delta run, one chat turn."""
    trace_id = uuid.uuid4().hex[:16]
    token = _current_trace_id.set(trace_id)
    try:
        with span(name, **attributes):
            yield trace_id
    finally:
        _current_trace_id.reset(token)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Start a child span under the current trace.

    Yields a mutable dict — callers can add result attributes discovered
    mid-span (e.g. `sp["elements_extracted"] = len(elements)`) and they'll
    be merged into the recorded span. If no trace is active, mints a
    standalone trace_id so spans are never silently dropped.
    """
    trace_id = _current_trace_id.get()
    trace_token = None
    if trace_id is None:
        trace_id = uuid.uuid4().hex[:16]
        trace_token = _current_trace_id.set(trace_id)

    parent_span_id = _current_span_id.get()
    span_id = uuid.uuid4().hex[:16]
    span_token = _current_span_id.set(span_id)

    extra: dict[str, Any] = {}
    start_perf = time.perf_counter()
    start_wall = time.time()
    status = "ok"
    try:
        yield extra
    except Exception as exc:
        status = "error"
        extra.setdefault("error", str(exc))
        extra.setdefault("error_type", type(exc).__name__)
        raise
    finally:
        elapsed = time.perf_counter() - start_perf
        record = SpanRecord(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=start_wall,
            end_time=start_wall + elapsed,
            duration_ms=elapsed * 1000,
            status=status,
            attributes={**attributes, **extra},
        )
        _write_span(record)
        _current_span_id.reset(span_token)
        if trace_token is not None:
            _current_trace_id.reset(trace_token)
