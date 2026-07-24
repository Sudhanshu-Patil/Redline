"""Structured JSON logging, correlated with the current trace/span via
observability.tracing's contextvars so a log line and its enclosing span
can be joined by trace_id when debugging (plan §9).
"""

import json
import logging
import sys
from typing import Any

from src.config import settings
from src.observability.tracing import current_span_id, current_trace_id


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = current_trace_id()
        span_id = current_span_id()
        if trace_id:
            payload["trace_id"] = trace_id
        if span_id:
            payload["span_id"] = span_id
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)


def log_fields(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log with extra structured fields, e.g. log_fields(log, logging.INFO, "done", pid=pid)."""
    logger.log(level, message, extra={"extra_fields": fields})
