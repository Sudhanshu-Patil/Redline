"""JSONFormatter tests. get_logger()/_configure() are already exercised
incidentally by every other test in the suite (every module calls
get_logger(__name__) at import time), so this file targets the one branch
that isn't: a log record carrying exc_info (logger.exception(...) or
logger.log(..., exc_info=True)) must serialize a formatted traceback.
"""

import json
import logging
import sys

from src.observability.logging import JSONFormatter


def _make_record(exc_info: object = None) -> logging.LogRecord:
    return logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=exc_info,
    )


class TestJSONFormatter:
    def test_basic_record_has_core_fields_and_no_exc_info(self):
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert payload["message"] == "boom"
        assert payload["level"] == "ERROR"
        assert payload["logger"] == "test.logger"
        assert "exc_info" not in payload

    def test_record_with_exception_includes_formatted_traceback(self):
        try:
            raise ValueError("boom")
        except ValueError:
            record = _make_record(exc_info=sys.exc_info())

        payload = json.loads(JSONFormatter().format(record))

        assert "exc_info" in payload
        assert "ValueError: boom" in payload["exc_info"]

    def test_extra_fields_are_merged_into_the_payload(self):
        record = _make_record()
        record.extra_fields = {"pid": "26-KA-901"}
        payload = json.loads(JSONFormatter().format(record))
        assert payload["pid"] == "26-KA-901"
