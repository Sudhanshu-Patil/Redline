"""LLMClient: the single place the anthropic SDK is touched (plan §2).

Everything downstream (OCR vision fallback now, chat/change-descriptions in
later phases) depends on this interface, so swapping providers or models is a
one-file change. Every call emits a telemetry span with token counts and an
estimated cost so Phase 7's /metrics can aggregate spend without a separate
bookkeeping store.

No API key is required at import time: `LLMClient.is_configured` reports
availability, callers are expected to degrade gracefully (the OCR adapter
keeps low-confidence tesseract text rather than failing the ingest).
"""

import base64

import anthropic

from src.config import settings
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM call is attempted without credentials configured."""


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * settings.llm_cost_per_mtok_input
        + output_tokens * settings.llm_cost_per_mtok_output
    ) / 1_000_000


class LLMClient:
    def __init__(self) -> None:
        self._client: anthropic.Anthropic | None = None

    @property
    def is_configured(self) -> bool:
        return bool(settings.anthropic_api_key)

    def _get_client(self) -> anthropic.Anthropic:
        if not self.is_configured:
            raise LLMNotConfiguredError(
                "ANTHROPIC_API_KEY is not set; LLM-dependent features are unavailable"
            )
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
                max_retries=settings.llm_max_retries,
                timeout=settings.llm_timeout_seconds,
            )
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Plain text completion (used by chat/report phases)."""
        client = self._get_client()
        with tracing.span("llm.complete", model=settings.anthropic_model) as sp:
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            sp["input_tokens"] = response.usage.input_tokens
            sp["output_tokens"] = response.usage.output_tokens
            sp["cost_usd"] = estimate_cost_usd(
                response.usage.input_tokens, response.usage.output_tokens
            )
            sp["stop_reason"] = response.stop_reason
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

    def read_image_text(self, png_bytes: bytes, context_hint: str = "") -> str:
        """Vision re-read of a small image crop; returns the verbatim text seen.

        Used by the scanned-PDF adapter for regions where tesseract's
        confidence fell below threshold (plan §2's "smarter than raw diff"
        judgment call). Thinking is explicitly disabled -- these are tiny
        verbatim transcription reads where reasoning adds latency, not value.
        """
        client = self._get_client()
        with tracing.span(
            "llm.read_image_text", model=settings.anthropic_vision_model
        ) as sp:
            image_data = base64.standard_b64encode(png_bytes).decode("utf-8")
            prompt = (
                "This is a small crop from a scanned engineering drawing (P&ID). "
                "Transcribe the text in this image exactly as it appears. Return ONLY "
                "the verbatim text, no commentary. If you cannot read any text, "
                "return an empty response."
            )
            if context_hint:
                prompt += f" Context: {context_hint}"
            response = client.messages.create(
                model=settings.anthropic_vision_model,
                max_tokens=200,
                thinking={"type": "disabled"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            sp["input_tokens"] = response.usage.input_tokens
            sp["output_tokens"] = response.usage.output_tokens
            sp["cost_usd"] = estimate_cost_usd(
                response.usage.input_tokens, response.usage.output_tokens
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
