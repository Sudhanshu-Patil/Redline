"""LLMClient: the single place any LLM SDK is touched (plan §2).

Two interchangeable backends behind one interface, selected by LLM_PROVIDER:

- "anthropic": Claude via the anthropic SDK (the plan's original choice).
- "openai_compatible": any provider speaking the OpenAI chat-completions
  format -- Groq's free tier, Moonshot/Kimi, OpenRouter, a local server --
  configured entirely by LLM_BASE_URL / LLM_API_KEY / LLM_MODEL /
  LLM_VISION_MODEL. No code change to swap providers.

Every call emits a telemetry span with provider, model, token counts and an
estimated cost so Phase 7's /metrics can aggregate spend without a separate
bookkeeping store. No API key is required at import time: `is_configured`
reports availability and callers degrade gracefully (the OCR adapter keeps
low-confidence tesseract text rather than failing the ingest).
"""

import base64
import re

import anthropic
import openai

from src.config import settings
from src.observability import tracing
from src.observability.logging import get_logger

log = get_logger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_EMPTY_REPLY_RE = re.compile(r"^\W*(empty|no text( visible| found)?|n/?a|none)\W*$", re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models whose
    provider doesn't (or can't be told to) hide them. Also handles the
    observed leak where only an orphan closing tag survives hidden-reasoning
    truncation: everything up to the last </think> is reasoning, not answer.
    """
    cleaned = _THINK_TAG_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    return cleaned.strip()


def normalize_empty_reply(text: str) -> str:
    """Models answer 'empty'/'no text'/'N/A' in words when asked to return
    nothing; canonicalize those to an actual empty string."""
    return "" if _EMPTY_REPLY_RE.match(text) else text

_VISION_PROMPT = (
    "This is a small crop from a scanned engineering drawing (P&ID). "
    "Transcribe the text in this image exactly as it appears. Return ONLY "
    "the verbatim text, no commentary. If you cannot read any text, "
    "return an empty response."
)


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
        self._oai_client: openai.OpenAI | None = None

    @property
    def is_configured(self) -> bool:
        if settings.llm_provider == "anthropic":
            return bool(settings.anthropic_api_key)
        return bool(settings.llm_api_key)

    # --- anthropic backend ---

    def _get_client(self) -> anthropic.Anthropic:
        if not settings.anthropic_api_key:
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

    def _anthropic_complete(self, system: str, user: str, max_tokens: int) -> str:
        client = self._get_client()
        with tracing.span(
            "llm.complete", provider="anthropic", model=settings.anthropic_model
        ) as sp:
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

    def _anthropic_read_image(self, png_bytes: bytes, context_hint: str) -> str:
        client = self._get_client()
        with tracing.span(
            "llm.read_image_text",
            provider="anthropic",
            model=settings.anthropic_vision_model,
        ) as sp:
            image_data = base64.standard_b64encode(png_bytes).decode("utf-8")
            prompt = _VISION_PROMPT + (f" Context: {context_hint}" if context_hint else "")
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
            reply = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            return normalize_empty_reply(reply)

    # --- openai_compatible backend (Groq, Moonshot/Kimi, OpenRouter, ...) ---

    def _get_oai_client(self) -> openai.OpenAI:
        if not settings.llm_api_key:
            raise LLMNotConfiguredError(
                "LLM_API_KEY is not set; LLM-dependent features are unavailable"
            )
        if self._oai_client is None:
            self._oai_client = openai.OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                max_retries=settings.llm_max_retries,
                timeout=settings.llm_timeout_seconds,
            )
        return self._oai_client

    def _oai_complete(self, system: str, user: str, max_tokens: int) -> str:
        client = self._get_oai_client()
        with tracing.span(
            "llm.complete", provider="openai_compatible", model=settings.llm_model
        ) as sp:
            response = client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            usage = response.usage
            sp["input_tokens"] = usage.prompt_tokens if usage else 0
            sp["output_tokens"] = usage.completion_tokens if usage else 0
            sp["cost_usd"] = estimate_cost_usd(
                sp["input_tokens"], sp["output_tokens"]
            )
            sp["stop_reason"] = response.choices[0].finish_reason
            return (response.choices[0].message.content or "").strip()

    def _oai_read_image(self, png_bytes: bytes, context_hint: str) -> str:
        client = self._get_oai_client()
        with tracing.span(
            "llm.read_image_text",
            provider="openai_compatible",
            model=settings.llm_vision_model,
        ) as sp:
            image_data = base64.standard_b64encode(png_bytes).decode("utf-8")
            prompt = _VISION_PROMPT + (f" Context: {context_hint}" if context_hint else "")
            kwargs: dict = {}
            if settings.llm_vision_reasoning_hidden:
                kwargs["extra_body"] = {"reasoning_format": "hidden"}
            response = client.chat.completions.create(
                model=settings.llm_vision_model,
                max_tokens=settings.llm_vision_max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                **kwargs,
            )
            usage = response.usage
            sp["input_tokens"] = usage.prompt_tokens if usage else 0
            sp["output_tokens"] = usage.completion_tokens if usage else 0
            sp["cost_usd"] = estimate_cost_usd(sp["input_tokens"], sp["output_tokens"])
            reply = strip_think_tags(response.choices[0].message.content or "")
            return normalize_empty_reply(reply)

    # --- public interface ---

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Plain text completion (used by chat/report phases)."""
        if settings.llm_provider == "anthropic":
            return self._anthropic_complete(system, user, max_tokens)
        return self._oai_complete(system, user, max_tokens)

    def read_image_text(self, png_bytes: bytes, context_hint: str = "") -> str:
        """Vision re-read of a small image crop; returns the verbatim text seen.

        Used by the scanned-PDF adapter for regions where tesseract's
        confidence fell below threshold.
        """
        if settings.llm_provider == "anthropic":
            return self._anthropic_read_image(png_bytes, context_hint)
        return self._oai_read_image(png_bytes, context_hint)
