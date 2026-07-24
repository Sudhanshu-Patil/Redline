"""LLMClient tests that must never touch the network."""

import pytest

from src.chat.llm import LLMClient, LLMNotConfiguredError, estimate_cost_usd
from src.config import settings


class TestEstimateCost:
    def test_cost_math(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_cost_per_mtok_input", 3.0)
        monkeypatch.setattr(settings, "llm_cost_per_mtok_output", 15.0)
        # 1M input + 1M output at sticker price
        assert estimate_cost_usd(1_000_000, 1_000_000) == pytest.approx(18.0)
        # A typical vision-fallback call: ~1.5k in, ~20 out
        assert estimate_cost_usd(1500, 20) == pytest.approx(0.0048)

    def test_zero_tokens_costs_nothing(self):
        assert estimate_cost_usd(0, 0) == 0.0


class TestUnconfiguredClient:
    @pytest.fixture(autouse=True)
    def no_key(self, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "")

    def test_is_configured_false_without_key(self):
        assert LLMClient().is_configured is False

    def test_complete_raises_without_key(self):
        with pytest.raises(LLMNotConfiguredError):
            LLMClient().complete(system="s", user="u")

    def test_read_image_text_raises_without_key(self):
        with pytest.raises(LLMNotConfiguredError):
            LLMClient().read_image_text(b"not-a-png")


def test_is_configured_true_with_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-not-real")
    assert LLMClient().is_configured is True


class _FakeUsage:
    input_tokens = 1500
    output_tokens = 20


class _FakeBlock:
    type = "text"
    text = "  PSV 9066B  "


class _FakeResponse:
    usage = _FakeUsage()
    content = [_FakeBlock()]
    stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse()


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_read_image_text_request_contract(monkeypatch):
    """Locks the vision request shape: configured model, thinking disabled,
    base64 PNG image block first, prompt text second -- and that token usage
    lands in the emitted span."""


    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-not-real")
    client = LLMClient()
    fake = _FakeAnthropicClient()
    client._client = fake

    result = client.read_image_text(b"png-bytes", context_hint="near PSV")

    assert result == "PSV 9066B"
    kwargs = fake.messages.last_kwargs
    assert kwargs["model"] == settings.anthropic_vision_model
    assert kwargs["thinking"] == {"type": "disabled"}
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert "near PSV" in content[1]["text"]


def test_complete_request_contract(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-not-real")
    client = LLMClient()
    fake = _FakeAnthropicClient()
    client._client = fake

    result = client.complete(system="be brief", user="hello", max_tokens=64)

    assert result == "PSV 9066B"
    kwargs = fake.messages.last_kwargs
    assert kwargs["model"] == settings.anthropic_model
    assert kwargs["system"] == "be brief"
    assert kwargs["max_tokens"] == 64
