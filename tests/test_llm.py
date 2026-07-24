"""LLMClient tests that must never touch the network.

Every test pins settings.llm_provider explicitly so results don't depend on
whatever a local .env selects.
"""

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


class TestUnconfigured:
    @pytest.fixture(autouse=True)
    def no_keys(self, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        monkeypatch.setattr(settings, "llm_api_key", "")

    @pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
    def test_is_configured_false_without_key(self, monkeypatch, provider):
        monkeypatch.setattr(settings, "llm_provider", provider)
        assert LLMClient().is_configured is False

    @pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
    def test_complete_raises_without_key(self, monkeypatch, provider):
        monkeypatch.setattr(settings, "llm_provider", provider)
        with pytest.raises(LLMNotConfiguredError):
            LLMClient().complete(system="s", user="u")

    @pytest.mark.parametrize("provider", ["anthropic", "openai_compatible"])
    def test_read_image_text_raises_without_key(self, monkeypatch, provider):
        monkeypatch.setattr(settings, "llm_provider", provider)
        with pytest.raises(LLMNotConfiguredError):
            LLMClient().read_image_text(b"not-a-png")


@pytest.mark.parametrize(
    ("provider", "key_field"),
    [("anthropic", "anthropic_api_key"), ("openai_compatible", "llm_api_key")],
)
def test_is_configured_true_with_key(monkeypatch, provider, key_field):
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, key_field, "sk-test-not-real")
    assert LLMClient().is_configured is True


# --- anthropic backend contract ---


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


@pytest.fixture
def anthropic_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-not-real")
    client = LLMClient()
    fake = _FakeAnthropicClient()
    client._client = fake
    return client, fake


def test_anthropic_read_image_text_request_contract(anthropic_client):
    """Locks the vision request shape: configured model, thinking disabled,
    base64 PNG image block first, prompt text second."""
    client, fake = anthropic_client
    result = client.read_image_text(b"png-bytes", context_hint="near PSV")

    assert result == "PSV 9066B"
    kwargs = fake.messages.last_kwargs
    assert kwargs["model"] == settings.anthropic_vision_model
    assert kwargs["thinking"] == {"type": "disabled"}
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert "near PSV" in content[1]["text"]


def test_anthropic_complete_request_contract(anthropic_client):
    client, fake = anthropic_client
    result = client.complete(system="be brief", user="hello", max_tokens=64)

    assert result == "PSV 9066B"
    kwargs = fake.messages.last_kwargs
    assert kwargs["model"] == settings.anthropic_model
    assert kwargs["system"] == "be brief"
    assert kwargs["max_tokens"] == 64


# --- openai_compatible backend contract (Groq / Moonshot / Kimi ...) ---


class _FakeOaiUsage:
    prompt_tokens = 900
    completion_tokens = 15


class _FakeOaiMessage:
    content = "  HH: 250  "


class _FakeOaiChoice:
    message = _FakeOaiMessage()
    finish_reason = "stop"


class _FakeOaiResponse:
    usage = _FakeOaiUsage()
    choices = [_FakeOaiChoice()]


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeOaiResponse()


class _FakeOaiChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOaiClient:
    def __init__(self):
        self.chat = _FakeOaiChat()


@pytest.fixture
def oai_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_api_key", "gsk-test-not-real")
    client = LLMClient()
    fake = _FakeOaiClient()
    client._oai_client = fake
    return client, fake


def test_oai_read_image_text_request_contract(oai_client, monkeypatch):
    """Locks the vision request shape for OpenAI-compatible providers:
    vision model, data-URI image_url block first, prompt text second,
    reasoning hidden via extra_body, headroom max_tokens."""
    monkeypatch.setattr(settings, "llm_vision_reasoning_hidden", True)
    client, fake = oai_client
    result = client.read_image_text(b"png-bytes", context_hint="near PIT")

    assert result == "HH: 250"
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs["model"] == settings.llm_vision_model
    assert kwargs["max_tokens"] == settings.llm_vision_max_tokens
    assert kwargs["extra_body"] == {"reasoning_format": "hidden"}
    content = kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "near PIT" in content[1]["text"]


def test_oai_read_image_omits_reasoning_param_when_disabled(oai_client, monkeypatch):
    """Providers that reject Groq's reasoning_format must not receive it."""
    monkeypatch.setattr(settings, "llm_vision_reasoning_hidden", False)
    client, fake = oai_client
    client.read_image_text(b"png-bytes")
    assert "extra_body" not in fake.chat.completions.last_kwargs


def test_think_tags_stripped_from_vision_reply(oai_client, monkeypatch):
    """A reasoning model that leaks <think> blocks must still yield clean text."""
    from src.chat.llm import strip_think_tags

    assert strip_think_tags("<think>hmm\nlines</think>\nPSV 9066B") == "PSV 9066B"
    assert strip_think_tags("no tags here") == "no tags here"
    # Orphan closing tag (observed live: hidden-reasoning truncation leak)
    assert strip_think_tags("leaked reasoning</think>\nHH: 250") == "HH: 250"

    monkeypatch.setattr(
        _FakeOaiMessage, "content", "<think>analysis...</think>\n  PIT-9062  "
    )
    client, _ = oai_client
    assert client.read_image_text(b"png-bytes") == "PIT-9062"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("<empty>", ""),
        ("(empty)", ""),
        ("EMPTY", ""),
        ("no text", ""),
        ("No text visible", ""),
        ("N/A", ""),
        ("none", ""),
        ("PSV 9066B", "PSV 9066B"),  # real answers untouched
        ("NONE SHALL PASS", "NONE SHALL PASS"),  # 'none' only as whole reply
    ],
)
def test_empty_reply_normalization(reply, expected):
    from src.chat.llm import normalize_empty_reply

    assert normalize_empty_reply(reply) == expected


def test_oai_complete_request_contract(oai_client):
    client, fake = oai_client
    result = client.complete(system="be brief", user="hello", max_tokens=64)

    assert result == "HH: 250"
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs["model"] == settings.llm_model
    assert kwargs["messages"][0] == {"role": "system", "content": "be brief"}
    assert kwargs["max_tokens"] == 64
