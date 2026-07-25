"""Tests for eval/judge.py: reply parsing (the part most likely to meet a
real free-tier model's quirks) and orchestration against a FakeLLM -- no
network call in this suite."""

import pytest

from eval.judge import JudgeParseError, _build_judge_prompt, _parse_judge_reply, judge_answer
from eval.schema import QAItem
from src.chat.answer import ChatAnswer, Citation


def qa_item(**overrides) -> QAItem:
    defaults = {
        "qa_id": "QA-01", "pair_id": "pair1", "question": "What is the set pressure?",
        "answerable": True, "expected_citation_texts": ["SP = 260 bar (g)"],
        "expected_answer_summary": "260 bar (g).",
    }  # fmt: skip
    defaults.update(overrides)
    return QAItem(**defaults)


def chat_answer(**overrides) -> ChatAnswer:
    defaults = {
        "query": "q", "answer_text": "The set pressure is 260 bar (g).",
        "citations": [], "refused": False, "retrieved_count": 5, "reranked_count": 5,
    }  # fmt: skip
    defaults.update(overrides)
    return ChatAnswer(**defaults)


class FakeLLM:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[tuple[str, str, int, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024, purpose: str = "") -> str:
        self.calls.append((system, user, max_tokens, purpose))
        return self._reply


class TestParseJudgeReply:
    def test_parses_well_formed_reply(self):
        text = "CORRECTNESS: 5\nGROUNDEDNESS: 4\nREASON: matches the reference."
        correctness, groundedness, reason = _parse_judge_reply(text)
        assert correctness == 5
        assert groundedness == 4
        assert reason == "matches the reference."

    def test_tolerant_of_markdown_emphasis(self):
        text = "**CORRECTNESS:** 3\n**GROUNDEDNESS:** 2\n**REASON:** partial match."
        correctness, groundedness, reason = _parse_judge_reply(text)
        assert correctness == 3
        assert groundedness == 2

    def test_tolerant_of_case(self):
        text = "correctness: 5\ngroundedness: 5\nreason: great."
        correctness, groundedness, _ = _parse_judge_reply(text)
        assert correctness == 5
        assert groundedness == 5

    def test_tolerant_of_preamble_before_fields(self):
        text = "Let me evaluate this.\n\nCORRECTNESS: 4\nGROUNDEDNESS: 4\nREASON: fine."
        correctness, groundedness, _ = _parse_judge_reply(text)
        assert correctness == 4
        assert groundedness == 4

    def test_strips_think_tags_before_parsing(self):
        text = (
            "<think>reasoning about the answer...</think>"
            "CORRECTNESS: 5\nGROUNDEDNESS: 5\nREASON: ok."
        )
        correctness, groundedness, _ = _parse_judge_reply(text)
        assert correctness == 5
        assert groundedness == 5

    def test_missing_reason_defaults_to_empty(self):
        text = "CORRECTNESS: 3\nGROUNDEDNESS: 3"
        correctness, groundedness, reason = _parse_judge_reply(text)
        assert correctness == 3
        assert reason == ""

    def test_unparseable_reply_raises(self):
        with pytest.raises(JudgeParseError):
            _parse_judge_reply("I refuse to score this.")

    def test_clamps_out_of_range_scores(self):
        text = "CORRECTNESS: 9\nGROUNDEDNESS: 0\nREASON: edge case."
        correctness, groundedness, _ = _parse_judge_reply(text)
        assert correctness == 5
        assert groundedness == 1


class TestBuildJudgePrompt:
    def test_includes_question_and_reference(self):
        prompt = _build_judge_prompt(qa_item(), chat_answer())
        assert "What is the set pressure?" in prompt
        assert "260 bar (g)." in prompt

    def test_includes_citations(self):
        answer = chat_answer(
            citations=[
                Citation(
                    element_id="p:a:00001", pid="p", revision_label=None,
                    type="setpoint", page=0, text="SP = 260 bar (g)",
                )
            ]
        )
        prompt = _build_judge_prompt(qa_item(), answer)
        assert "p:a:00001" in prompt

    def test_no_citations_shows_placeholder(self):
        prompt = _build_judge_prompt(qa_item(), chat_answer(citations=[]))
        assert "(none)" in prompt

    def test_includes_refusal_state(self):
        answer = chat_answer(refused=True, answer_text="NOT_GROUNDED: n/a")
        prompt = _build_judge_prompt(qa_item(), answer)
        assert "True" in prompt


class TestJudgeAnswer:
    def test_returns_parsed_score(self):
        llm = FakeLLM("CORRECTNESS: 5\nGROUNDEDNESS: 5\nREASON: accurate and grounded.")
        result = judge_answer(qa_item(), chat_answer(), llm=llm)
        assert result.qa_id == "QA-01"
        assert result.correctness == 5
        assert result.groundedness == 5
        assert result.reason == "accurate and grounded."
        assert result.raw_reply == llm._reply

    def test_calls_llm_with_judge_purpose_tag(self):
        llm = FakeLLM("CORRECTNESS: 5\nGROUNDEDNESS: 5\nREASON: ok.")
        judge_answer(qa_item(), chat_answer(), llm=llm)
        assert len(llm.calls) == 1
        _, _, _, purpose = llm.calls[0]
        assert purpose == "eval_judge"

    def test_malformed_reply_propagates_parse_error(self):
        llm = FakeLLM("not a valid judge reply at all")
        with pytest.raises(JudgeParseError):
            judge_answer(qa_item(), chat_answer(), llm=llm)
