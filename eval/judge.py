"""LLM-judge for chat groundedness/correctness (plan §10), validated against
a hand-scored held-out sample -- BRIEF's explicit "validate the judge" bar
(see eval/run_eval.py for the judge/human agreement report this feeds).

The judge only ever scores an already-produced ChatAnswer; it never
influences retrieval, reranking, or the chat's own refusal decision (all
deterministic, src/chat/answer.py) -- a bad judge score can't leak back into
system behavior, it's purely an offline eval signal.
"""

import re
from typing import Protocol

from pydantic import BaseModel

from eval.schema import QAItem
from src.chat.answer import ChatAnswer
from src.chat.llm import LLMClient, strip_think_tags
from src.observability import tracing

_JUDGE_SYSTEM_PROMPT = """You are evaluating an AI assistant's answer to a question about an \
engineering drawing (P&ID) revision. You are given the question, whether the question is \
actually answerable from the document (a ground-truth fact, not the assistant's opinion), a \
short reference summary of what a correct answer should contain, and the assistant's actual \
answer with its citations.

Score two dimensions, each from 1 (worst) to 5 (best):
- CORRECTNESS: does the assistant's answer accurately reflect the reference summary's content?
- GROUNDEDNESS: is every factual claim in the assistant's answer consistent with the \
reference summary and actually cited (not fabricated, not invented detail, not outside \
knowledge)?

Use exactly this decision procedure -- do not substitute your own judgment of whether the \
question "seems" answerable, use the given "answerable" field:

1. If answerable=False and the assistant declined to answer: CORRECTNESS=5, GROUNDEDNESS=5 \
(refusing an impossible question is the correct behavior).
2. If answerable=False and the assistant guessed or fabricated an answer instead of declining: \
CORRECTNESS=1, GROUNDEDNESS=1, regardless of how plausible the guess sounds.
3. If answerable=True and the assistant declined to answer (any refusal, "not grounded", "no \
information", etc.): this is a FAILURE, not a safe response -- CORRECTNESS=1 always, because \
zero of the expected information was delivered. GROUNDEDNESS may be 3-4 (not 1) only because no \
false claim was actually made -- declining is safer than fabricating, but it is still wrong and \
correctness must reflect that, not be inflated to match groundedness.
4. If answerable=True and the assistant answered: compare the answer against the reference \
summary normally -- CORRECTNESS reflects factual accuracy/completeness, GROUNDEDNESS reflects \
whether every claim is actually supported by a citation (not invented).

Respond with EXACTLY this format and nothing else:
CORRECTNESS: <integer 1-5>
GROUNDEDNESS: <integer 1-5>
REASON: <one short sentence>"""


class LLM(Protocol):
    def complete(
        self, system: str, user: str, max_tokens: int = 1024, purpose: str = ""
    ) -> str: ...


class JudgeScore(BaseModel):
    qa_id: str
    correctness: int
    groundedness: int
    reason: str
    raw_reply: str


class JudgeParseError(RuntimeError):
    """Raised when the judge's reply doesn't match the required format --
    surfaced loudly rather than silently defaulted to some score, since a
    silent default would corrupt the very eval it's meant to validate."""


# Tolerant of markdown emphasis, missing colons, and reasoning-model
# preamble before the actual fields (free-tier models don't always follow
# "EXACTLY this format" -- same defensive posture as src/chat/llm.py's
# think-tag stripping and empty-reply normalization).
_SCORE_RE = re.compile(
    r"correctness\**:?\**\s*(\d)"
    r".*?groundedness\**:?\**\s*(\d)"
    r"(?:.*?reason\**:?\**\s*(.+))?",
    re.IGNORECASE | re.DOTALL,
)


def _parse_judge_reply(text: str) -> tuple[int, int, str]:
    cleaned = strip_think_tags(text)
    match = _SCORE_RE.search(cleaned)
    if not match:
        raise JudgeParseError(f"could not parse judge reply: {text!r}")
    correctness = max(1, min(5, int(match.group(1))))
    groundedness = max(1, min(5, int(match.group(2))))
    reason = (match.group(3) or "").strip().splitlines()[0] if match.group(3) else ""
    return correctness, groundedness, reason


def _build_judge_prompt(item: QAItem, answer: ChatAnswer) -> str:
    citations = "; ".join(f"[{c.element_id}] {c.text}" for c in answer.citations) or "(none)"
    return (
        f"Question: {item.question}\n"
        f"Reference summary (what a correct answer should say): {item.expected_answer_summary}\n"
        f"Question is answerable from the document: {item.answerable}\n\n"
        f"Assistant's answer: {answer.answer_text}\n"
        f"Assistant's citations: {citations}\n"
        f"Assistant refused to answer: {answer.refused}"
    )


def judge_answer(item: QAItem, answer: ChatAnswer, llm: LLM | None = None) -> JudgeScore:
    with tracing.span("eval.judge_answer", qa_id=item.qa_id) as sp:
        client = llm or LLMClient()
        raw = client.complete(
            system=_JUDGE_SYSTEM_PROMPT,
            user=_build_judge_prompt(item, answer),
            max_tokens=200,
            purpose="eval_judge",
        )
        correctness, groundedness, reason = _parse_judge_reply(raw)
        sp["correctness"] = correctness
        sp["groundedness"] = groundedness
        return JudgeScore(
            qa_id=item.qa_id,
            correctness=correctness,
            groundedness=groundedness,
            reason=reason,
            raw_reply=raw,
        )
