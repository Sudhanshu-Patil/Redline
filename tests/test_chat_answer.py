"""Tests for the grounded-chat orchestration layer: citation extraction and
the refusal paths are pure/deterministic and tested directly; answer_question
itself is tested against fakes for retrieval, reranking, and the LLM, so no
network call and no real model load happens in this suite.
"""

from typing import Literal

from src.chat.answer import (
    _NO_GROUNDING_TEXT,
    Citation,
    _build_context,
    _extract_citations,
    answer_question,
)
from src.chat.index import RetrievedChunk


def chunk(
    element_id: str,
    text: str,
    pid: str = "pid",
    page: int = 0,
    source: Literal["exact", "vector"] = "vector",
) -> RetrievedChunk:
    return RetrievedChunk(
        element_id=element_id, pid=pid, revision_label="Rev A", type="note",
        page=page, text=text, score=0.9, source=source,
    )  # fmt: skip


class FakeIndex:
    """Stands in for ChatIndex: hybrid_search returns whatever the test wants,
    with no chromadb/embedder involved."""

    def __init__(self, results: list[RetrievedChunk]):
        self._results = results
        self.search_calls: list[tuple[str, int]] = []

    def hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.search_calls.append((query, top_k))
        return self._results


class FakeReranker:
    def __init__(self, order: list[str] | None = None):
        self._order = order

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        if self._order is None:
            return [1.0] * len(passages)
        return [float(len(self._order) - self._order.index(p)) for p in passages]


class FakeLLM:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[tuple[str, str, int, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024, purpose: str = "") -> str:
        self.calls.append((system, user, max_tokens, purpose))
        return self._reply


class ExplodingLLM:
    """Proves the deterministic refusal path never reaches the LLM -- same
    intent as test_delta_engine.py::test_classifier_never_invokes_llm_client."""

    def complete(self, system: str, user: str, max_tokens: int = 1024, purpose: str = "") -> str:
        raise AssertionError("LLM must not be called when retrieval found nothing")


class TestExtractCitations:
    def test_extracts_valid_citation(self):
        valid, invalid = _extract_citations("The value is 5 [a:b:00001].", {"a:b:00001"})
        assert valid == ["a:b:00001"]
        assert invalid == []

    def test_flags_citation_not_in_valid_set(self):
        valid, invalid = _extract_citations("The value is 5 [made:up:id].", {"a:b:00001"})
        assert valid == []
        assert invalid == ["made:up:id"]

    def test_dedupes_repeated_citations(self):
        text = "First [a:b:00001]. Also [a:b:00001] again."
        valid, _ = _extract_citations(text, {"a:b:00001"})
        assert valid == ["a:b:00001"]

    def test_no_citations_returns_empty_lists(self):
        valid, invalid = _extract_citations("No citations here.", {"a:b:00001"})
        assert valid == []
        assert invalid == []

    def test_preserves_order_of_first_appearance(self):
        text = "[b:b:2] then [a:a:1]"
        valid, _ = _extract_citations(text, {"a:a:1", "b:b:2"})
        assert valid == ["b:b:2", "a:a:1"]


class TestBuildContext:
    def test_includes_id_page_and_text(self):
        c = chunk("pid:adapter:00001", "some text", pid="P1", page=3)
        context = _build_context([c])
        assert "[pid:adapter:00001]" in context
        assert "page 3" in context
        assert "some text" in context

    def test_includes_revision_label_when_present(self):
        c = chunk("id1", "text")
        context = _build_context([c])
        assert "Rev A" in context

    def test_empty_chunks_returns_empty_string(self):
        assert _build_context([]) == ""


class TestAnswerQuestionRefusalOnEmptyRetrieval:
    def test_refuses_without_calling_llm(self):
        index = FakeIndex([])
        result = answer_question(index, "unanswerable question", llm=ExplodingLLM())
        assert result.refused is True
        assert result.answer_text == _NO_GROUNDING_TEXT
        assert result.citations == []
        assert result.retrieved_count == 0
        assert result.reranked_count == 0

    def test_refusal_reason_set(self):
        index = FakeIndex([])
        result = answer_question(index, "q", llm=ExplodingLLM())
        assert result.refusal_reason == "no relevant content retrieved"


class TestAnswerQuestionNotGroundedFromLlm:
    def test_llm_declining_is_reported_as_refusal(self):
        index = FakeIndex([chunk("id1", "irrelevant passage")])
        llm = FakeLLM("NOT_GROUNDED: the passages don't mention that.")
        result = answer_question(index, "unrelated question", llm=llm, reranker=FakeReranker())
        assert result.refused is True
        assert result.refusal_reason == "the passages don't mention that."
        assert result.citations == []

    def test_not_grounded_with_no_reason_gets_default(self):
        index = FakeIndex([chunk("id1", "text")])
        llm = FakeLLM("NOT_GROUNDED:")
        result = answer_question(index, "q", llm=llm, reranker=FakeReranker())
        assert result.refused is True
        assert result.refusal_reason == "insufficient grounded context"


class TestAnswerQuestionGroundedAnswer:
    def test_valid_citation_included(self):
        index = FakeIndex([chunk("pid:adapter:00001", "The setpoint is 260 bar")])
        llm = FakeLLM("The setpoint is 260 bar [pid:adapter:00001].")
        result = answer_question(index, "what is the setpoint?", llm=llm, reranker=FakeReranker())
        assert result.refused is False
        assert len(result.citations) == 1
        assert result.citations[0] == Citation(
            element_id="pid:adapter:00001", pid="pid", revision_label="Rev A",
            type="note", page=0, text="The setpoint is 260 bar",
        )  # fmt: skip
        assert result.invalid_citation_ids == []

    def test_hallucinated_citation_flagged_not_silently_trusted(self):
        index = FakeIndex([chunk("pid:adapter:00001", "real passage")])
        llm = FakeLLM("Some claim [pid:adapter:99999].")  # id never retrieved
        result = answer_question(index, "q", llm=llm, reranker=FakeReranker())
        assert result.citations == []
        assert result.invalid_citation_ids == ["pid:adapter:99999"]

    def test_answer_with_no_citations_still_returned(self):
        index = FakeIndex([chunk("id1", "text")])
        llm = FakeLLM("An answer with no bracketed citation at all.")
        result = answer_question(index, "q", llm=llm, reranker=FakeReranker())
        assert result.refused is False
        assert result.citations == []

    def test_reranking_narrows_context_sent_to_llm(self):
        chunks = [chunk("a", "irrelevant"), chunk("b", "the relevant one")]
        index = FakeIndex(chunks)
        llm = FakeLLM("answer [b].")
        reranker = FakeReranker(order=["the relevant one", "irrelevant"])
        answer_question(index, "q", llm=llm, reranker=reranker)
        # only the reranked-and-kept passages reach the LLM's user prompt
        _, user_prompt, _, _ = llm.calls[0]
        assert "the relevant one" in user_prompt

    def test_retrieval_uses_configured_top_k(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "retrieval_top_k", 7)
        index = FakeIndex([chunk("id1", "text")])
        answer_question(index, "q", llm=FakeLLM("NOT_GROUNDED: n/a"), reranker=FakeReranker())
        assert index.search_calls == [("q", 7)]


class TestExactMatchesBypassReranking:
    """Real bug found via a live eval run (2026-07-25): after fixing
    chat/index.py's note-number exact lookup, several "what does note N
    say" questions STILL refused, because the now-correctly-retrieved exact
    match could still be reranked out of the top rerank_top_k slots -- the
    same crowding-out mechanism chat/index.py's own docstring documents for
    the HH:250/PIT-9062 case. An exact match is a deterministic, certain
    hit (the query named a literal tag/note number); it must never lose
    that competition to the cross-encoder's approximate judgment."""

    def test_exact_match_survives_even_when_reranker_would_score_it_last(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "rerank_top_k", 2)
        exact = chunk("exact1", "the exact note content", source="exact")
        vectors = [chunk("v1", "vector passage 1"), chunk("v2", "vector passage 2")]
        index = FakeIndex([exact, *vectors])
        llm = FakeLLM("answer [exact1].")
        # The reranker is only ever asked to score the vector passages --
        # if it saw "the exact note content" it would need to be in this
        # list or FakeReranker.score would raise ValueError.
        reranker = FakeReranker(order=["vector passage 1", "vector passage 2"])

        result = answer_question(index, "what does note say", llm=llm, reranker=reranker)

        _, user_prompt, _, _ = llm.calls[0]
        assert "the exact note content" in user_prompt
        assert result.reranked_count == 2  # 1 exact + (rerank_top_k(2) - exact(1)) vector

    def test_exact_matches_exceeding_the_rerank_budget_are_all_kept(self, monkeypatch):
        from src.config import settings

        monkeypatch.setattr(settings, "rerank_top_k", 1)
        exacts = [chunk(f"exact{i}", f"exact note {i}", source="exact") for i in range(3)]
        index = FakeIndex(exacts)
        llm = FakeLLM("answer.")

        result = answer_question(index, "q", llm=llm, reranker=FakeReranker())

        assert result.reranked_count == 3


class TestChatAnswerQueryEcho:
    def test_query_is_preserved_on_answer(self):
        index = FakeIndex([])
        result = answer_question(index, "my question", llm=ExplodingLLM())
        assert result.query == "my question"
