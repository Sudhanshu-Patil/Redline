"""Tests for eval/retrieval_eval.py: scoring logic against a FakeIndex/
FakeReranker (no real chromadb/cross-encoder involved) -- deterministic."""

import pytest

from eval.retrieval_eval import _score_query, _text_matches, evaluate_retrieval
from eval.schema import QAItem
from src.chat.index import RetrievedChunk


def qa_item(**overrides) -> QAItem:
    defaults = {
        "qa_id": "QA-01", "pair_id": "pair1", "question": "What is the set pressure?",
        "answerable": True, "expected_citation_texts": ["SP = 260 bar (g)"],
        "expected_answer_summary": "260 bar (g).",
    }  # fmt: skip
    defaults.update(overrides)
    return QAItem(**defaults)


def chunk(element_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        element_id=element_id, pid="p", revision_label=None, type="setpoint",
        page=0, text=text, score=0.9, source="vector",
    )  # fmt: skip


class FakeIndex:
    def __init__(self, results: list[RetrievedChunk]):
        self._results = results

    def hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._results


class FakeReranker:
    """Returns candidates in the SAME order given -- lets tests control
    exact rank without needing real cross-encoder scores."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        return list(reversed(range(len(passages))))


class TestTextMatches:
    def test_exact_match(self):
        assert _text_matches("SP = 260 bar (g)", "SP = 260 bar (g)") is True

    def test_fuzzy_match(self):
        assert _text_matches("SP = 260 bar (g)", "SP = 26O bar (g)") is True

    def test_no_match(self):
        assert _text_matches("HH: 245", "SP = 260 bar (g)") is False

    def test_identical_empty_strings_match_trivially(self):
        # Same exact-match-first semantics as eval/metrics.py::_texts_match;
        # never actually reached in practice since real chunk text is never
        # empty, but the behavior should still be well-defined and consistent.
        assert _text_matches("", "") is True

    def test_one_empty_one_not_does_not_match(self):
        assert _text_matches("", "SP = 260 bar (g)") is False


class TestScoreQuery:
    def test_hit_at_rank_one(self):
        recall, rr = _score_query(["SP = 260 bar (g)", "other"], ["SP = 260 bar (g)"])
        assert recall == 1.0
        assert rr == 1.0

    def test_hit_at_rank_three(self):
        recall, rr = _score_query(["a", "b", "SP = 260 bar (g)"], ["SP = 260 bar (g)"])
        assert recall == 1.0
        assert rr == pytest.approx(1 / 3)

    def test_no_hit(self):
        recall, rr = _score_query(["a", "b"], ["SP = 260 bar (g)"])
        assert recall == 0.0
        assert rr == 0.0

    def test_no_expected_texts_scores_zero(self):
        recall, rr = _score_query(["SP = 260 bar (g)"], [])
        assert recall == 0.0
        assert rr == 0.0

    def test_any_of_multiple_expected_texts_counts(self):
        recall, rr = _score_query(["b"], ["a", "b", "c"])
        assert recall == 1.0


class TestEvaluateRetrieval:
    def test_scores_answerable_items_with_citations(self):
        index = FakeIndex([chunk("id1", "SP = 260 bar (g)"), chunk("id2", "other text")])
        items = [qa_item()]
        result = evaluate_retrieval(index, items, reranker=FakeReranker())
        assert len(result.queries) == 1
        assert result.queries[0].qa_id == "QA-01"
        assert result.recall_at_k == 1.0

    def test_skips_unanswerable_items(self):
        index = FakeIndex([chunk("id1", "irrelevant")])
        items = [qa_item(answerable=False, expected_citation_texts=[])]
        result = evaluate_retrieval(index, items, reranker=FakeReranker())
        assert result.queries == []
        assert result.recall_at_k is None

    def test_skips_answerable_items_with_no_expected_texts(self):
        index = FakeIndex([chunk("id1", "irrelevant")])
        items = [qa_item(expected_citation_texts=[])]
        result = evaluate_retrieval(index, items, reranker=FakeReranker())
        assert result.queries == []

    def test_averages_across_multiple_queries(self):
        index_hit = FakeIndex([chunk("id1", "SP = 260 bar (g)")])
        items = [
            qa_item(qa_id="Q1", expected_citation_texts=["SP = 260 bar (g)"]),
            qa_item(qa_id="Q2", expected_citation_texts=["not present anywhere"]),
        ]
        result = evaluate_retrieval(index_hit, items, reranker=FakeReranker())
        assert len(result.queries) == 2
        assert result.recall_at_k == 0.5  # one hit, one miss

    def test_no_scorable_items_returns_none_averages(self):
        index = FakeIndex([])
        result = evaluate_retrieval(index, [], reranker=FakeReranker())
        assert result.queries == []
        assert result.recall_at_k is None
        assert result.mean_reciprocal_rank is None
