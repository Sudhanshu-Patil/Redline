"""Tests for rerank_chunks: pure reordering/truncation logic, against a
FakeReranker so results don't depend on the real cross-encoder model."""

from src.chat.index import RetrievedChunk
from src.chat.rerank import rerank_chunks


class FakeReranker:
    """Deterministic stand-in: score is looked up by passage text, same
    determinism contract as tests.conftest.FakeEmbedder."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        return [self._scores[p] for p in passages]


def chunk(element_id: str, text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        element_id=element_id, pid="pid", revision_label=None, type="note",
        page=0, text=text, score=score, source="vector",
    )  # fmt: skip


class TestRerankChunks:
    def test_reorders_by_reranker_score(self):
        chunks = [chunk("a", "low relevance"), chunk("b", "high relevance")]
        reranker = FakeReranker({"low relevance": 0.1, "high relevance": 0.9})
        result = rerank_chunks("query", chunks, reranker, top_k=2)
        assert [c.element_id for c in result] == ["b", "a"]
        assert result[0].score == 0.9

    def test_truncates_to_top_k(self):
        chunks = [chunk(str(i), f"text{i}") for i in range(5)]
        reranker = FakeReranker({f"text{i}": float(i) for i in range(5)})
        result = rerank_chunks("query", chunks, reranker, top_k=2)
        assert len(result) == 2
        assert [c.element_id for c in result] == ["4", "3"]  # highest scores first

    def test_empty_input_returns_empty_without_calling_reranker(self):
        class ExplodingReranker:
            def score(self, query, passages):
                raise AssertionError("must not be called on empty input")

        assert rerank_chunks("query", [], ExplodingReranker(), top_k=5) == []

    def test_deterministic_tie_break_by_element_id(self):
        chunks = [chunk("z", "same"), chunk("a", "same")]
        reranker = FakeReranker({"same": 0.5})
        result = rerank_chunks("query", chunks, reranker, top_k=2)
        assert [c.element_id for c in result] == ["a", "z"]

    def test_top_k_larger_than_candidates_returns_all(self):
        chunks = [chunk("a", "x")]
        reranker = FakeReranker({"x": 0.5})
        result = rerank_chunks("query", chunks, reranker, top_k=10)
        assert len(result) == 1

    def test_original_score_is_replaced_by_reranker_score(self):
        chunks = [chunk("a", "text", score=0.99)]
        reranker = FakeReranker({"text": 0.1})
        result = rerank_chunks("query", chunks, reranker, top_k=1)
        assert result[0].score == 0.1
