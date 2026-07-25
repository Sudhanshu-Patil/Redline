"""Retrieval-quality evaluation (plan §10): recall@k and MRR of the real
chat retrieval pipeline (hybrid search + rerank, src/chat/index.py,
src/chat/rerank.py) against the labeled QA set's expected citation texts.

This is fundamentally an offline metric -- unlike chat.retrieve.hybrid's own
production span (latency/candidate-count telemetry, emitted on every real
query, no ground truth involved), recall@k/MRR need ground truth no live
query has. Each scored query is recorded under a distinct eval.retrieval_query
span name rather than overloading the production one -- see
src/observability/metrics.py, which learned this the hard way (see its
_RETRIEVAL_EVAL_SPAN_NAME comment).

Matching is by text, not id, for the same reason eval/metrics.py's delta
matching is: element ids are ingestion-order-dependent.
"""

from difflib import SequenceMatcher

from pydantic import BaseModel

from eval.schema import QAItem
from src.chat.index import ChatIndex
from src.chat.rerank import CrossEncoderReranker, Reranker, rerank_chunks
from src.config import settings
from src.observability import tracing

FUZZY_MATCH_THRESHOLD = 0.85


def _text_matches(candidate: str, expected: str) -> bool:
    candidate, expected = candidate.strip(), expected.strip()
    if candidate == expected:
        return True
    if not candidate or not expected:
        return False
    ratio = SequenceMatcher(None, candidate.lower(), expected.lower()).ratio()
    return ratio >= FUZZY_MATCH_THRESHOLD


def _score_query(reranked_texts: list[str], expected_texts: list[str]) -> tuple[float, float]:
    """Returns (recall_at_k, reciprocal_rank) for one query: recall is 1.0
    if any expected text was retrieved anywhere in the reranked set, 0.0
    otherwise; MRR credits the rank of the *first* hit (1-indexed)."""
    if not expected_texts:
        return 0.0, 0.0
    hit_ranks = [
        rank
        for rank, text in enumerate(reranked_texts, start=1)
        if any(_text_matches(text, expected) for expected in expected_texts)
    ]
    if not hit_ranks:
        return 0.0, 0.0
    return 1.0, 1.0 / hit_ranks[0]


class ScoredQuery(BaseModel):
    qa_id: str
    question: str
    recall_at_k: float
    reciprocal_rank: float
    top_k: int


class RetrievalEvalResult(BaseModel):
    queries: list[ScoredQuery]
    recall_at_k: float | None
    mean_reciprocal_rank: float | None


def evaluate_retrieval(
    index: ChatIndex, qa_items: list[QAItem], reranker: Reranker | None = None
) -> RetrievalEvalResult:
    reranker = reranker or CrossEncoderReranker()
    scored: list[ScoredQuery] = []
    for item in qa_items:
        if not item.answerable or not item.expected_citation_texts:
            continue
        top_k = settings.rerank_top_k
        with tracing.span("eval.retrieval_query", qa_id=item.qa_id, top_k=top_k) as sp:
            hybrid = index.hybrid_search(item.question, top_k=settings.retrieval_top_k)
            reranked = rerank_chunks(item.question, hybrid, reranker, top_k=top_k)
            recall, rr = _score_query([c.text for c in reranked], item.expected_citation_texts)
            sp["recall_at_k"] = recall
            sp["mrr"] = rr
        scored.append(
            ScoredQuery(
                qa_id=item.qa_id,
                question=item.question,
                recall_at_k=recall,
                reciprocal_rank=rr,
                top_k=top_k,
            )
        )

    if not scored:
        return RetrievalEvalResult(queries=[], recall_at_k=None, mean_reciprocal_rank=None)
    recall_avg = sum(q.recall_at_k for q in scored) / len(scored)
    mrr_avg = sum(q.reciprocal_rank for q in scored) / len(scored)
    return RetrievalEvalResult(
        queries=scored, recall_at_k=round(recall_avg, 4), mean_reciprocal_rank=round(mrr_avg, 4)
    )
