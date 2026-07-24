"""Cross-encoder reranking pass over hybrid retrieval results (plan §7).

A bi-encoder (the embedder in chat/index.py) scores query and passage
independently, so it's fast but only approximately relevant -- a
cross-encoder attends over the (query, passage) pair jointly and reorders
the top-k by genuine relevance before the LLM ever sees the context, which
is the whole reason this pass exists over the v1 design (plan §7).
"""

from typing import Protocol

from src.chat.index import RetrievedChunk
from src.config import settings
from src.observability import tracing


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage, same order, higher = more relevant."""
        ...


class CrossEncoderReranker:
    """Local, deterministic cross-encoder (plan §2) -- never the LLMClient.
    Lazily loaded once and cached at class level, same pattern as
    SentenceTransformerEmbedder (src/delta/align.py).
    """

    _model: object | None = None

    def score(self, query: str, passages: list[str]) -> list[float]:
        if CrossEncoderReranker._model is None:
            from sentence_transformers import CrossEncoder

            with tracing.span("chat.load_reranker_model", model=settings.reranker_model):
                CrossEncoderReranker._model = CrossEncoder(settings.reranker_model)
        model = CrossEncoderReranker._model
        pairs = [(query, p) for p in passages]
        scores = model.predict(pairs)  # type: ignore[attr-defined]
        return [float(s) for s in scores]


def rerank_chunks(
    query: str, chunks: list[RetrievedChunk], reranker: Reranker, top_k: int
) -> list[RetrievedChunk]:
    with tracing.span("chat.rerank", candidates=len(chunks), top_k=top_k) as sp:
        if not chunks:
            sp["reranked"] = 0
            return []
        scores = reranker.score(query, [c.text for c in chunks])
        rescored = [
            c.model_copy(update={"score": s})
            for c, s in zip(chunks, scores, strict=True)
        ]
        # Deterministic tie-break by element_id, same rationale as
        # align.py::_greedy_assign: results never depend on input order.
        rescored.sort(key=lambda c: (-c.score, c.element_id))
        top = rescored[:top_k]
        sp["reranked"] = len(top)
        return top
