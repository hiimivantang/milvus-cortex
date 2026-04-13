"""Reranking strategies for search results."""

from __future__ import annotations

from abc import ABC, abstractmethod

from milvus_cortex.models import SearchResult


class Reranker(ABC):
    """Base class for search result rerankers."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Rerank search results by relevance to query."""


class CrossEncoderReranker(Reranker):
    """Reranks using a cross-encoder model from sentence-transformers.

    Requires: pip install milvus-cortex[rerank]
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install milvus-cortex[rerank]"
            )
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return results
        pairs = [(query, r.memory.content) for r in results]
        scores = self._model.predict(pairs)
        for r, s in zip(results, scores):
            r.score = float(s)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
