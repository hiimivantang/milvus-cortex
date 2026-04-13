"""Retrieval orchestration — search, filter, rank, assemble context.

The orchestrator NEVER imports sparse embedding functions or branches on
BM25 mode. It always passes query_text to storage and lets the storage
layer decide how to handle it.
"""

from __future__ import annotations

import time

from milvus_cortex.config import HybridSearchConfig, MultiVectorConfig
from milvus_cortex.embedding.base import EmbeddingProvider
from milvus_cortex.models import ContextBundle, MemoryType, SearchResult
from milvus_cortex.retrieval.reranker import Reranker
from milvus_cortex.storage.milvus import MilvusStorage


class RetrievalOrchestrator:
    """Coordinates search across storage with embedding + filtering + ranking."""

    def __init__(
        self,
        storage: MilvusStorage,
        embedder: EmbeddingProvider,
        hybrid_cfg: HybridSearchConfig,
        multi_vec_cfg: MultiVectorConfig,
        reranker: Reranker | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._hybrid_cfg = hybrid_cfg
        self._multi_vec_cfg = multi_vec_cfg
        self._reranker = reranker

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_score: float = 0.0,
        mode: str = "auto",
        context_query: str | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Vector search with optional hybrid/multi-vector modes."""
        query_embedding = self._embedder.embed_one(query)

        combined_filters = dict(filters) if filters else {}
        if memory_types and len(memory_types) == 1:
            combined_filters["memory_type"] = memory_types[0].value

        if mode == "auto":
            mode = "hybrid" if self._hybrid_cfg.enabled else "dense"

        if mode == "hybrid" and self._hybrid_cfg.enabled:
            # Always pass raw query text — storage decides BM25 vs client sparse
            results = self._storage.hybrid_search(
                dense_embedding=query_embedding,
                query_text=query,
                filters=combined_filters,
                top_k=top_k,
            )
        elif mode == "multi_vector" and self._multi_vec_cfg.enabled and context_query:
            context_embedding = self._embedder.embed_one(context_query)
            results = self._storage.multi_vector_search(
                content_embedding=query_embedding,
                context_embedding=context_embedding,
                filters=combined_filters,
                top_k=top_k,
            )
        else:
            results = self._storage.search(
                embedding=query_embedding,
                filters=combined_filters,
                top_k=top_k,
            )

        # Post-filter by memory type if multiple types
        if memory_types and len(memory_types) > 1:
            type_set = {t.value for t in memory_types}
            results = [r for r in results if r.memory.memory_type.value in type_set]

        # Filter expired memories
        now = time.time()
        results = [
            r for r in results
            if r.memory.expires_at is None or r.memory.expires_at > now
        ]

        # Apply minimum score threshold
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]

        # Rerank if requested
        if rerank and self._reranker:
            results = self._reranker.rerank(query, results, top_k)

        return results

    def get_context(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_score: float = 0.0,
        mode: str = "auto",
        context_query: str | None = None,
        rerank: bool = False,
    ) -> ContextBundle:
        """Search and assemble a ContextBundle ready for prompt injection."""
        results = self.search(
            query=query,
            filters=filters,
            top_k=top_k,
            memory_types=memory_types,
            min_score=min_score,
            mode=mode,
            context_query=context_query,
            rerank=rerank,
        )

        total_chars = sum(len(r.memory.content) for r in results)
        token_estimate = total_chars // 4

        return ContextBundle(
            memories=results,
            token_estimate=token_estimate,
        )
