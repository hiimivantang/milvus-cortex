"""Retrieval orchestration — search, filter, rank, assemble context."""

from __future__ import annotations

import time

from milvus_cortex.embedding.base import EmbeddingProvider
from milvus_cortex.models import ContextBundle, MemoryType, SearchResult
from milvus_cortex.storage.milvus import MilvusStorage


class RetrievalOrchestrator:
    """Coordinates search across storage with embedding + filtering + ranking."""

    def __init__(self, storage: MilvusStorage, embedder: EmbeddingProvider) -> None:
        self._storage = storage
        self._embedder = embedder

    def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Vector search with optional type/scope filtering."""
        query_embedding = self._embedder.embed_one(query)

        combined_filters = dict(filters) if filters else {}
        if memory_types and len(memory_types) == 1:
            combined_filters["memory_type"] = memory_types[0].value

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

        return results

    def get_context(
        self,
        query: str,
        filters: dict | None = None,
        top_k: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_score: float = 0.0,
    ) -> ContextBundle:
        """Search and assemble a ContextBundle ready for prompt injection."""
        results = self.search(
            query=query,
            filters=filters,
            top_k=top_k,
            memory_types=memory_types,
            min_score=min_score,
        )

        # Rough token estimate: ~4 chars per token
        total_chars = sum(len(r.memory.content) for r in results)
        token_estimate = total_chars // 4

        return ContextBundle(
            memories=results,
            token_estimate=token_estimate,
        )
