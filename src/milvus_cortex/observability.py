"""Milvus-native observability — stats, health, and diagnostics."""

from __future__ import annotations

import time
from collections import defaultdict

from milvus_cortex.models import CollectionHealth, MemoryStats
from milvus_cortex.storage.milvus import MilvusStorage


class ObservabilityManager:
    """Provides stats, health checks, and diagnostics for the memory store."""

    def __init__(self, storage: MilvusStorage) -> None:
        self._storage = storage
        self._search_latencies: list[float] = []
        self._max_latency_samples = 100

    def record_search_latency(self, latency_ms: float) -> None:
        """Record a search latency measurement."""
        self._search_latencies.append(latency_ms)
        if len(self._search_latencies) > self._max_latency_samples:
            self._search_latencies = self._search_latencies[-self._max_latency_samples:]

    def get_stats(
        self,
        app_id: str | None = None,
        user_id: str | None = None,
    ) -> MemoryStats:
        """Get memory statistics, optionally scoped to app/user."""
        filters = {}
        if app_id:
            filters["app_id"] = app_id
        if user_id:
            filters["user_id"] = user_id

        try:
            memories = self._storage.list_memories(filters=filters or None, limit=10000)
        except Exception:
            memories = []

        if not memories:
            return MemoryStats(
                total_entities=self._storage.collection_row_count(self._storage.entity_collection_name),
                total_relationships=self._storage.collection_row_count(self._storage.relationship_collection_name),
            )

        by_type: dict[str, int] = defaultdict(int)
        by_scope: dict[str, int] = defaultdict(int)
        importance_sum = 0.0
        oldest = float("inf")
        newest = 0.0

        for m in memories:
            by_type[m.memory_type.value] += 1
            if m.app_id:
                by_scope[f"app:{m.app_id}"] += 1
            if m.user_id:
                by_scope[f"user:{m.user_id}"] += 1
            importance_sum += m.importance
            if m.created_at < oldest:
                oldest = m.created_at
            if m.created_at > newest:
                newest = m.created_at

        return MemoryStats(
            total_memories=len(memories),
            by_type=dict(by_type),
            by_scope=dict(by_scope),
            oldest_memory=oldest if oldest != float("inf") else None,
            newest_memory=newest if newest > 0 else None,
            avg_importance=importance_sum / len(memories) if memories else 0.0,
            total_entities=self._storage.collection_row_count(self._storage.entity_collection_name),
            total_relationships=self._storage.collection_row_count(self._storage.relationship_collection_name),
        )

    def get_health(self) -> CollectionHealth:
        """Get health status of the Milvus collections."""
        return CollectionHealth(
            collection_name=self._storage.memory_collection_name,
            row_count=self._storage.collection_row_count(self._storage.memory_collection_name),
            index_status="ready",
            entity_collection_rows=self._storage.collection_row_count(self._storage.entity_collection_name),
            relationship_collection_rows=self._storage.collection_row_count(self._storage.relationship_collection_name),
        )

    def get_search_diagnostics(self) -> dict:
        """Get search performance diagnostics."""
        if not self._search_latencies:
            return {"sample_count": 0}

        latencies = sorted(self._search_latencies)
        n = len(latencies)
        return {
            "sample_count": n,
            "avg_ms": sum(latencies) / n,
            "p50_ms": latencies[n // 2],
            "p95_ms": latencies[int(n * 0.95)],
            "p99_ms": latencies[int(n * 0.99)],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
        }
