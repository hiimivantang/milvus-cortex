"""Memory lifecycle management — dedup, merge, expiry, forget, consolidation."""

from __future__ import annotations

import time
from collections import defaultdict

from milvus_cortex.config import LifecycleConfig
from milvus_cortex.embedding.base import EmbeddingProvider
from milvus_cortex.models import Memory, MemoryType
from milvus_cortex.storage.milvus import MilvusStorage


class LifecycleManager:
    """Handles memory lifecycle: deduplication, expiry, forgetting, and consolidation."""

    def __init__(
        self,
        storage: MilvusStorage,
        embedder: EmbeddingProvider,
        config: LifecycleConfig,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._config = config

    def apply_ttl(self, memory: Memory) -> Memory:
        """Apply default TTL if the memory doesn't already have an expiry."""
        if memory.expires_at is not None:
            return memory

        if memory.memory_type == MemoryType.WORKING:
            ttl = self._config.working_memory_ttl_seconds
        else:
            ttl = self._config.default_ttl_seconds

        if ttl is not None:
            memory.expires_at = memory.created_at + ttl
        return memory

    def check_dedup(self, memory: Memory, scope_filters: dict) -> Memory | None:
        """Check if a near-duplicate already exists.

        Returns the existing memory if a duplicate is found, None otherwise.
        """
        if not self._config.auto_dedup or not memory.embedding:
            return None

        candidates = self._storage.search(
            embedding=memory.embedding,
            filters=scope_filters,
            top_k=3,
        )

        for result in candidates:
            if result.score >= self._config.dedup_threshold:
                return result.memory
        return None

    def expire_memories(self, filters: dict | None = None) -> int:
        """Delete all expired memories. Returns count of deleted memories."""
        if not self._config.auto_expire:
            return 0

        now = time.time()
        all_memories = self._storage.list_memories(filters=filters, limit=10000)
        expired_ids = [
            m.id for m in all_memories
            if m.expires_at is not None and m.expires_at <= now
        ]
        if expired_ids:
            return self._storage.delete(expired_ids)
        return 0

    def forget(self, memory_ids: list[str]) -> int:
        """Explicitly delete specific memories."""
        if not memory_ids:
            return 0
        return self._storage.delete(memory_ids)

    def merge_memories(self, memory_ids: list[str], merged_content: str) -> Memory | None:
        """Merge multiple memories into one. Deletes originals, returns the new memory."""
        originals: list[Memory] = []
        for mid in memory_ids:
            m = self._storage.get(mid)
            if m:
                originals.append(m)

        if len(originals) < 2:
            return None

        embedding = self._embedder.embed_one(merged_content)
        merged = Memory(
            content=merged_content,
            memory_type=originals[0].memory_type,
            app_id=originals[0].app_id,
            user_id=originals[0].user_id,
            session_id=originals[0].session_id,
            agent_id=originals[0].agent_id,
            workspace_id=originals[0].workspace_id,
            embedding=embedding,
            importance=max(m.importance for m in originals),
            source="merge",
            created_at=min(m.created_at for m in originals),
        )
        merged = self.apply_ttl(merged)

        self._storage.delete([m.id for m in originals])
        self._storage.insert([merged])
        return merged

    # ------------------------------------------------------------------
    # Consolidation pipeline
    # ------------------------------------------------------------------

    def consolidate(
        self,
        filters: dict | None = None,
        similarity_threshold: float | None = None,
        min_cluster_size: int | None = None,
    ) -> list[Memory]:
        """Cluster related memories and merge them into consolidated versions.

        1. Fetch all memories in scope
        2. For each memory, find near-duplicates (cluster by vector similarity)
        3. Merge clusters that meet the minimum size
        4. Return the new consolidated memories

        Does NOT use LLM — merges by concatenation with dedup.
        For LLM-summarized consolidation, use merge_memories() directly.
        """
        threshold = similarity_threshold or self._config.consolidation_threshold
        min_size = min_cluster_size or self._config.consolidation_min_cluster

        all_memories = self._storage.list_memories(filters=filters, limit=10000)
        if len(all_memories) < min_size:
            return []

        # Build clusters via greedy nearest-neighbor
        clustered: set[str] = set()
        clusters: list[list[Memory]] = []

        for memory in all_memories:
            if memory.id in clustered or not memory.embedding:
                continue

            # Find similar memories
            candidates = self._storage.search(
                embedding=memory.embedding,
                filters=filters,
                top_k=20,
            )

            cluster = [memory]
            clustered.add(memory.id)

            for result in candidates:
                if result.memory.id in clustered:
                    continue
                if result.score >= threshold:
                    cluster.append(result.memory)
                    clustered.add(result.memory.id)

            if len(cluster) >= min_size:
                clusters.append(cluster)

        # Merge each cluster
        consolidated: list[Memory] = []
        for cluster in clusters:
            # Sort by importance (highest first), then by recency
            cluster.sort(key=lambda m: (-m.importance, -m.created_at))

            # Build consolidated content: keep unique content
            seen_content: set[str] = set()
            parts: list[str] = []
            for m in cluster:
                normalized = m.content.strip().lower()
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    parts.append(m.content)

            merged_content = " | ".join(parts)
            merged = self.merge_memories(
                memory_ids=[m.id for m in cluster],
                merged_content=merged_content,
            )
            if merged:
                consolidated.append(merged)

        return consolidated
