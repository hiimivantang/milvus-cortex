"""Memory lifecycle management — dedup, merge, expiry, forget, consolidation."""

from __future__ import annotations

import math
import time

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
        Uses both exact content matching and vector similarity with a
        token-overlap guard to prevent false deduplication of semantically
        similar but distinct memories.
        """
        if not self._config.auto_dedup or not memory.embedding:
            return None

        candidates = self._storage.search(
            embedding=memory.embedding,
            filters=scope_filters,
            top_k=3,
        )

        normalized_new = memory.content.strip().lower()
        new_tokens = set(normalized_new.split())

        for result in candidates:
            if result.score < self._config.dedup_threshold:
                continue
            existing_normalized = result.memory.content.strip().lower()
            # Exact content match — always a duplicate
            if existing_normalized == normalized_new:
                return result.memory
            # Token-overlap guard: require sufficient Jaccard similarity
            # to prevent false dedup of semantically similar but distinct content
            # (e.g. "User prefers Python" vs "User uses Python at work")
            existing_tokens = set(existing_normalized.split())
            union = new_tokens | existing_tokens
            if union:
                overlap = len(new_tokens & existing_tokens) / len(union)
                if overlap >= self._config.dedup_content_threshold:
                    return result.memory
        return None

    def expire_memories(self, filters: dict | None = None) -> int:
        """Delete all expired memories using server-side scalar filtering."""
        if not self._config.auto_expire:
            return 0
        return self._storage.delete_expired(time.time(), filters)

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

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors in-memory."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def consolidate(
        self,
        filters: dict | None = None,
        similarity_threshold: float | None = None,
        min_cluster_size: int | None = None,
    ) -> list[Memory]:
        """Cluster related memories and merge them into consolidated versions.

        1. Fetch all memories in scope
        2. Cluster by in-memory pairwise cosine similarity (no O(n) search calls)
        3. Merge clusters that meet the minimum size
        4. Return the new consolidated memories

        Does NOT use LLM — merges by structured dedup.
        For LLM-summarized consolidation, use merge_memories() directly.
        """
        threshold = similarity_threshold or self._config.consolidation_threshold
        min_size = min_cluster_size or self._config.consolidation_min_cluster

        # Cap at 2000 to keep in-memory pairwise similarity tractable
        all_memories = self._storage.list_memories(filters=filters, limit=2000)
        if len(all_memories) < min_size:
            return []

        # Filter to memories with embeddings
        memories = [m for m in all_memories if m.embedding]
        if len(memories) < min_size:
            return []

        # Build clusters via in-memory pairwise cosine similarity.
        # Avoids O(n) Milvus network round-trips. Capped at 2000 memories
        # to keep O(n²) in-memory computation fast on high-dim vectors.
        clustered: set[str] = set()
        clusters: list[list[Memory]] = []

        for i, memory in enumerate(memories):
            if memory.id in clustered:
                continue

            cluster = [memory]
            clustered.add(memory.id)

            for j in range(i + 1, len(memories)):
                candidate = memories[j]
                if candidate.id in clustered:
                    continue
                sim = self._cosine_similarity(memory.embedding, candidate.embedding)
                if sim >= threshold:
                    cluster.append(candidate)
                    clustered.add(candidate.id)

            if len(cluster) >= min_size:
                clusters.append(cluster)

        # Merge each cluster
        consolidated: list[Memory] = []
        for cluster in clusters:
            # Sort by importance (highest first), then by recency
            cluster.sort(key=lambda m: (-m.importance, -m.created_at))

            # Build consolidated content: keep unique content, join with newlines
            seen_content: set[str] = set()
            parts: list[str] = []
            for m in cluster:
                normalized = m.content.strip().lower()
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    parts.append(m.content)

            merged_content = "\n".join(parts)
            merged = self.merge_memories(
                memory_ids=[m.id for m in cluster],
                merged_content=merged_content,
            )
            if merged:
                consolidated.append(merged)

        return consolidated
