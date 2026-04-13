"""Main MemoryRuntime — the public API surface."""

from __future__ import annotations

import time
from typing import Any

from milvus_cortex.config import CortexConfig
from milvus_cortex.embedding.base import EmbeddingProvider
from milvus_cortex.embedding.fake import FakeEmbedding
from milvus_cortex.embedding.openai import OpenAIEmbedding
from milvus_cortex.embedding.sparse import text_to_sparse
from milvus_cortex.extraction.base import MemoryExtractor
from milvus_cortex.extraction.llm import LLMExtractor
from milvus_cortex.graph.engine import GraphEngine
from milvus_cortex.lifecycle.manager import LifecycleManager
from milvus_cortex.models import (
    CollectionHealth,
    ContextBundle,
    Entity,
    Memory,
    MemoryStats,
    MemoryType,
    Message,
    Relationship,
    SearchResult,
)
from milvus_cortex.observability import ObservabilityManager
from milvus_cortex.retrieval.orchestrator import RetrievalOrchestrator
from milvus_cortex.storage.milvus import MilvusStorage


def _build_embedder(config: CortexConfig) -> EmbeddingProvider:
    if config.embedding.provider == "fake":
        return FakeEmbedding(config.embedding)
    return OpenAIEmbedding(config.embedding)


def _build_extractor(config: CortexConfig) -> MemoryExtractor | None:
    if config.extraction.provider == "none":
        return None
    return LLMExtractor(config.extraction)


class MemoryRuntime:
    """Developer-facing API for the Milvus memory runtime.

    Supports hybrid search (dense+sparse), multi-vector representations,
    graph-on-Milvus (entity/relationship memory), memory consolidation,
    and Milvus-native observability.
    """

    def __init__(
        self,
        config: CortexConfig,
        storage: MilvusStorage,
        embedder: EmbeddingProvider,
        extractor: MemoryExtractor | None,
        retrieval: RetrievalOrchestrator,
        lifecycle: LifecycleManager,
        graph: GraphEngine | None,
        observability: ObservabilityManager,
    ) -> None:
        self._config = config
        self._storage = storage
        self._embedder = embedder
        self._extractor = extractor
        self._retrieval = retrieval
        self._lifecycle = lifecycle
        self._graph = graph
        self._observability = observability

    @classmethod
    def from_config(cls, config: CortexConfig | None = None) -> MemoryRuntime:
        """Build a fully-wired runtime from config."""
        config = config or CortexConfig()
        storage = MilvusStorage(config)
        storage.initialize()
        embedder = _build_embedder(config)
        extractor = _build_extractor(config)
        retrieval = RetrievalOrchestrator(storage, embedder, config.hybrid_search, config.multi_vector)
        lifecycle = LifecycleManager(storage, embedder, config.lifecycle)

        graph = None
        if config.graph.enabled:
            graph = GraphEngine(storage, embedder, config.graph)
            graph.initialize()

        observability = ObservabilityManager(storage)

        return cls(
            config=config,
            storage=storage,
            embedder=embedder,
            extractor=extractor,
            retrieval=retrieval,
            lifecycle=lifecycle,
            graph=graph,
            observability=observability,
        )

    def close(self) -> None:
        self._storage.close()

    def __enter__(self) -> MemoryRuntime:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core API: remember
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        workspace_id: str | None = None,
        memory_type: str | MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        expires_at: float | None = None,
        context: str | None = None,  # Multi-vector: surrounding context text
    ) -> Memory:
        """Store a single memory with automatic embedding, dedup, and TTL."""
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        embedding = self._embedder.embed_one(content)

        # Multi-vector: context embedding
        context_embedding = None
        if self._config.multi_vector.enabled and context:
            context_embedding = self._embedder.embed_one(context)

        # Hybrid search: sparse embedding
        sparse_embedding = None
        if self._config.hybrid_search.enabled:
            sparse_embedding = text_to_sparse(content)

        memory = Memory(
            content=content,
            memory_type=memory_type,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            embedding=embedding,
            context_embedding=context_embedding,
            sparse_embedding=sparse_embedding,
            importance=importance,
            metadata=metadata or {},
            source="manual",
            expires_at=expires_at,
        )
        memory = self._lifecycle.apply_ttl(memory)

        # Dedup check
        scope_filters = self._scope_filters(
            app_id=app_id, user_id=user_id, session_id=session_id,
            agent_id=agent_id, workspace_id=workspace_id,
        )
        existing = self._lifecycle.check_dedup(memory, scope_filters)
        if existing:
            if importance > existing.importance:
                existing.content = content
                existing.embedding = embedding
                existing.context_embedding = context_embedding
                existing.sparse_embedding = sparse_embedding
                existing.importance = importance
                existing.metadata.update(metadata or {})
                self._storage.update(existing)
            return existing

        self._storage.insert([memory])
        return memory

    # ------------------------------------------------------------------
    # Core API: ingest messages
    # ------------------------------------------------------------------

    def ingest_messages(
        self,
        messages: list[dict | Message],
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        workspace_id: str | None = None,
        extract: bool = True,
    ) -> list[Memory]:
        """Ingest a conversation with automatic embedding, extraction, and graph updates."""
        parsed: list[Message] = []
        for m in messages:
            if isinstance(m, dict):
                parsed.append(Message(**m))
            else:
                parsed.append(m)

        stored: list[Memory] = []

        conversation_text = "\n".join(f"{m.role}: {m.content}" for m in parsed)
        episodic = self.remember(
            content=conversation_text,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            memory_type=MemoryType.EPISODIC,
            metadata={"message_count": len(parsed)},
        )
        stored.append(episodic)

        # Extract durable memories
        if extract and self._extractor:
            extracted = self._extractor.extract_from_messages(parsed)
            for mem in extracted:
                mem.app_id = app_id
                mem.user_id = user_id
                mem.session_id = session_id
                mem.agent_id = agent_id
                mem.workspace_id = workspace_id
                mem.embedding = self._embedder.embed_one(mem.content)
                if self._config.hybrid_search.enabled:
                    mem.sparse_embedding = text_to_sparse(mem.content)
                mem = self._lifecycle.apply_ttl(mem)
                scope_filters = self._scope_filters(
                    app_id=app_id, user_id=user_id, session_id=session_id,
                    agent_id=agent_id, workspace_id=workspace_id,
                )
                existing = self._lifecycle.check_dedup(mem, scope_filters)
                if not existing:
                    self._storage.insert([mem])
                    stored.append(mem)

        return stored

    # ------------------------------------------------------------------
    # Core API: search (with hybrid/multi-vector modes)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        workspace_id: str | None = None,
        memory_types: list[str | MemoryType] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        mode: str = "auto",  # "dense", "hybrid", "multi_vector", "auto"
        context_query: str | None = None,  # For multi_vector mode
    ) -> list[SearchResult]:
        """Search for relevant memories with optional hybrid or multi-vector mode."""
        filters = self._scope_filters(
            app_id=app_id, user_id=user_id, session_id=session_id,
            agent_id=agent_id, workspace_id=workspace_id,
        )
        types = None
        if memory_types:
            types = [MemoryType(t) if isinstance(t, str) else t for t in memory_types]

        t0 = time.time()
        results = self._retrieval.search(
            query=query, filters=filters, top_k=top_k,
            memory_types=types, min_score=min_score,
            mode=mode, context_query=context_query,
        )
        latency_ms = (time.time() - t0) * 1000
        self._observability.record_search_latency(latency_ms)

        return results

    # ------------------------------------------------------------------
    # Core API: get_context
    # ------------------------------------------------------------------

    def get_context(
        self,
        query: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        workspace_id: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        mode: str = "auto",
    ) -> ContextBundle:
        """Retrieve and assemble context for prompt injection."""
        filters = self._scope_filters(
            app_id=app_id, user_id=user_id, session_id=session_id,
            agent_id=agent_id, workspace_id=workspace_id,
        )
        return self._retrieval.get_context(
            query=query, filters=filters, top_k=top_k,
            min_score=min_score, mode=mode,
        )

    # ------------------------------------------------------------------
    # Core API: forget
    # ------------------------------------------------------------------

    def forget(
        self,
        memory_id: str | None = None,
        *,
        memory_ids: list[str] | None = None,
    ) -> int:
        ids = list(memory_ids or [])
        if memory_id:
            ids.append(memory_id)
        return self._lifecycle.forget(ids)

    # ------------------------------------------------------------------
    # Core API: get, list, count
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Memory | None:
        return self._storage.get(memory_id)

    def list_memories(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        filters = self._scope_filters(app_id=app_id, user_id=user_id)
        return self._storage.list_memories(filters=filters, limit=limit)

    def count(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        filters = self._scope_filters(app_id=app_id, user_id=user_id)
        return self._storage.count(filters=filters)

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def expire(self, **scope: str | None) -> int:
        filters = self._scope_filters(**scope) if scope else None
        return self._lifecycle.expire_memories(filters=filters)

    def merge(self, memory_ids: list[str], merged_content: str) -> Memory | None:
        return self._lifecycle.merge_memories(memory_ids, merged_content)

    def consolidate(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        similarity_threshold: float | None = None,
        min_cluster_size: int | None = None,
    ) -> list[Memory]:
        """Cluster and merge related memories into consolidated versions."""
        filters = self._scope_filters(app_id=app_id, user_id=user_id)
        return self._lifecycle.consolidate(
            filters=filters or None,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )

    # ------------------------------------------------------------------
    # Graph-on-Milvus API
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        """Add an entity to the knowledge graph (with automatic resolution)."""
        if not self._graph:
            raise RuntimeError("Graph is not enabled. Set config.graph.enabled = True")
        return self._graph.add_entity(
            name=name, entity_type=entity_type, description=description,
            app_id=app_id, user_id=user_id, metadata=metadata,
        )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        *,
        app_id: str | None = None,
        user_id: str | None = None,
    ) -> Relationship:
        """Add a relationship between two entities."""
        if not self._graph:
            raise RuntimeError("Graph is not enabled. Set config.graph.enabled = True")
        return self._graph.add_relationship(
            source_id=source_id, target_id=target_id,
            relation_type=relation_type, description=description,
            app_id=app_id, user_id=user_id,
        )

    def get_relationships(self, entity_id: str, direction: str = "both") -> list[Relationship]:
        """Get relationships for an entity."""
        if not self._graph:
            raise RuntimeError("Graph is not enabled. Set config.graph.enabled = True")
        return self._graph.get_relationships(entity_id, direction=direction)

    def graph_search(
        self,
        query: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 5,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Search the knowledge graph for entities and their neighborhoods."""
        if not self._graph:
            raise RuntimeError("Graph is not enabled. Set config.graph.enabled = True")
        return self._graph.graph_search(
            query=query, app_id=app_id, user_id=user_id,
            top_k=top_k, depth=depth,
        )

    # ------------------------------------------------------------------
    # Observability API
    # ------------------------------------------------------------------

    def stats(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
    ) -> MemoryStats:
        """Get memory statistics."""
        return self._observability.get_stats(app_id=app_id, user_id=user_id)

    def health(self) -> CollectionHealth:
        """Get Milvus collection health status."""
        return self._observability.get_health()

    def search_diagnostics(self) -> dict:
        """Get search performance diagnostics."""
        return self._observability.get_search_diagnostics()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_filters(**kwargs: str | None) -> dict[str, str]:
        return {k: v for k, v in kwargs.items() if v is not None}
