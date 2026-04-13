"""Main MemoryRuntime — the public API surface."""

from __future__ import annotations

import time
from typing import Any

from milvus_cortex.config import CortexConfig
from milvus_cortex.embedding.base import EmbeddingProvider
from milvus_cortex.embedding.fake import FakeEmbedding
from milvus_cortex.embedding.openai import OpenAIEmbedding
from milvus_cortex.extraction.base import MemoryExtractor
from milvus_cortex.extraction.llm import LLMExtractor
from milvus_cortex.lifecycle.manager import LifecycleManager
from milvus_cortex.models import (
    ContextBundle,
    Memory,
    MemoryType,
    Message,
    SearchResult,
)
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

    Usage::

        runtime = MemoryRuntime.from_config(CortexConfig())
        runtime.remember(app_id="myapp", user_id="u1", content="User likes Python")
        results = runtime.search(app_id="myapp", user_id="u1", query="language preferences")
    """

    def __init__(
        self,
        config: CortexConfig,
        storage: MilvusStorage,
        embedder: EmbeddingProvider,
        extractor: MemoryExtractor | None,
        retrieval: RetrievalOrchestrator,
        lifecycle: LifecycleManager,
    ) -> None:
        self._config = config
        self._storage = storage
        self._embedder = embedder
        self._extractor = extractor
        self._retrieval = retrieval
        self._lifecycle = lifecycle

    @classmethod
    def from_config(cls, config: CortexConfig | None = None) -> MemoryRuntime:
        """Build a fully-wired runtime from config."""
        config = config or CortexConfig()
        storage = MilvusStorage(config.milvus, config.embedding)
        storage.initialize()
        embedder = _build_embedder(config)
        extractor = _build_extractor(config)
        retrieval = RetrievalOrchestrator(storage, embedder)
        lifecycle = LifecycleManager(storage, embedder, config.lifecycle)
        return cls(
            config=config,
            storage=storage,
            embedder=embedder,
            extractor=extractor,
            retrieval=retrieval,
            lifecycle=lifecycle,
        )

    def close(self) -> None:
        """Release storage connections."""
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
    ) -> Memory:
        """Store a single memory."""
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        embedding = self._embedder.embed_one(content)
        memory = Memory(
            content=content,
            memory_type=memory_type,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            embedding=embedding,
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
            # Update existing memory if the new one is more important
            if importance > existing.importance:
                existing.content = content
                existing.embedding = embedding
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
        """Ingest a conversation. Optionally extract durable memories.

        Args:
            messages: List of Message objects or dicts with role/content keys.
            extract: If True and an extractor is configured, automatically
                extract durable memories from the conversation.

        Returns:
            List of memories that were stored (episodic + any extracted).
        """
        parsed: list[Message] = []
        for m in messages:
            if isinstance(m, dict):
                parsed.append(Message(**m))
            else:
                parsed.append(m)

        stored: list[Memory] = []

        # Store the full conversation as an episodic memory
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
    # Core API: search
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
    ) -> list[SearchResult]:
        """Search for relevant memories."""
        filters = self._scope_filters(
            app_id=app_id, user_id=user_id, session_id=session_id,
            agent_id=agent_id, workspace_id=workspace_id,
        )
        types = None
        if memory_types:
            types = [
                MemoryType(t) if isinstance(t, str) else t
                for t in memory_types
            ]
        return self._retrieval.search(
            query=query, filters=filters, top_k=top_k,
            memory_types=types, min_score=min_score,
        )

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
    ) -> ContextBundle:
        """Retrieve and assemble context for prompt injection."""
        filters = self._scope_filters(
            app_id=app_id, user_id=user_id, session_id=session_id,
            agent_id=agent_id, workspace_id=workspace_id,
        )
        return self._retrieval.get_context(
            query=query, filters=filters, top_k=top_k, min_score=min_score,
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
        """Explicitly forget one or more memories."""
        ids = list(memory_ids or [])
        if memory_id:
            ids.append(memory_id)
        return self._lifecycle.forget(ids)

    # ------------------------------------------------------------------
    # Core API: get, list, count
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID."""
        return self._storage.get(memory_id)

    def list_memories(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """List memories matching scope filters."""
        filters = self._scope_filters(app_id=app_id, user_id=user_id)
        return self._storage.list_memories(filters=filters, limit=limit)

    def count(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Count memories matching scope filters."""
        filters = self._scope_filters(app_id=app_id, user_id=user_id)
        return self._storage.count(filters=filters)

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    def expire(self, **scope: str | None) -> int:
        """Run expiry sweep, removing memories past their TTL."""
        filters = self._scope_filters(**scope) if scope else None
        return self._lifecycle.expire_memories(filters=filters)

    def merge(self, memory_ids: list[str], merged_content: str) -> Memory | None:
        """Merge multiple memories into one with the given content."""
        return self._lifecycle.merge_memories(memory_ids, merged_content)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_filters(**kwargs: str | None) -> dict[str, str]:
        return {k: v for k, v in kwargs.items() if v is not None}
