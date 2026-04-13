"""Milvus storage adapter."""

from __future__ import annotations

import json
import time
from typing import Any

from pymilvus import DataType, MilvusClient

from milvus_cortex.config import MilvusConfig, EmbeddingConfig
from milvus_cortex.models import Memory, MemoryType, SearchResult


# Scope fields that can be used for filtering.
SCOPE_FIELDS = ("app_id", "user_id", "session_id", "agent_id", "workspace_id")


class MilvusStorage:
    """Milvus-backed storage for memories.

    Uses a minimal explicit schema (id + embedding) with dynamic fields
    for all other memory attributes. This approach is compatible across
    Milvus Lite, standalone, and cloud deployments.
    """

    def __init__(self, milvus_config: MilvusConfig, embedding_config: EmbeddingConfig) -> None:
        self._cfg = milvus_config
        self._dim = embedding_config.dimensions
        self._collection_name = f"{milvus_config.collection_prefix}_memories"
        self._client: MilvusClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        connect_params: dict[str, Any] = {"uri": self._cfg.uri}
        if self._cfg.token:
            connect_params["token"] = self._cfg.token
        if self._cfg.db_name != "default":
            connect_params["db_name"] = self._cfg.db_name
        self._client = MilvusClient(**connect_params)
        self._ensure_collection()

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def insert(self, memories: list[Memory]) -> list[str]:
        rows = [self._memory_to_row(m) for m in memories]
        self._client.insert(collection_name=self._collection_name, data=rows)
        return [m.id for m in memories]

    def get(self, memory_id: str) -> Memory | None:
        results = self._client.get(
            collection_name=self._collection_name,
            ids=[memory_id],
            output_fields=["*"],
        )
        if not results:
            return None
        return self._row_to_memory(results[0])

    def update(self, memory: Memory) -> None:
        memory.updated_at = time.time()
        row = self._memory_to_row(memory)
        self._client.upsert(collection_name=self._collection_name, data=[row])

    def delete(self, memory_ids: list[str]) -> int:
        result = self._client.delete(
            collection_name=self._collection_name,
            ids=memory_ids,
        )
        if isinstance(result, dict):
            return result.get("delete_count", len(memory_ids))
        return len(memory_ids)

    # ------------------------------------------------------------------
    # Search & query
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        filter_expr = self._build_filter_expr(filters) if filters else ""

        results = self._client.search(
            collection_name=self._collection_name,
            data=[embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=top_k,
            filter=filter_expr or None,
            output_fields=["*"],
        )

        search_results: list[SearchResult] = []
        if results:
            for hit in results[0]:
                entity = hit.get("entity", hit)
                entity["id"] = hit.get("id", entity.get("id"))
                memory = self._row_to_memory(entity)
                score = hit.get("distance", 0.0)
                search_results.append(SearchResult(memory=memory, score=score))
        return search_results

    def list_memories(
        self,
        filters: dict | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        filter_expr = self._build_filter_expr(filters) if filters else ""
        results = self._client.query(
            collection_name=self._collection_name,
            filter=filter_expr or "id != ''",
            output_fields=["*"],
            limit=limit,
            offset=offset,
        )
        return [self._row_to_memory(r) for r in results]

    def count(self, filters: dict | None = None) -> int:
        filter_expr = self._build_filter_expr(filters) if filters else "id != ''"
        results = self._client.query(
            collection_name=self._collection_name,
            filter=filter_expr,
            output_fields=["count(*)"],
        )
        if results:
            return results[0].get("count(*)", 0)
        return 0

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self._collection_name):
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    def _memory_to_row(self, memory: Memory) -> dict[str, Any]:
        return {
            "id": memory.id,
            "embedding": memory.embedding or [0.0] * self._dim,
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            **{scope: getattr(memory, scope) or "" for scope in SCOPE_FIELDS},
            "metadata_json": json.dumps(memory.metadata),
            "importance": memory.importance,
            "source": memory.source or "",
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "expires_at": memory.expires_at or 0.0,
        }

    def _row_to_memory(self, row: dict[str, Any]) -> Memory:
        metadata = row.get("metadata_json", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        expires_at = row.get("expires_at", 0.0)

        return Memory(
            id=row["id"],
            content=row.get("content", ""),
            memory_type=MemoryType(row.get("memory_type", "semantic")),
            **{scope: row.get(scope) or None for scope in SCOPE_FIELDS},
            embedding=row.get("embedding"),
            metadata=metadata,
            importance=row.get("importance", 0.5),
            source=row.get("source") or None,
            created_at=row.get("created_at", 0.0),
            updated_at=row.get("updated_at", 0.0),
            expires_at=expires_at if expires_at else None,
        )

    # ------------------------------------------------------------------
    # Filter builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter_expr(filters: dict[str, Any]) -> str:
        parts: list[str] = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, str):
                escaped = value.replace("'", "\\'")
                parts.append(f"{key} == '{escaped}'")
            elif isinstance(value, (int, float)):
                parts.append(f"{key} == {value}")
        return " and ".join(parts) if parts else ""
