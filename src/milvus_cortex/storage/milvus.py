"""Milvus storage adapter with hybrid search, multi-vector, and partition key support."""

from __future__ import annotations

import json
import time
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from milvus_cortex.config import CortexConfig, HybridSearchConfig, MilvusConfig, EmbeddingConfig, MultiVectorConfig
from milvus_cortex.models import (
    Entity,
    Memory,
    MemoryType,
    Relationship,
    SearchResult,
)

SCOPE_FIELDS = ("app_id", "user_id", "session_id", "agent_id", "workspace_id")


class MilvusStorage:
    """Milvus-backed storage with hybrid search, multi-vector, and partition key support."""

    def __init__(self, config: CortexConfig) -> None:
        self._milvus_cfg = config.milvus
        self._dim = config.embedding.dimensions
        self._ctx_dim = config.multi_vector.context_dimensions or config.embedding.dimensions
        self._hybrid_cfg = config.hybrid_search
        self._multi_vec = config.multi_vector
        self._use_partition_key = config.milvus.use_partition_key
        self._collection_name = f"{config.milvus.collection_prefix}_memories"
        self._entity_collection = f"{config.milvus.collection_prefix}_entities"
        self._rel_collection = f"{config.milvus.collection_prefix}_relationships"
        self._client: MilvusClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        params: dict[str, Any] = {"uri": self._milvus_cfg.uri}
        if self._milvus_cfg.token:
            params["token"] = self._milvus_cfg.token
        if self._milvus_cfg.db_name != "default":
            params["db_name"] = self._milvus_cfg.db_name
        self._client = MilvusClient(**params)
        self._ensure_memory_collection()

    def initialize_graph_collections(self) -> None:
        """Create entity and relationship collections for graph-on-Milvus."""
        self._ensure_entity_collection()
        self._ensure_relationship_collection()

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Memory CRUD
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
        result = self._client.delete(collection_name=self._collection_name, ids=memory_ids)
        if isinstance(result, dict):
            return result.get("delete_count", len(memory_ids))
        return len(memory_ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Dense-only vector search."""
        filter_expr = self._build_filter_expr(filters) if filters else None
        results = self._client.search(
            collection_name=self._collection_name,
            data=[embedding],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            filter=filter_expr,
            output_fields=["*"],
        )
        return self._parse_search_results(results)

    def hybrid_search(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict[int, float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Hybrid dense+sparse search using RRF fusion."""
        filter_expr = self._build_filter_expr(filters) if filters else ""

        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=filter_expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k,
            expr=filter_expr or None,
        )

        results = self._client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=self._hybrid_cfg.rrf_k),
            limit=top_k,
            output_fields=["*"],
        )
        return self._parse_search_results(results)

    def multi_vector_search(
        self,
        content_embedding: list[float],
        context_embedding: list[float],
        content_weight: float = 0.7,
        context_weight: float = 0.3,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search across content and context vectors with weighted RRF fusion."""
        filter_expr = self._build_filter_expr(filters) if filters else ""

        content_req = AnnSearchRequest(
            data=[content_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=filter_expr or None,
        )
        context_req = AnnSearchRequest(
            data=[context_embedding],
            anns_field="context_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=filter_expr or None,
        )

        results = self._client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[content_req, context_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["*"],
        )
        return self._parse_search_results(results)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

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
    # Entity CRUD (graph-on-Milvus)
    # ------------------------------------------------------------------

    def insert_entity(self, entity: Entity) -> str:
        row = {
            "id": entity.id,
            "embedding": entity.embedding or [0.0] * self._dim,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "app_id": entity.app_id or "",
            "user_id": entity.user_id or "",
            "metadata_json": json.dumps(entity.metadata),
            "created_at": entity.created_at,
        }
        self._client.insert(collection_name=self._entity_collection, data=[row])
        return entity.id

    def search_entities(
        self,
        embedding: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[tuple[Entity, float]]:
        filter_expr = self._build_filter_expr(filters) if filters else None
        results = self._client.search(
            collection_name=self._entity_collection,
            data=[embedding],
            anns_field="embedding",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            filter=filter_expr,
            output_fields=["*"],
        )
        entities: list[tuple[Entity, float]] = []
        if results:
            for hit in results[0]:
                entity_data = hit.get("entity", hit)
                entity_data["id"] = hit.get("id", entity_data.get("id"))
                metadata = entity_data.get("metadata_json", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                entity = Entity(
                    id=entity_data["id"],
                    name=entity_data.get("name", ""),
                    entity_type=entity_data.get("entity_type", ""),
                    description=entity_data.get("description", ""),
                    app_id=entity_data.get("app_id") or None,
                    user_id=entity_data.get("user_id") or None,
                    embedding=entity_data.get("embedding"),
                    metadata=metadata,
                    created_at=entity_data.get("created_at", 0.0),
                )
                entities.append((entity, hit.get("distance", 0.0)))
        return entities

    def get_entity(self, entity_id: str) -> Entity | None:
        results = self._client.get(
            collection_name=self._entity_collection,
            ids=[entity_id],
            output_fields=["*"],
        )
        if not results:
            return None
        row = results[0]
        metadata = row.get("metadata_json", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Entity(
            id=row["id"],
            name=row.get("name", ""),
            entity_type=row.get("entity_type", ""),
            description=row.get("description", ""),
            app_id=row.get("app_id") or None,
            user_id=row.get("user_id") or None,
            embedding=row.get("embedding"),
            metadata=metadata,
            created_at=row.get("created_at", 0.0),
        )

    def delete_entity(self, entity_id: str) -> None:
        self._client.delete(collection_name=self._entity_collection, ids=[entity_id])

    # ------------------------------------------------------------------
    # Relationship CRUD (graph-on-Milvus)
    # ------------------------------------------------------------------

    def insert_relationship(self, rel: Relationship) -> str:
        row = {
            "id": rel.id,
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "_vec": [0.0, 0.0, 0.0, 0.0],  # Dummy vector required by Milvus
            "relation_type": rel.relation_type,
            "description": rel.description,
            "weight": rel.weight,
            "app_id": rel.app_id or "",
            "user_id": rel.user_id or "",
            "metadata_json": json.dumps(rel.metadata),
            "created_at": rel.created_at,
        }
        self._client.insert(collection_name=self._rel_collection, data=[row])
        return rel.id

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> list[Relationship]:
        results: list[Relationship] = []
        if direction in ("outgoing", "both"):
            rows = self._client.query(
                collection_name=self._rel_collection,
                filter=f"source_id == '{entity_id}'",
                output_fields=["*"],
                limit=100,
            )
            results.extend(self._rows_to_relationships(rows))
        if direction in ("incoming", "both"):
            rows = self._client.query(
                collection_name=self._rel_collection,
                filter=f"target_id == '{entity_id}'",
                output_fields=["*"],
                limit=100,
            )
            results.extend(self._rows_to_relationships(rows))
        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)
        return unique

    def delete_relationship(self, rel_id: str) -> None:
        self._client.delete(collection_name=self._rel_collection, ids=[rel_id])

    # ------------------------------------------------------------------
    # Collection stats (observability)
    # ------------------------------------------------------------------

    def collection_row_count(self, collection_name: str | None = None) -> int:
        name = collection_name or self._collection_name
        try:
            if not self._client.has_collection(name):
                return 0
            results = self._client.query(
                collection_name=name,
                filter="id != ''",
                output_fields=["count(*)"],
            )
            return results[0].get("count(*)", 0) if results else 0
        except Exception:
            return 0

    @property
    def memory_collection_name(self) -> str:
        return self._collection_name

    @property
    def entity_collection_name(self) -> str:
        return self._entity_collection

    @property
    def relationship_collection_name(self) -> str:
        return self._rel_collection

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------

    def _ensure_memory_collection(self) -> None:
        if self._client.has_collection(self._collection_name):
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

        # Multi-vector: context embedding
        if self._multi_vec.enabled:
            schema.add_field("context_embedding", DataType.FLOAT_VECTOR, dim=self._ctx_dim)
            index_params.add_index(field_name="context_embedding", index_type="AUTOINDEX", metric_type="COSINE")

        # Hybrid search: sparse vector
        if self._hybrid_cfg.enabled:
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
            index_params.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _ensure_entity_collection(self) -> None:
        if self._client.has_collection(self._entity_collection):
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

        self._client.create_collection(
            collection_name=self._entity_collection,
            schema=schema,
            index_params=index_params,
        )

    def _ensure_relationship_collection(self) -> None:
        if self._client.has_collection(self._rel_collection):
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        # source_id and target_id as explicit fields for filtered queries
        schema.add_field("source_id", DataType.VARCHAR, max_length=64)
        schema.add_field("target_id", DataType.VARCHAR, max_length=64)
        # Dummy vector field required by Milvus — minimal dim
        schema.add_field("_vec", DataType.FLOAT_VECTOR, dim=4)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="_vec", index_type="AUTOINDEX", metric_type="COSINE")

        self._client.create_collection(
            collection_name=self._rel_collection,
            schema=schema,
            index_params=index_params,
        )

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    def _memory_to_row(self, memory: Memory) -> dict[str, Any]:
        row: dict[str, Any] = {
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
        if self._multi_vec.enabled:
            row["context_embedding"] = memory.context_embedding or [0.0] * self._ctx_dim
        if self._hybrid_cfg.enabled:
            row["sparse_embedding"] = memory.sparse_embedding or {}
        return row

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
            context_embedding=row.get("context_embedding"),
            sparse_embedding=row.get("sparse_embedding"),
            metadata=metadata,
            importance=row.get("importance", 0.5),
            source=row.get("source") or None,
            created_at=row.get("created_at", 0.0),
            updated_at=row.get("updated_at", 0.0),
            expires_at=expires_at if expires_at else None,
        )

    def _rows_to_relationships(self, rows: list[dict]) -> list[Relationship]:
        rels = []
        for row in rows:
            metadata = row.get("metadata_json", "{}")
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            rels.append(Relationship(
                id=row["id"],
                source_id=row.get("source_id", ""),
                target_id=row.get("target_id", ""),
                relation_type=row.get("relation_type", ""),
                description=row.get("description", ""),
                weight=row.get("weight", 1.0),
                app_id=row.get("app_id") or None,
                user_id=row.get("user_id") or None,
                metadata=metadata,
                created_at=row.get("created_at", 0.0),
            ))
        return rels

    def _parse_search_results(self, results: list) -> list[SearchResult]:
        search_results: list[SearchResult] = []
        if results:
            for hit in results[0]:
                entity = hit.get("entity", hit)
                entity["id"] = hit.get("id", entity.get("id"))
                memory = self._row_to_memory(entity)
                score = hit.get("distance", 0.0)
                search_results.append(SearchResult(memory=memory, score=score))
        return search_results

    @staticmethod
    def _build_filter_expr(filters: dict[str, Any] | None) -> str:
        if not filters:
            return ""
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
