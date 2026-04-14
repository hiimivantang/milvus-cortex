"""Milvus storage adapter with hybrid search, multi-vector, and partition key support.

BM25 mode branching is fully encapsulated here. The orchestrator and runtime
never import sparse embedding functions or branch on BM25 mode.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from milvus_cortex.config import CortexConfig
from milvus_cortex.embedding.sparse import query_to_sparse, text_to_sparse
from milvus_cortex.models import (
    Entity,
    Memory,
    MemoryType,
    Relationship,
    SearchResult,
)

logger = logging.getLogger(__name__)

SCOPE_FIELDS = ("app_id", "user_id", "session_id", "agent_id", "workspace_id")
ALLOWED_FILTER_KEYS = {*SCOPE_FIELDS, "memory_type", "importance", "source"}


class MilvusStorage:
    """Milvus-backed storage with hybrid search, multi-vector, and partition key support.

    All BM25 mode branching is encapsulated here. When connected to Milvus
    standalone/cloud (2.5+), uses server-side BM25 via Function(FunctionType.BM25).
    When connected to Milvus Lite (.db file), falls back to client-side sparse
    vectors from embedding/sparse.py.
    """

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

        # BM25 mode: auto-detect from URI unless explicitly set
        if config.hybrid_search.use_server_bm25 is not None:
            self._use_server_bm25 = config.hybrid_search.use_server_bm25
        else:
            # Auto-detect: standalone/cloud = server BM25, Lite (.db) = client fallback
            is_lite = config.milvus.uri.endswith(".db")
            self._use_server_bm25 = config.hybrid_search.enabled and not is_lite

    @property
    def use_server_bm25(self) -> bool:
        """Whether server-side BM25 is active (vs client-side sparse fallback)."""
        return self._use_server_bm25

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

    def delete_expired(self, now: float, filters: dict | None = None) -> int:
        """Delete expired memories using server-side scalar filtering."""
        filter_parts = ["expires_at > 0", f"expires_at <= {now}"]
        if filters:
            extra = self._build_filter_expr(filters)
            if extra:
                filter_parts.append(extra)
        filter_expr = " and ".join(filter_parts)

        results = self._client.query(
            collection_name=self._collection_name,
            filter=filter_expr,
            output_fields=["id"],
            limit=10000,
        )
        if not results:
            return 0
        expired_ids = [r["id"] for r in results]
        return self.delete(expired_ids)

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
        query_text: str,
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Hybrid dense+sparse search using RRF fusion.

        Storage decides internally whether to use server-side BM25
        (pass raw text to Milvus) or client-side sparse (compute
        sparse vector from query_text).
        """
        filter_expr = self._build_filter_expr(filters) if filters else ""

        dense_req = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=filter_expr or None,
        )

        if self._use_server_bm25:
            # Server-side BM25: pass raw text, Milvus handles tokenization + scoring
            sparse_req = AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_embedding",
                param={"metric_type": "BM25"},
                limit=top_k,
                expr=filter_expr or None,
            )
        else:
            # Client-side fallback: compute sparse vector from text
            sparse = query_to_sparse(query_text)
            if not sparse:
                # No meaningful tokens — fall back to dense-only
                return self.search(dense_embedding, filters, top_k)
            sparse_req = AnnSearchRequest(
                data=[sparse],
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
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search across content and context vectors with RRF fusion."""
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
            "embedding": rel.embedding or [0.0] * self._dim,
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
        direction: str = "both",
    ) -> list[Relationship]:
        escaped_id = entity_id.replace("\\", "\\\\").replace("'", "\\'")
        results: list[Relationship] = []
        if direction in ("outgoing", "both"):
            rows = self._client.query(
                collection_name=self._rel_collection,
                filter=f"source_id == '{escaped_id}'",
                output_fields=["*"],
                limit=100,
            )
            results.extend(self._rows_to_relationships(rows))
        if direction in ("incoming", "both"):
            rows = self._client.query(
                collection_name=self._rel_collection,
                filter=f"target_id == '{escaped_id}'",
                output_fields=["*"],
                limit=100,
            )
            results.extend(self._rows_to_relationships(rows))
        seen = set()
        unique = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)
        return unique

    def search_relationships(
        self,
        embedding: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[tuple[Relationship, float]]:
        """Semantic search over relationships by embedding similarity."""
        filter_expr = self._build_filter_expr(filters) if filters else None
        results = self._client.search(
            collection_name=self._rel_collection,
            data=[embedding],
            anns_field="embedding",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            filter=filter_expr,
            output_fields=["*"],
        )
        rels: list[tuple[Relationship, float]] = []
        if results:
            for hit in results[0]:
                row = hit.get("entity", hit)
                row["id"] = hit.get("id", row.get("id"))
                parsed = self._rows_to_relationships([row])
                if parsed:
                    rels.append((parsed[0], hit.get("distance", 0.0)))
        return rels

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
        except Exception as e:
            logger.debug("Failed to get row count for '%s': %s", name, e)
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
            # Schema migration detection for server BM25
            if self._use_server_bm25:
                try:
                    info = self._client.describe_collection(self._collection_name)
                    has_bm25 = False
                    if hasattr(info, "get"):
                        functions = info.get("functions", [])
                        has_bm25 = any(
                            f.get("type", "") == "BM25" or "bm25" in str(f.get("name", "")).lower()
                            for f in functions
                        ) if functions else False
                    if not has_bm25:
                        logger.warning(
                            "Collection '%s' exists without BM25 Function. "
                            "Falling back to client-side sparse. "
                            "To use server BM25, drop and recreate the collection.",
                            self._collection_name,
                        )
                        self._use_server_bm25 = False
                except Exception as e:
                    logger.debug("Schema detection failed, falling back to client-side sparse: %s", e)
                    self._use_server_bm25 = False
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

        # Partition key for physical multi-tenancy (requires standalone/cloud Milvus)
        if self._use_partition_key:
            schema.add_field(
                "user_id", DataType.VARCHAR, max_length=256,
                is_partition_key=True,
            )

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

        # Multi-vector: context embedding
        if self._multi_vec.enabled:
            schema.add_field("context_embedding", DataType.FLOAT_VECTOR, dim=self._ctx_dim)
            index_params.add_index(field_name="context_embedding", index_type="AUTOINDEX", metric_type="COSINE")

        # Hybrid search: sparse vector
        if self._hybrid_cfg.enabled:
            if self._use_server_bm25:
                # Server-side BM25: content as explicit VARCHAR with analyzer, BM25 Function
                schema.add_field(
                    "content", DataType.VARCHAR, max_length=65535,
                    enable_analyzer=True,
                )
                schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
                try:
                    from pymilvus import Function, FunctionType
                    bm25_fn = Function(
                        name="content_bm25",
                        function_type=FunctionType.BM25,
                        input_field_names=["content"],
                        output_field_names=["sparse_embedding"],
                    )
                    schema.add_function(bm25_fn)
                    index_params.add_index(
                        field_name="sparse_embedding",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="BM25",
                    )
                except (ImportError, Exception) as e:
                    logger.warning("Server BM25 unavailable (%s), falling back to client-side sparse", e)
                    self._use_server_bm25 = False
                    index_params.add_index(
                        field_name="sparse_embedding",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                    )
            else:
                # Client-side sparse fallback
                schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
                index_params.add_index(
                    field_name="sparse_embedding",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",
                )

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
            collection_name=self._entity_collection, schema=schema, index_params=index_params,
        )

    def _ensure_relationship_collection(self) -> None:
        if self._client.has_collection(self._rel_collection):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("source_id", DataType.VARCHAR, max_length=64)
        schema.add_field("target_id", DataType.VARCHAR, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(
            collection_name=self._rel_collection, schema=schema, index_params=index_params,
        )

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    def _memory_to_row(self, memory: Memory) -> dict[str, Any]:
        content = memory.content

        # Truncation guard for server BM25 (VARCHAR max_length=65535 bytes)
        if self._use_server_bm25:
            content_bytes = content.encode("utf-8")
            if len(content_bytes) > 65535:
                logger.warning(
                    "Content exceeds 65535 bytes (%d), truncating for BM25 VARCHAR field",
                    len(content_bytes),
                )
                content = content_bytes[:65535].decode("utf-8", errors="ignore")

        row: dict[str, Any] = {
            "id": memory.id,
            "embedding": memory.embedding or [0.0] * self._dim,
            "content": content,
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
            if self._use_server_bm25:
                # Server BM25: Milvus auto-generates sparse from content — do NOT include sparse_embedding
                pass
            else:
                # Client-side fallback: compute sparse vector from content
                row["sparse_embedding"] = memory.sparse_embedding or text_to_sparse(content)
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
            if key not in ALLOWED_FILTER_KEYS or value is None:
                continue
            if isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace("'", "\\'")
                parts.append(f"{key} == '{escaped}'")
            elif isinstance(value, (int, float)):
                parts.append(f"{key} == {float(value)}")
        return " and ".join(parts) if parts else ""
