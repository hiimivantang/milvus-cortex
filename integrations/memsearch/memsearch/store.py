"""CortexStore — memsearch storage adapter backed by milvus-cortex MemoryRuntime.

Replaces the original MilvusStore with identical public method signatures,
but delegates all Milvus operations to cortex's MemoryRuntime. No pymilvus
imports remain in this module.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from milvus_cortex.models import MemoryType
from milvus_cortex.runtime import MemoryRuntime

from .cortex_bridge import (
    build_cortex_config,
    memory_to_chunk,
    search_result_to_chunk,
)

logger = logging.getLogger(__name__)


def _escape_filter_value(value: str) -> str:
    """Escape backslashes and double quotes for filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class CortexStore:
    """Memsearch storage backed by milvus-cortex MemoryRuntime.

    Drop-in replacement for MilvusStore with the same public interface.
    Uses cortex's hybrid BM25+dense search, multi-vector context embeddings,
    and partition key multi-tenancy.
    """

    DEFAULT_COLLECTION = "memsearch_chunks"

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        *,
        token: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        dimension: int | None = 1536,
        description: str = "",
        user_id: str = "default",
    ) -> None:
        self._uri = uri
        self._collection = collection
        self._dimension = dimension or 1536
        self._user_id = user_id
        self._app_id = collection  # use collection name as app_id for scope isolation

        config = build_cortex_config(
            milvus_uri=uri,
            token=token,
            dimension=self._dimension,
            user_id=user_id,
            collection_prefix=collection,
        )
        self._runtime = MemoryRuntime.from_config(config)

    @property
    def runtime(self) -> MemoryRuntime:
        """Expose the underlying MemoryRuntime for cortex-specific features."""
        return self._runtime

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        """Insert or update chunks (keyed by chunk_hash primary key).

        Each chunk dict must have: chunk_hash, embedding, content, source,
        heading, heading_level, start_line, end_line.
        """
        if not chunks:
            return 0

        count = 0
        for chunk in chunks:
            # Build context string from heading + source for multi-vector
            heading = chunk.get("heading", "")
            source = chunk.get("source", "")
            context = f"{heading} (from {source})" if heading else source

            self._runtime.remember(
                content=chunk["content"],
                app_id=self._app_id,
                user_id=self._user_id,
                memory_type=MemoryType.SEMANTIC,
                metadata={
                    k: chunk[k]
                    for k in ("heading", "heading_level", "start_line", "end_line")
                    if k in chunk
                },
                embedding=chunk.get("embedding"),
                context=context,
                importance=0.5,
                memory_id=chunk.get("chunk_hash"),
                source=source,
            )
            count += 1

        return count

    def search(
        self,
        query_embedding: list[float],
        *,
        query_text: str = "",
        top_k: int = 10,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        """Hybrid search: dense vector + BM25 full-text with RRF reranking."""
        results = self._runtime.search(
            query=query_text or "search",
            app_id=self._app_id,
            user_id=self._user_id,
            top_k=top_k,
            mode="hybrid",
            query_embedding=query_embedding,
        )
        return [search_result_to_chunk(r) for r in results]

    _QUERY_FIELDS: ClassVar[list[str]] = [
        "content",
        "source",
        "heading",
        "chunk_hash",
        "heading_level",
        "start_line",
        "end_line",
    ]

    def query(self, *, filter_expr: str = "") -> list[dict[str, Any]]:
        """Retrieve chunks by scalar filter (no vector needed)."""
        # Handle chunk_hash lookup directly via runtime.get()
        if filter_expr and "chunk_hash" in filter_expr:
            parts = filter_expr.split('"')
            hash_val = parts[1] if len(parts) >= 3 else ""
            if hash_val:
                memory = self._runtime.get(hash_val)
                return [memory_to_chunk(memory)] if memory else []

        memories = self._runtime.list_memories(
            app_id=self._app_id,
            user_id=self._user_id,
            limit=10000,
        )
        chunks = [memory_to_chunk(m) for m in memories]

        # Apply source filter if present
        if filter_expr and "source" in filter_expr and "==" in filter_expr:
            parts = filter_expr.split('"')
            source_val = parts[1] if len(parts) >= 3 else ""
            if source_val:
                chunks = [c for c in chunks if c["source"] == source_val]

        return chunks

    def hashes_by_source(self, source: str) -> set[str]:
        """Return all chunk_hash values for a given source file."""
        memories = self._runtime.list_memories(
            app_id=self._app_id,
            user_id=self._user_id,
            limit=10000,
        )
        return {
            m.id for m in memories
            if (m.source or "") == source
        }

    def indexed_sources(self) -> set[str]:
        """Return all distinct source values in the collection."""
        memories = self._runtime.list_memories(
            app_id=self._app_id,
            user_id=self._user_id,
            limit=10000,
        )
        return {m.source for m in memories if m.source}

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a given source file."""
        memories = self._runtime.list_memories(
            app_id=self._app_id,
            user_id=self._user_id,
            limit=10000,
        )
        ids = [m.id for m in memories if (m.source or "") == source]
        if ids:
            self._runtime.forget(memory_ids=ids)

    def delete_by_hashes(self, hashes: list[str]) -> None:
        """Delete chunks by their content hashes (primary keys)."""
        if hashes:
            self._runtime.forget(memory_ids=hashes)

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._runtime.count(
            app_id=self._app_id,
            user_id=self._user_id,
        )

    def drop(self) -> None:
        """Drop all data by deleting all memories in scope."""
        memories = self._runtime.list_memories(
            app_id=self._app_id,
            user_id=self._user_id,
            limit=10000,
        )
        if memories:
            self._runtime.forget(memory_ids=[m.id for m in memories])

    def close(self) -> None:
        """Release resources."""
        self._runtime.close()

    def __enter__(self) -> CortexStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# Backwards-compatible alias so existing imports still work
MilvusStore = CortexStore
