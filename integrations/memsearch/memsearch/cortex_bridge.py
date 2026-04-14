"""Bridge between memsearch chunk model and milvus-cortex Memory model.

Converts chunk dicts ↔ Memory objects at the adapter boundary so that
memsearch's public API (MemSearch class, CLI) remains unchanged while
all storage is delegated to cortex's MemoryRuntime.
"""

from __future__ import annotations

from typing import Any

from milvus_cortex.config import (
    CortexConfig,
    EmbeddingConfig,
    GraphConfig,
    HybridSearchConfig,
    LifecycleConfig,
    MilvusConfig,
    MultiVectorConfig,
)
from milvus_cortex.models import Memory, MemoryType, SearchResult


def chunk_to_memory(
    chunk: dict[str, Any],
    app_id: str,
    user_id: str,
) -> Memory:
    """Convert a memsearch chunk dict into a cortex Memory object.

    Mapping:
        chunk["chunk_hash"]    → Memory.id
        chunk["content"]       → Memory.content
        chunk["embedding"]     → Memory.embedding
        chunk["source"]        → Memory.source  (top-level, filterable)
        chunk["heading"]       → Memory.metadata["heading"]
        chunk["heading_level"] → Memory.metadata["heading_level"]
        chunk["start_line"]    → Memory.metadata["start_line"]
        chunk["end_line"]      → Memory.metadata["end_line"]
    """
    metadata: dict[str, Any] = {}
    for key in ("heading", "heading_level", "start_line", "end_line"):
        if key in chunk:
            metadata[key] = chunk[key]

    return Memory(
        id=chunk["chunk_hash"],
        content=chunk["content"],
        memory_type=MemoryType.SEMANTIC,
        app_id=app_id,
        user_id=user_id,
        embedding=chunk.get("embedding"),
        source=chunk.get("source", ""),
        metadata=metadata,
        importance=0.5,
    )


def memory_to_chunk(memory: Memory) -> dict[str, Any]:
    """Convert a cortex Memory back to a memsearch chunk dict."""
    return {
        "chunk_hash": memory.id,
        "content": memory.content,
        "source": memory.source or "",
        "heading": memory.metadata.get("heading", ""),
        "heading_level": memory.metadata.get("heading_level", 0),
        "start_line": memory.metadata.get("start_line", 0),
        "end_line": memory.metadata.get("end_line", 0),
    }


def search_result_to_chunk(result: SearchResult) -> dict[str, Any]:
    """Convert a cortex SearchResult to a memsearch result dict (with score)."""
    chunk = memory_to_chunk(result.memory)
    chunk["score"] = result.score
    return chunk


def build_cortex_config(
    milvus_uri: str,
    token: str | None = None,
    dimension: int = 1536,
    user_id: str = "default",
    *,
    collection_prefix: str = "memsearch",
) -> CortexConfig:
    """Build a CortexConfig from memsearch settings.

    Enables hybrid search, multi-vector, partition key, and graph features
    when connected to Milvus standalone (non-.db URI).
    """
    is_standalone = not milvus_uri.endswith(".db")

    return CortexConfig(
        milvus=MilvusConfig(
            uri=milvus_uri,
            token=token or "",
            collection_prefix=collection_prefix,
            use_partition_key=is_standalone,
        ),
        embedding=EmbeddingConfig(
            provider="fake",  # memsearch handles its own embeddings
            dimensions=dimension,
        ),
        hybrid_search=HybridSearchConfig(
            enabled=True,
            use_server_bm25=True if is_standalone else False,
        ),
        multi_vector=MultiVectorConfig(
            enabled=True,
        ),
        graph=GraphConfig(
            enabled=is_standalone,
        ),
        lifecycle=LifecycleConfig(
            auto_dedup=False,  # memsearch handles dedup via chunk_hash
            auto_expire=False,
        ),
    )
