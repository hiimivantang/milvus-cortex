"""Configuration system for the memory runtime."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MilvusConfig(BaseModel):
    """Connection settings for Milvus."""

    uri: str = "http://localhost:19530"
    token: str = ""
    db_name: str = "default"
    collection_prefix: str = "cortex"
    use_partition_key: bool = False  # Physical tenant isolation (requires standalone/cloud Milvus)


class EmbeddingConfig(BaseModel):
    """Settings for the embedding provider."""

    provider: str = "openai"  # "openai" | "fake"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    api_key: str | None = None  # Falls back to OPENAI_API_KEY env var
    batch_size: int = 64


class HybridSearchConfig(BaseModel):
    """Settings for hybrid dense+sparse search."""

    enabled: bool = True
    sparse_weight: float = 0.3  # Weight for sparse results in RRF fusion
    dense_weight: float = 0.7  # Weight for dense results in RRF fusion
    rrf_k: int = 60  # RRF constant (standard default)


class MultiVectorConfig(BaseModel):
    """Settings for multi-vector memory representations."""

    enabled: bool = True
    context_dimensions: int | None = None  # None = same as embedding dimensions


class ExtractionConfig(BaseModel):
    """Settings for memory extraction."""

    provider: str = "llm"  # "llm" | "none"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    max_tokens: int = 1024


class GraphConfig(BaseModel):
    """Settings for graph-on-Milvus entity/relationship memory."""

    enabled: bool = False
    extraction_model: str = "gpt-4o-mini"
    api_key: str | None = None
    similarity_threshold: float = 0.85  # Entity resolution threshold


class LifecycleConfig(BaseModel):
    """Settings for memory lifecycle management."""

    default_ttl_seconds: float | None = None  # None = no expiry
    working_memory_ttl_seconds: float = 3600.0  # 1 hour
    dedup_threshold: float = 0.95  # Cosine similarity threshold for dedup
    auto_dedup: bool = True
    auto_expire: bool = True
    consolidation_threshold: float = 0.85  # Similarity threshold for clustering
    consolidation_min_cluster: int = 3  # Min memories to trigger consolidation


class CortexConfig(BaseModel):
    """Top-level configuration for the memory runtime."""

    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    multi_vector: MultiVectorConfig = Field(default_factory=MultiVectorConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)

    @classmethod
    def for_testing(cls) -> CortexConfig:
        """Config preset for tests — uses Milvus Lite (local file)."""
        return cls(
            milvus=MilvusConfig(uri="./test_milvus.db"),
            embedding=EmbeddingConfig(provider="fake", dimensions=8),
        )
