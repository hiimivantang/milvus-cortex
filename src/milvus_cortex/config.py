"""Configuration system for the memory runtime."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MilvusConfig(BaseModel):
    """Connection settings for Milvus."""

    uri: str = "http://localhost:19530"
    token: str = ""
    db_name: str = "default"
    collection_prefix: str = "cortex"


class EmbeddingConfig(BaseModel):
    """Settings for the embedding provider."""

    provider: str = "openai"  # "openai" | "sentence_transformers" | "custom"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    api_key: str | None = None  # Falls back to OPENAI_API_KEY env var
    batch_size: int = 64


class ExtractionConfig(BaseModel):
    """Settings for memory extraction."""

    provider: str = "llm"  # "llm" | "custom"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    max_tokens: int = 1024


class LifecycleConfig(BaseModel):
    """Settings for memory lifecycle management."""

    default_ttl_seconds: float | None = None  # None = no expiry
    working_memory_ttl_seconds: float = 3600.0  # 1 hour
    dedup_threshold: float = 0.95  # Cosine similarity threshold for dedup
    auto_dedup: bool = True
    auto_expire: bool = True


class CortexConfig(BaseModel):
    """Top-level configuration for the memory runtime."""

    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)

    @classmethod
    def for_testing(cls) -> CortexConfig:
        """Config preset for tests — uses Milvus Lite (local file)."""
        return cls(
            milvus=MilvusConfig(uri="./test_milvus.db"),
            embedding=EmbeddingConfig(provider="fake", dimensions=8),
        )
