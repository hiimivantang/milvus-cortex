"""Tests for cortex_bridge — chunk↔Memory mapping."""

from __future__ import annotations

from milvus_cortex.models import Memory, MemoryType, SearchResult

from memsearch.cortex_bridge import (
    build_cortex_config,
    chunk_to_memory,
    memory_to_chunk,
    search_result_to_chunk,
)


class TestChunkToMemory:
    def test_basic_mapping(self):
        chunk = {
            "chunk_hash": "abc123",
            "content": "Hello world",
            "source": "/docs/readme.md",
            "heading": "Intro",
            "heading_level": 2,
            "start_line": 1,
            "end_line": 5,
        }
        memory = chunk_to_memory(chunk, app_id="test_app", user_id="user1")

        assert memory.id == "abc123"
        assert memory.content == "Hello world"
        assert memory.source == "/docs/readme.md"
        assert memory.app_id == "test_app"
        assert memory.user_id == "user1"
        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.metadata["heading"] == "Intro"
        assert memory.metadata["heading_level"] == 2
        assert memory.metadata["start_line"] == 1
        assert memory.metadata["end_line"] == 5

    def test_missing_optional_fields(self):
        chunk = {
            "chunk_hash": "xyz789",
            "content": "Minimal chunk",
        }
        memory = chunk_to_memory(chunk, app_id="app", user_id="u")

        assert memory.id == "xyz789"
        assert memory.content == "Minimal chunk"
        assert memory.source == ""
        assert memory.metadata == {}


class TestMemoryToChunk:
    def test_round_trip(self):
        chunk_in = {
            "chunk_hash": "abc123",
            "content": "Hello world",
            "source": "/docs/readme.md",
            "heading": "Intro",
            "heading_level": 2,
            "start_line": 1,
            "end_line": 5,
        }
        memory = chunk_to_memory(chunk_in, app_id="app", user_id="u")
        chunk_out = memory_to_chunk(memory)

        assert chunk_out["chunk_hash"] == "abc123"
        assert chunk_out["content"] == "Hello world"
        assert chunk_out["source"] == "/docs/readme.md"
        assert chunk_out["heading"] == "Intro"
        assert chunk_out["heading_level"] == 2
        assert chunk_out["start_line"] == 1
        assert chunk_out["end_line"] == 5

    def test_defaults_for_missing_metadata(self):
        memory = Memory(
            id="test1",
            content="No metadata",
            memory_type=MemoryType.SEMANTIC,
        )
        chunk = memory_to_chunk(memory)

        assert chunk["chunk_hash"] == "test1"
        assert chunk["heading"] == ""
        assert chunk["heading_level"] == 0
        assert chunk["start_line"] == 0
        assert chunk["end_line"] == 0


class TestSearchResultToChunk:
    def test_includes_score(self):
        memory = Memory(
            id="sr1",
            content="Search result content",
            source="/docs/test.md",
            metadata={"heading": "Test"},
        )
        result = SearchResult(memory=memory, score=0.95)
        chunk = search_result_to_chunk(result)

        assert chunk["chunk_hash"] == "sr1"
        assert chunk["content"] == "Search result content"
        assert chunk["score"] == 0.95
        assert chunk["source"] == "/docs/test.md"
        assert chunk["heading"] == "Test"


class TestBuildCortexConfig:
    def test_standalone_uri(self):
        config = build_cortex_config(
            milvus_uri="http://localhost:19530",
            dimension=1536,
        )
        assert config.hybrid_search.use_server_bm25 is True
        assert config.milvus.use_partition_key is True
        assert config.graph.enabled is True
        assert config.embedding.provider == "fake"
        assert config.embedding.dimensions == 1536

    def test_lite_uri(self):
        config = build_cortex_config(
            milvus_uri="/tmp/test.db",
            dimension=384,
        )
        assert config.hybrid_search.use_server_bm25 is False
        assert config.milvus.use_partition_key is False
        assert config.graph.enabled is False
        assert config.embedding.dimensions == 384

    def test_custom_collection_prefix(self):
        config = build_cortex_config(
            milvus_uri="http://localhost:19530",
            collection_prefix="my_app",
        )
        assert config.milvus.collection_prefix == "my_app"

    def test_token_passthrough(self):
        config = build_cortex_config(
            milvus_uri="http://localhost:19530",
            token="my_token",
        )
        assert config.milvus.token == "my_token"

    def test_dedup_disabled(self):
        config = build_cortex_config(milvus_uri="/tmp/test.db")
        assert config.lifecycle.auto_dedup is False
        assert config.lifecycle.auto_expire is False
