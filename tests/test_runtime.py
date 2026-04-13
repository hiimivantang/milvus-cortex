"""Integration tests for MemoryRuntime — uses Milvus Lite + fake embeddings."""

import pytest

from milvus_cortex.models import MemoryType
from milvus_cortex.runtime import MemoryRuntime


class TestRemember:
    def test_remember_and_get(self, runtime: MemoryRuntime):
        mem = runtime.remember(
            content="User prefers Python",
            app_id="test",
            user_id="u1",
        )
        assert mem.id
        assert mem.content == "User prefers Python"

        fetched = runtime.get(mem.id)
        assert fetched is not None
        assert fetched.content == "User prefers Python"
        assert fetched.app_id == "test"
        assert fetched.user_id == "u1"

    def test_remember_with_type(self, runtime: MemoryRuntime):
        mem = runtime.remember(
            content="Always run tests before committing",
            app_id="test",
            memory_type="procedural",
        )
        assert mem.memory_type == MemoryType.PROCEDURAL

    def test_remember_with_metadata(self, runtime: MemoryRuntime):
        mem = runtime.remember(
            content="Likes dark mode",
            app_id="test",
            user_id="u1",
            metadata={"source": "settings_page"},
        )
        fetched = runtime.get(mem.id)
        assert fetched.metadata["source"] == "settings_page"

    def test_remember_with_context(self, runtime: MemoryRuntime):
        """Multi-vector: store with context embedding."""
        mem = runtime.remember(
            content="User prefers TypeScript",
            app_id="test",
            user_id="u1",
            context="Discussion about frontend frameworks",
        )
        fetched = runtime.get(mem.id)
        assert fetched.context_embedding is not None
        assert len(fetched.context_embedding) == 8

    def test_remember_generates_sparse(self, runtime: MemoryRuntime):
        """Hybrid search: sparse embedding is auto-generated."""
        mem = runtime.remember(
            content="Python is a great programming language",
            app_id="test",
        )
        fetched = runtime.get(mem.id)
        assert fetched.sparse_embedding is not None
        assert len(fetched.sparse_embedding) > 0


class TestSearch:
    def test_search_returns_results(self, runtime: MemoryRuntime):
        runtime.remember(content="User likes Python", app_id="test", user_id="u1")
        runtime.remember(content="User dislikes Java", app_id="test", user_id="u1")
        runtime.remember(content="Meeting at 3pm", app_id="test", user_id="u2")

        results = runtime.search(
            query="programming language preferences",
            app_id="test",
            user_id="u1",
        )
        assert len(results) >= 1
        for r in results:
            assert r.memory.user_id == "u1"

    def test_search_dense_mode(self, runtime: MemoryRuntime):
        runtime.remember(content="fact", app_id="t")
        results = runtime.search(query="anything", app_id="t", mode="dense")
        assert len(results) >= 1

    def test_search_hybrid_mode(self, runtime: MemoryRuntime):
        runtime.remember(content="Python is great for data science", app_id="t")
        runtime.remember(content="JavaScript rules the frontend", app_id="t")
        results = runtime.search(query="Python data", app_id="t", mode="hybrid")
        assert len(results) >= 1

    def test_search_with_type_filter(self, runtime: MemoryRuntime):
        runtime.remember(content="fact", app_id="t", memory_type="semantic")
        runtime.remember(content="event", app_id="t", memory_type="episodic")

        results = runtime.search(
            query="anything",
            app_id="t",
            memory_types=["semantic"],
        )
        for r in results:
            assert r.memory.memory_type == MemoryType.SEMANTIC


class TestGetContext:
    def test_get_context_returns_bundle(self, runtime: MemoryRuntime):
        runtime.remember(content="Prefers short answers", app_id="a", user_id="u1")
        runtime.remember(content="Works on ML projects", app_id="a", user_id="u1")

        ctx = runtime.get_context(query="user style", app_id="a", user_id="u1")
        assert ctx.memories
        assert ctx.token_estimate > 0
        text = ctx.to_text()
        assert len(text) > 0


class TestForget:
    def test_forget_single(self, runtime: MemoryRuntime):
        mem = runtime.remember(content="secret", app_id="test")
        assert runtime.get(mem.id) is not None
        runtime.forget(memory_id=mem.id)
        assert runtime.get(mem.id) is None

    def test_forget_multiple(self, runtime: MemoryRuntime):
        m1 = runtime.remember(content="a", app_id="test")
        m2 = runtime.remember(content="b", app_id="test")
        deleted = runtime.forget(memory_ids=[m1.id, m2.id])
        assert deleted >= 2


class TestListAndCount:
    def test_list_memories(self, runtime: MemoryRuntime):
        runtime.remember(content="one", app_id="test", user_id="u1")
        runtime.remember(content="two", app_id="test", user_id="u1")
        runtime.remember(content="three", app_id="test", user_id="u2")

        memories = runtime.list_memories(app_id="test", user_id="u1")
        assert len(memories) >= 2
        for m in memories:
            assert m.user_id == "u1"

    def test_count(self, runtime: MemoryRuntime):
        runtime.remember(content="x", app_id="test")
        runtime.remember(content="y", app_id="test")
        c = runtime.count(app_id="test")
        assert c >= 2


class TestContextManager:
    def test_with_statement(self, config):
        with MemoryRuntime.from_config(config) as rt:
            rt.remember(content="test", app_id="test")
            assert rt.count(app_id="test") >= 1
