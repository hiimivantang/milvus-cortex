"""Tests for multi-vector memory representations."""

from milvus_cortex.runtime import MemoryRuntime


class TestMultiVector:
    def test_remember_with_context_embedding(self, runtime: MemoryRuntime):
        mem = runtime.remember(
            content="User prefers dark mode",
            app_id="mv",
            user_id="u1",
            context="Settings discussion about UI preferences",
        )
        fetched = runtime.get(mem.id)
        assert fetched.embedding is not None
        assert fetched.context_embedding is not None
        assert len(fetched.context_embedding) == 8
        # Content and context embeddings should be different
        assert fetched.embedding != fetched.context_embedding

    def test_remember_without_context(self, runtime: MemoryRuntime):
        """Without context text, context_embedding should be zero vector."""
        mem = runtime.remember(content="Just a fact", app_id="mv")
        fetched = runtime.get(mem.id)
        assert fetched.embedding is not None
        # context_embedding stored as zero vector when no context provided
        assert fetched.context_embedding is not None

    def test_multi_vector_search(self, runtime: MemoryRuntime):
        runtime.remember(
            content="User likes Python",
            app_id="mv",
            user_id="u1",
            context="Programming language discussion",
        )
        runtime.remember(
            content="User likes hiking",
            app_id="mv",
            user_id="u1",
            context="Weekend activity planning",
        )

        results = runtime.search(
            query="programming preferences",
            app_id="mv",
            user_id="u1",
            mode="multi_vector",
            context_query="technical discussion",
        )
        assert len(results) >= 1
