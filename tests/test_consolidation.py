"""Tests for memory consolidation pipeline."""

from milvus_cortex.embedding.sparse import text_to_sparse
from milvus_cortex.models import Memory, MemoryType
from milvus_cortex.runtime import MemoryRuntime


class TestConsolidation:
    def test_consolidate_similar_memories(self, runtime: MemoryRuntime):
        # Insert memories with same embedding (shared base text) but unique content
        # to avoid dedup but guarantee clustering by vector similarity
        shared_embedding = runtime._embedder.embed_one("Python data science")
        shared_sparse = text_to_sparse("Python data science")
        contents = [
            "User prefers Python for data science",
            "User prefers Python for machine learning",
            "User prefers Python for AI research",
            "User prefers Python for deep learning",
            "User prefers Python for analytics",
        ]
        for i, content in enumerate(contents):
            mem = Memory(
                content=content,
                memory_type=MemoryType.SEMANTIC,
                app_id="cons",
                user_id="u1",
                importance=0.5 + i * 0.05,
                embedding=shared_embedding,
                sparse_embedding=shared_sparse,
                source="manual",
            )
            mem = runtime._lifecycle.apply_ttl(mem)
            runtime._storage.insert([mem])

        initial_count = runtime.count(app_id="cons", user_id="u1")
        assert initial_count >= 5

        consolidated = runtime.consolidate(
            app_id="cons",
            user_id="u1",
            similarity_threshold=0.9,  # High threshold works because embeddings are identical
            min_cluster_size=3,
        )

        final_count = runtime.count(app_id="cons", user_id="u1")
        # Consolidation should reduce memory count
        assert final_count < initial_count

    def test_consolidate_no_duplicates(self, runtime: MemoryRuntime):
        """Dissimilar memories should not be consolidated."""
        runtime.remember(content="Python is great", app_id="cons2")
        runtime.remember(content="The weather is nice today", app_id="cons2")
        runtime.remember(content="Quantum computing advances", app_id="cons2")

        consolidated = runtime.consolidate(
            app_id="cons2",
            similarity_threshold=0.99,  # Very high threshold
            min_cluster_size=2,
        )
        assert len(consolidated) == 0

    def test_consolidate_too_few_memories(self, runtime: MemoryRuntime):
        """Should not consolidate if below minimum cluster size."""
        runtime.remember(content="single memory", app_id="cons3")
        consolidated = runtime.consolidate(app_id="cons3", min_cluster_size=5)
        assert len(consolidated) == 0
