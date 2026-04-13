"""Tests for Milvus-native observability."""

from milvus_cortex.runtime import MemoryRuntime


class TestStats:
    def test_stats_empty(self, runtime: MemoryRuntime):
        stats = runtime.stats(app_id="empty")
        assert stats.total_memories == 0

    def test_stats_with_memories(self, runtime: MemoryRuntime):
        runtime.remember(content="fact one", app_id="obs", user_id="u1", memory_type="semantic")
        runtime.remember(content="fact two", app_id="obs", user_id="u1", memory_type="episodic")
        runtime.remember(content="fact three", app_id="obs", user_id="u2", memory_type="semantic")

        stats = runtime.stats(app_id="obs")
        assert stats.total_memories == 3
        assert stats.by_type.get("semantic", 0) >= 2
        assert stats.by_type.get("episodic", 0) >= 1
        assert stats.avg_importance > 0
        assert stats.oldest_memory is not None
        assert stats.newest_memory is not None

    def test_stats_scoped(self, runtime: MemoryRuntime):
        runtime.remember(content="a", app_id="obs2", user_id="u1")
        runtime.remember(content="b", app_id="obs2", user_id="u2")

        stats = runtime.stats(app_id="obs2", user_id="u1")
        assert stats.total_memories == 1


class TestHealth:
    def test_health_returns_status(self, runtime: MemoryRuntime):
        runtime.remember(content="seed", app_id="health")
        health = runtime.health()
        assert health.collection_name
        assert health.row_count >= 1
        assert health.index_status == "ready"


class TestSearchDiagnostics:
    def test_diagnostics_empty(self, runtime: MemoryRuntime):
        diag = runtime.search_diagnostics()
        assert diag["sample_count"] == 0

    def test_diagnostics_after_searches(self, runtime: MemoryRuntime):
        runtime.remember(content="test", app_id="diag")
        runtime.search(query="test", app_id="diag")
        runtime.search(query="test", app_id="diag")

        diag = runtime.search_diagnostics()
        assert diag["sample_count"] == 2
        assert diag["avg_ms"] >= 0
        assert "p50_ms" in diag
        assert "p95_ms" in diag
