"""Tests for CortexStore — the memsearch↔cortex adapter.

These tests use Milvus Lite (via tmp_db fixture) so no Docker required.
"""

from __future__ import annotations

import pytest

from memsearch.store import CortexStore, MilvusStore


class TestCortexStoreAlias:
    def test_milvus_store_is_cortex_store(self):
        """MilvusStore alias exists for backwards compatibility."""
        assert MilvusStore is CortexStore


class TestNoPymilvusImports:
    def test_store_has_no_pymilvus(self):
        """Verify store.py does not import pymilvus directly."""
        import inspect
        import memsearch.store as store_mod

        source = inspect.getsource(store_mod)
        assert "from pymilvus" not in source
        assert "import pymilvus" not in source

    def test_cortex_bridge_has_no_pymilvus(self):
        """Verify cortex_bridge.py does not import pymilvus directly."""
        import inspect
        import memsearch.cortex_bridge as bridge_mod

        source = inspect.getsource(bridge_mod)
        assert "from pymilvus" not in source
        assert "import pymilvus" not in source


class TestUpsert:
    def test_upsert_single_chunk(self, cortex_store, sample_chunks):
        count = cortex_store.upsert([sample_chunks[0]])
        assert count == 1
        assert cortex_store.count() == 1

    def test_upsert_multiple_chunks(self, cortex_store, sample_chunks):
        count = cortex_store.upsert(sample_chunks)
        assert count == 3
        assert cortex_store.count() == 3

    def test_upsert_empty(self, cortex_store):
        count = cortex_store.upsert([])
        assert count == 0


class TestSearch:
    def test_search_returns_results(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        results = cortex_store.search(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            query_text="python programming",
            top_k=5,
        )
        assert isinstance(results, list)
        assert len(results) > 0
        # Results should be dicts with expected keys
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "score" in r
            assert "chunk_hash" in r

    def test_search_empty_store(self, cortex_store):
        results = cortex_store.search(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            query_text="anything",
        )
        assert results == []


class TestQuery:
    def test_query_all(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        chunks = cortex_store.query()
        assert len(chunks) == 3

    def test_query_returns_chunk_format(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        chunks = cortex_store.query()
        for c in chunks:
            assert "chunk_hash" in c
            assert "content" in c
            assert "source" in c
            assert "heading" in c


class TestHashesBySource:
    def test_returns_hashes_for_source(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        hashes = cortex_store.hashes_by_source("/docs/milvus.md")
        assert "hash_002" in hashes
        assert "hash_003" in hashes
        assert "hash_001" not in hashes

    def test_empty_for_unknown_source(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        hashes = cortex_store.hashes_by_source("/nonexistent.md")
        assert hashes == set()


class TestIndexedSources:
    def test_returns_all_sources(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        sources = cortex_store.indexed_sources()
        assert "/docs/python.md" in sources
        assert "/docs/milvus.md" in sources


class TestDelete:
    def test_delete_by_source(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        assert cortex_store.count() == 3
        cortex_store.delete_by_source("/docs/milvus.md")
        assert cortex_store.count() == 1

    def test_delete_by_hashes(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        cortex_store.delete_by_hashes(["hash_001", "hash_002"])
        assert cortex_store.count() == 1

    def test_delete_empty_hashes(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        cortex_store.delete_by_hashes([])
        assert cortex_store.count() == 3


class TestDrop:
    def test_drop_clears_all(self, cortex_store, sample_chunks):
        cortex_store.upsert(sample_chunks)
        assert cortex_store.count() == 3
        cortex_store.drop()
        assert cortex_store.count() == 0


class TestContextManager:
    def test_context_manager(self, tmp_db):
        with CortexStore(uri=tmp_db, collection="ctx_test", dimension=8) as store:
            store.upsert([{
                "chunk_hash": "ctx1",
                "content": "Context manager test",
                "embedding": [0.1] * 8,
                "source": "test.md",
                "heading": "Test",
                "heading_level": 1,
                "start_line": 1,
                "end_line": 2,
            }])
            assert store.count() == 1


class TestRuntime:
    def test_runtime_exposed(self, cortex_store):
        """CortexStore exposes the underlying MemoryRuntime."""
        runtime = cortex_store.runtime
        assert runtime is not None
        assert hasattr(runtime, "search")
        assert hasattr(runtime, "remember")
        assert hasattr(runtime, "consolidate")
