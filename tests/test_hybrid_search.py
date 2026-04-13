"""Tests for hybrid search (dense + sparse/BM25)."""

from milvus_cortex.embedding.sparse import text_to_sparse, query_to_sparse, tokenize
from milvus_cortex.runtime import MemoryRuntime


class TestSparseVectorizer:
    def test_tokenize_basic(self):
        tokens = tokenize("Hello World, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Stop words removed
        assert "this" not in tokens
        assert "is" not in tokens

    def test_text_to_sparse_returns_dict(self):
        sparse = text_to_sparse("Python is a great programming language for data science")
        assert isinstance(sparse, dict)
        assert len(sparse) > 0
        # All values should be positive floats
        for dim, val in sparse.items():
            assert isinstance(dim, int)
            assert val > 0

    def test_query_to_sparse(self):
        sparse = query_to_sparse("Python data science")
        assert isinstance(sparse, dict)
        assert len(sparse) > 0

    def test_empty_text(self):
        assert text_to_sparse("") == {}
        assert text_to_sparse("a is the") == {}  # All stop words

    def test_deterministic(self):
        s1 = text_to_sparse("hello world")
        s2 = text_to_sparse("hello world")
        assert s1 == s2


class TestHybridSearch:
    def test_hybrid_search_returns_results(self, runtime: MemoryRuntime):
        runtime.remember(content="Python is great for machine learning", app_id="h")
        runtime.remember(content="JavaScript powers modern web applications", app_id="h")
        runtime.remember(content="Rust provides memory safety guarantees", app_id="h")

        results = runtime.search(
            query="Python machine learning",
            app_id="h",
            mode="hybrid",
        )
        assert len(results) >= 1

    def test_hybrid_vs_dense_both_work(self, runtime: MemoryRuntime):
        runtime.remember(content="The quick brown fox jumps", app_id="hd")
        runtime.remember(content="A lazy dog sleeps all day", app_id="hd")

        dense_results = runtime.search(query="animals", app_id="hd", mode="dense")
        hybrid_results = runtime.search(query="animals", app_id="hd", mode="hybrid")

        assert len(dense_results) >= 1
        assert len(hybrid_results) >= 1

    def test_sparse_embedding_stored(self, runtime: MemoryRuntime):
        mem = runtime.remember(content="Testing sparse vector storage", app_id="sp")
        fetched = runtime.get(mem.id)
        assert fetched.sparse_embedding is not None
        assert isinstance(fetched.sparse_embedding, dict)
