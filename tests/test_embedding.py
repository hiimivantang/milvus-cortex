"""Tests for embedding providers."""

import math

from milvus_cortex.config import EmbeddingConfig
from milvus_cortex.embedding.fake import FakeEmbedding


def test_fake_embedding_dimensions():
    config = EmbeddingConfig(provider="fake", dimensions=16)
    embedder = FakeEmbedding(config)
    assert embedder.dimensions == 16
    vec = embedder.embed_one("hello")
    assert len(vec) == 16


def test_fake_embedding_deterministic():
    config = EmbeddingConfig(provider="fake", dimensions=8)
    embedder = FakeEmbedding(config)
    v1 = embedder.embed_one("test string")
    v2 = embedder.embed_one("test string")
    assert v1 == v2


def test_fake_embedding_unit_norm():
    config = EmbeddingConfig(provider="fake", dimensions=32)
    embedder = FakeEmbedding(config)
    vec = embedder.embed_one("normalize me")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_fake_embedding_batch():
    config = EmbeddingConfig(provider="fake", dimensions=8)
    embedder = FakeEmbedding(config)
    vecs = embedder.embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 8 for v in vecs)
    # Different inputs produce different vectors
    assert vecs[0] != vecs[1]
