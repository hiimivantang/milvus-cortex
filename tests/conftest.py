"""Shared fixtures for tests."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from milvus_cortex.config import (
    CortexConfig,
    EmbeddingConfig,
    GraphConfig,
    HybridSearchConfig,
    MilvusConfig,
    MultiVectorConfig,
)
from milvus_cortex.runtime import MemoryRuntime


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp(prefix="cortex_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def config(tmp_dir: str) -> CortexConfig:
    """CortexConfig wired for local testing (Milvus Lite, fake embeddings, all features on)."""
    return CortexConfig(
        milvus=MilvusConfig(uri=os.path.join(tmp_dir, "test.db")),
        embedding=EmbeddingConfig(provider="fake", dimensions=8),
        hybrid_search=HybridSearchConfig(enabled=True),
        multi_vector=MultiVectorConfig(enabled=True),
    )


@pytest.fixture()
def graph_config(tmp_dir: str) -> CortexConfig:
    """Config with graph-on-Milvus enabled."""
    return CortexConfig(
        milvus=MilvusConfig(uri=os.path.join(tmp_dir, "test_graph.db")),
        embedding=EmbeddingConfig(provider="fake", dimensions=8),
        hybrid_search=HybridSearchConfig(enabled=True),
        multi_vector=MultiVectorConfig(enabled=True),
        graph=GraphConfig(enabled=True),
    )


@pytest.fixture()
def runtime(config: CortexConfig) -> MemoryRuntime:
    rt = MemoryRuntime.from_config(config)
    yield rt
    rt.close()


@pytest.fixture()
def graph_runtime(graph_config: CortexConfig) -> MemoryRuntime:
    rt = MemoryRuntime.from_config(graph_config)
    yield rt
    rt.close()
