"""Shared fixtures for tests."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from milvus_cortex.config import CortexConfig, EmbeddingConfig, MilvusConfig
from milvus_cortex.runtime import MemoryRuntime


@pytest.fixture()
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="cortex_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def config(tmp_dir: str) -> CortexConfig:
    """CortexConfig wired for local testing (Milvus Lite, fake embeddings)."""
    return CortexConfig(
        milvus=MilvusConfig(uri=os.path.join(tmp_dir, "test.db")),
        embedding=EmbeddingConfig(provider="fake", dimensions=8),
    )


@pytest.fixture()
def runtime(config: CortexConfig) -> MemoryRuntime:
    """A fully initialized MemoryRuntime for testing."""
    rt = MemoryRuntime.from_config(config)
    yield rt
    rt.close()
