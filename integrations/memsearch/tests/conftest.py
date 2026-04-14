"""Shared fixtures for memsearch-cortex integration tests."""

from __future__ import annotations

import os
import tempfile

import pytest

from memsearch.cortex_bridge import build_cortex_config
from memsearch.store import CortexStore


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary Milvus Lite .db file."""
    return str(tmp_path / "test_memsearch.db")


@pytest.fixture
def cortex_store(tmp_db):
    """Create a CortexStore backed by Milvus Lite for testing."""
    store = CortexStore(
        uri=tmp_db,
        collection="test_chunks",
        dimension=8,  # small dimension for fast tests
        user_id="test_user",
    )
    yield store
    store.close()


@pytest.fixture
def sample_chunks():
    """Return sample chunk dicts for testing."""
    return [
        {
            "chunk_hash": "hash_001",
            "content": "Python is a programming language created by Guido van Rossum.",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "source": "/docs/python.md",
            "heading": "Python Overview",
            "heading_level": 2,
            "start_line": 1,
            "end_line": 5,
        },
        {
            "chunk_hash": "hash_002",
            "content": "Milvus is a vector database for AI applications and similarity search.",
            "embedding": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "source": "/docs/milvus.md",
            "heading": "Milvus Introduction",
            "heading_level": 2,
            "start_line": 1,
            "end_line": 4,
        },
        {
            "chunk_hash": "hash_003",
            "content": "Hybrid search combines dense vector and sparse BM25 retrieval.",
            "embedding": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "source": "/docs/milvus.md",
            "heading": "Hybrid Search",
            "heading_level": 3,
            "start_line": 6,
            "end_line": 10,
        },
    ]
