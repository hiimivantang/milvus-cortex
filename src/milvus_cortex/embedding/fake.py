"""Fake embedding provider for testing — deterministic, no API calls."""

from __future__ import annotations

import hashlib
import math

from milvus_cortex.config import EmbeddingConfig
from milvus_cortex.embedding.base import EmbeddingProvider


class FakeEmbedding(EmbeddingProvider):
    """Produces deterministic pseudo-embeddings from content hashes.

    Not useful for real retrieval quality, but lets tests run without
    an API key or network access. Nearby strings will NOT be nearby in
    this space — use only for integration/unit tests.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._dimensions = config.dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).hexdigest()
        # Expand the hash to fill the vector dimensions
        raw: list[float] = []
        for i in range(self._dimensions):
            byte_val = int(digest[(i * 2) % len(digest) : (i * 2) % len(digest) + 2], 16)
            raw.append((byte_val / 255.0) * 2 - 1)  # Map to [-1, 1]
        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
