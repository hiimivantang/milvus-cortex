"""OpenAI embedding adapter."""

from __future__ import annotations

import os

from milvus_cortex.config import EmbeddingConfig
from milvus_cortex.embedding.base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    """Generates embeddings via the OpenAI API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        from openai import OpenAI

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)
        self._model = config.model
        self._dimensions = config.dimensions
        self._batch_size = config.batch_size

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model=self._model,
                dimensions=self._dimensions,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings
