"""HTTP-compatible embedding provider for Ollama, HuggingFace TEI, vLLM, etc.

Works with any endpoint that implements the OpenAI /v1/embeddings API format.
Uses the `openai` library (already a dependency) with a custom base_url.
"""

from __future__ import annotations

import logging
import os

from milvus_cortex.config import EmbeddingConfig
from milvus_cortex.embedding.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class HttpEmbedding(EmbeddingProvider):
    """Generates embeddings via any OpenAI-compatible HTTP endpoint.

    Supports Ollama, HuggingFace TEI, vLLM, LiteLLM, and other
    providers that implement the /v1/embeddings API.

    Example configs:
        Ollama:  base_url="http://localhost:11434/v1", model="nomic-embed-text"
        HF TEI:  base_url="http://localhost:8080/v1", model="BAAI/bge-base-en-v1.5"
        vLLM:    base_url="http://localhost:8000/v1", model="intfloat/e5-mistral-7b-instruct"
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        from openai import OpenAI

        base_url = config.base_url or "http://localhost:11434/v1"
        api_key = config.api_key or os.environ.get("EMBEDDING_API_KEY", "no-key")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
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
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                    dimensions=self._dimensions,
                )
            except Exception as e:
                # Some providers (Ollama, older HF TEI) don't support
                # the dimensions parameter — retry without it
                logger.debug("Retrying without dimensions param: %s", e)
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                )
            items = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend(
                [item.embedding[: self._dimensions] for item in items]
            )
        return all_embeddings
