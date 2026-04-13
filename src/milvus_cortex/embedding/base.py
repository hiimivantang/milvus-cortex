"""Abstract embedding interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for turning text into vectors."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the output vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single text."""
        return self.embed([text])[0]
