"""Abstract extraction interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from milvus_cortex.models import Memory, Message


class MemoryExtractor(ABC):
    """Extracts durable memories from raw content (messages, text, etc.)."""

    @abstractmethod
    def extract_from_messages(self, messages: list[Message]) -> list[Memory]:
        """Analyze a conversation and return extracted memories."""

    @abstractmethod
    def extract_from_text(self, text: str, source: str | None = None) -> list[Memory]:
        """Extract memories from arbitrary text content."""
