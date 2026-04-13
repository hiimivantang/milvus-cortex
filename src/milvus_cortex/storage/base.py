"""Abstract storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from milvus_cortex.models import Memory, SearchResult


class StorageBackend(ABC):
    """Interface that any storage adapter must implement."""

    @abstractmethod
    def initialize(self) -> None:
        """Create collections/tables if they don't exist."""

    @abstractmethod
    def insert(self, memories: list[Memory]) -> list[str]:
        """Insert memories, return their IDs."""

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID."""

    @abstractmethod
    def update(self, memory: Memory) -> None:
        """Update an existing memory in-place."""

    @abstractmethod
    def delete(self, memory_ids: list[str]) -> int:
        """Delete memories by ID. Return count deleted."""

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        filters: dict | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Vector similarity search with optional scalar filters."""

    @abstractmethod
    def list_memories(
        self,
        filters: dict | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories matching scalar filters."""

    @abstractmethod
    def count(self, filters: dict | None = None) -> int:
        """Count memories matching filters."""

    def close(self) -> None:
        """Clean up connections. Override if needed."""
