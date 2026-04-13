"""Domain models for the memory runtime."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory the runtime can store."""

    EPISODIC = "episodic"  # What happened — conversation turns, events
    SEMANTIC = "semantic"  # What is known — facts, preferences, knowledge
    PROCEDURAL = "procedural"  # How to do things — learned workflows, patterns
    WORKING = "working"  # Short-lived scratch context for active sessions


class Message(BaseModel):
    """A single conversation message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class Memory(BaseModel):
    """Core memory object stored in the runtime."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    # Scope fields — all optional to support flexible namespacing
    app_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    workspace_id: str | None = None
    # Vector
    embedding: list[float] | None = None
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5  # 0.0–1.0 importance score
    source: str | None = None  # e.g. "extraction", "manual", "ingest"
    # Timestamps (epoch seconds)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    expires_at: float | None = None  # None = never expires


class SearchResult(BaseModel):
    """A memory with a relevance score from search."""

    memory: Memory
    score: float  # Higher = more relevant


class ContextBundle(BaseModel):
    """Assembled context ready to inject into an agent prompt."""

    memories: list[SearchResult]
    summary: str | None = None
    token_estimate: int = 0

    def to_text(self, max_memories: int | None = None) -> str:
        """Render context as plain text for prompt injection."""
        items = self.memories[:max_memories] if max_memories else self.memories
        lines: list[str] = []
        if self.summary:
            lines.append(f"Summary: {self.summary}")
            lines.append("")
        for i, result in enumerate(items, 1):
            m = result.memory
            lines.append(f"[{i}] ({m.memory_type.value}, score={result.score:.2f}) {m.content}")
        return "\n".join(lines)
