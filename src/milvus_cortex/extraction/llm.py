"""LLM-based memory extraction."""

from __future__ import annotations

import json
import os

from milvus_cortex.config import ExtractionConfig
from milvus_cortex.extraction.base import MemoryExtractor
from milvus_cortex.models import Memory, MemoryType, Message

_SYSTEM_PROMPT = """\
You are a memory extraction engine. Given a conversation or text, extract \
durable memories that would be useful to remember in future interactions.

For each memory, output a JSON object with:
- "content": the memory text (concise, self-contained)
- "memory_type": one of "episodic", "semantic", "procedural"
- "importance": float 0.0-1.0 (how important is this to remember)

Return a JSON array of memory objects. If nothing worth remembering, return [].
Only extract genuinely useful information — preferences, facts, decisions, \
instructions, relationships, constraints. Skip greetings and filler.\
"""


class LLMExtractor(MemoryExtractor):
    """Uses an LLM to extract structured memories from content."""

    def __init__(self, config: ExtractionConfig) -> None:
        from openai import OpenAI

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)
        self._model = config.model
        self._max_tokens = config.max_tokens

    def extract_from_messages(self, messages: list[Message]) -> list[Memory]:
        conversation = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self._extract(conversation, source="conversation")

    def extract_from_text(self, text: str, source: str | None = None) -> list[Memory]:
        return self._extract(text, source=source or "text")

    def _extract(self, content: str, source: str) -> list[Memory]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=self._max_tokens,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "[]"
        try:
            parsed = json.loads(raw)
            # Handle both {"memories": [...]} and [...] formats
            if isinstance(parsed, dict):
                items = parsed.get("memories", parsed.get("results", []))
            else:
                items = parsed
        except json.JSONDecodeError:
            return []

        memories: list[Memory] = []
        for item in items:
            if not isinstance(item, dict) or "content" not in item:
                continue
            memories.append(
                Memory(
                    content=item["content"],
                    memory_type=MemoryType(item.get("memory_type", "semantic")),
                    importance=float(item.get("importance", 0.5)),
                    source=f"extraction:{source}",
                )
            )
        return memories
