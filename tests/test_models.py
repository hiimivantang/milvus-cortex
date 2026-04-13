"""Tests for domain models."""

from milvus_cortex.models import ContextBundle, Memory, MemoryType, Message, SearchResult


def test_memory_defaults():
    m = Memory(content="hello world")
    assert m.content == "hello world"
    assert m.memory_type == MemoryType.SEMANTIC
    assert m.id  # Auto-generated
    assert m.importance == 0.5
    assert m.expires_at is None
    assert m.created_at > 0


def test_memory_type_enum():
    assert MemoryType("episodic") == MemoryType.EPISODIC
    assert MemoryType("semantic") == MemoryType.SEMANTIC
    assert MemoryType("procedural") == MemoryType.PROCEDURAL
    assert MemoryType("working") == MemoryType.WORKING


def test_message():
    m = Message(role="user", content="hi")
    assert m.role == "user"
    assert m.timestamp > 0


def test_context_bundle_to_text():
    memories = [
        SearchResult(
            memory=Memory(content="fact one", memory_type=MemoryType.SEMANTIC),
            score=0.95,
        ),
        SearchResult(
            memory=Memory(content="fact two", memory_type=MemoryType.EPISODIC),
            score=0.80,
        ),
    ]
    bundle = ContextBundle(memories=memories, summary="User context", token_estimate=10)
    text = bundle.to_text()
    assert "Summary: User context" in text
    assert "fact one" in text
    assert "fact two" in text
    assert "semantic" in text


def test_context_bundle_to_text_with_limit():
    memories = [
        SearchResult(memory=Memory(content=f"fact {i}"), score=0.9 - i * 0.1)
        for i in range(5)
    ]
    bundle = ContextBundle(memories=memories)
    text = bundle.to_text(max_memories=2)
    assert "fact 0" in text
    assert "fact 1" in text
    assert "fact 2" not in text
