"""Tests for lifecycle management."""

import time

from milvus_cortex.models import Memory, MemoryType
from milvus_cortex.runtime import MemoryRuntime


class TestExpiry:
    def test_working_memory_gets_ttl(self, runtime: MemoryRuntime):
        mem = runtime.remember(
            content="scratch note",
            app_id="test",
            memory_type="working",
        )
        assert mem.expires_at is not None
        assert mem.expires_at > mem.created_at

    def test_semantic_memory_no_default_ttl(self, runtime: MemoryRuntime):
        mem = runtime.remember(
            content="permanent fact",
            app_id="test",
            memory_type="semantic",
        )
        # Default config has no TTL for semantic memories
        assert mem.expires_at is None

    def test_expire_removes_old_memories(self, runtime: MemoryRuntime):
        # Store with an expiry in the past
        mem = runtime.remember(
            content="should expire",
            app_id="test",
            expires_at=time.time() - 100,
        )
        assert runtime.get(mem.id) is not None
        deleted = runtime.expire(app_id="test")
        assert deleted >= 1
        assert runtime.get(mem.id) is None


class TestForget:
    def test_forget_removes_memory(self, runtime: MemoryRuntime):
        mem = runtime.remember(content="forget me", app_id="test")
        runtime.forget(memory_id=mem.id)
        assert runtime.get(mem.id) is None
