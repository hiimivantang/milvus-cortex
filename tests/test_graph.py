"""Tests for graph-on-Milvus (entity/relationship memory)."""

import pytest

from milvus_cortex.runtime import MemoryRuntime


class TestGraphEntities:
    def test_add_and_get_entity(self, graph_runtime: MemoryRuntime):
        entity = graph_runtime.add_entity(
            name="Alice",
            entity_type="person",
            description="Software engineer",
            app_id="g",
            user_id="u1",
        )
        assert entity.id
        assert entity.name == "Alice"
        assert entity.entity_type == "person"

    def test_entity_resolution(self, graph_runtime: MemoryRuntime):
        """Adding the exact same entity should resolve to the existing one."""
        e1 = graph_runtime.add_entity(
            name="Alice",
            entity_type="person",
            description="Software engineer",
            app_id="g",
            user_id="u1",
        )
        # Exact same entity
        e2 = graph_runtime.add_entity(
            name="Alice",
            entity_type="person",
            description="Software engineer",
            app_id="g",
            user_id="u1",
        )
        # Should resolve to same entity (identical embedding)
        assert e1.id == e2.id

    def test_different_entities_not_resolved(self, graph_runtime: MemoryRuntime):
        e1 = graph_runtime.add_entity(
            name="Alice", entity_type="person", app_id="g",
        )
        e2 = graph_runtime.add_entity(
            name="Python", entity_type="tool", app_id="g",
        )
        assert e1.id != e2.id

    def test_graph_not_enabled_raises(self, runtime: MemoryRuntime):
        with pytest.raises(RuntimeError, match="Graph is not enabled"):
            runtime.add_entity(name="test", entity_type="other")


class TestGraphRelationships:
    def test_add_and_get_relationships(self, graph_runtime: MemoryRuntime):
        alice = graph_runtime.add_entity(name="Alice", entity_type="person", app_id="g")
        acme = graph_runtime.add_entity(name="Acme Corp", entity_type="organization", app_id="g")

        rel = graph_runtime.add_relationship(
            source_id=alice.id,
            target_id=acme.id,
            relation_type="works_at",
            description="Alice works at Acme Corp",
            app_id="g",
        )
        assert rel.id
        assert rel.relation_type == "works_at"

        rels = graph_runtime.get_relationships(alice.id)
        assert len(rels) >= 1
        assert any(r.target_id == acme.id for r in rels)

    def test_bidirectional_relationships(self, graph_runtime: MemoryRuntime):
        a = graph_runtime.add_entity(name="A", entity_type="concept", app_id="g")
        b = graph_runtime.add_entity(name="B", entity_type="concept", app_id="g")

        graph_runtime.add_relationship(
            source_id=a.id, target_id=b.id,
            relation_type="related_to", app_id="g",
        )

        # Should find relationship from both directions
        rels_from_a = graph_runtime.get_relationships(a.id, direction="outgoing")
        rels_from_b = graph_runtime.get_relationships(b.id, direction="incoming")
        assert len(rels_from_a) >= 1
        assert len(rels_from_b) >= 1


class TestGraphSearch:
    def test_graph_search(self, graph_runtime: MemoryRuntime):
        alice = graph_runtime.add_entity(
            name="Alice", entity_type="person",
            description="Senior engineer", app_id="g",
        )
        python = graph_runtime.add_entity(
            name="Python", entity_type="tool",
            description="Programming language", app_id="g",
        )
        graph_runtime.add_relationship(
            source_id=alice.id, target_id=python.id,
            relation_type="uses", app_id="g",
        )

        result = graph_runtime.graph_search(query="engineer", app_id="g")
        assert len(result["entities"]) >= 1
        assert "relationships" in result
        assert "neighbors" in result
