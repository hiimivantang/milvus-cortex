"""Graph-on-Milvus engine — entity/relationship memory without Neo4j."""

from __future__ import annotations

import json
import os
from typing import Any

from milvus_cortex.config import GraphConfig
from milvus_cortex.embedding.base import EmbeddingProvider
from milvus_cortex.models import Entity, Relationship
from milvus_cortex.storage.milvus import MilvusStorage

_EXTRACT_PROMPT = """\
You are an entity/relationship extraction engine. Given text, extract entities \
and relationships.

For each entity, output: {"name": str, "type": str, "description": str}
For each relationship, output: {"source": str, "target": str, "relation": str, "description": str}

Entity types: person, organization, concept, tool, location, event, other.
Relationship types: works_at, uses, prefers, knows, related_to, part_of, created, other.

Return JSON: {"entities": [...], "relationships": [...]}
Only extract clearly stated facts. Do not infer or speculate.\
"""


class GraphEngine:
    """Manages entity and relationship extraction, storage, and traversal on Milvus."""

    def __init__(
        self,
        storage: MilvusStorage,
        embedder: EmbeddingProvider,
        config: GraphConfig,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._config = config
        self._llm_client = None

    def initialize(self) -> None:
        """Create graph collections."""
        self._storage.initialize_graph_collections()

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        """Add or resolve an entity. If a similar entity exists, return it instead."""
        embedding = self._embedder.embed_one(f"{name}: {description}" if description else name)

        # Entity resolution: check for existing similar entity
        filters = {}
        if app_id:
            filters["app_id"] = app_id
        if user_id:
            filters["user_id"] = user_id

        candidates = self._storage.search_entities(embedding, filters=filters, top_k=3)
        for existing, score in candidates:
            if score >= self._config.similarity_threshold and existing.entity_type == entity_type:
                return existing

        entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description,
            app_id=app_id,
            user_id=user_id,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._storage.insert_entity(entity)
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._storage.get_entity(entity_id)

    def find_entities(
        self,
        query: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[Entity, float]]:
        """Find entities by semantic similarity to query."""
        embedding = self._embedder.embed_one(query)
        filters = {}
        if app_id:
            filters["app_id"] = app_id
        if user_id:
            filters["user_id"] = user_id
        return self._storage.search_entities(embedding, filters=filters, top_k=top_k)

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        weight: float = 1.0,
    ) -> Relationship:
        """Create a relationship between two entities."""
        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            description=description,
            weight=weight,
            app_id=app_id,
            user_id=user_id,
        )
        self._storage.insert_relationship(rel)
        return rel

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> list[Relationship]:
        """Get all relationships for an entity."""
        return self._storage.get_relationships(entity_id, direction=direction)

    def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[Entity]:
        """Get entities connected to the given entity up to N hops."""
        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}
        neighbors: list[Entity] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for eid in frontier:
                rels = self._storage.get_relationships(eid)
                for rel in rels:
                    other_id = rel.target_id if rel.source_id == eid else rel.source_id
                    if other_id not in visited:
                        visited.add(other_id)
                        next_frontier.add(other_id)
                        entity = self._storage.get_entity(other_id)
                        if entity:
                            neighbors.append(entity)
            frontier = next_frontier
            if not frontier:
                break

        return neighbors

    # ------------------------------------------------------------------
    # Graph search
    # ------------------------------------------------------------------

    def graph_search(
        self,
        query: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 5,
        depth: int = 1,
    ) -> dict[str, Any]:
        """Search for entities and their neighborhood.

        Returns a dict with:
        - entities: list of (Entity, score) tuples
        - relationships: list of Relationship objects
        - neighbors: list of Entity objects
        """
        entities = self.find_entities(query, app_id=app_id, user_id=user_id, top_k=top_k)

        all_rels: list[Relationship] = []
        all_neighbors: list[Entity] = []
        seen_rels: set[str] = set()
        seen_neighbors: set[str] = set()

        for entity, _score in entities:
            rels = self.get_relationships(entity.id)
            for rel in rels:
                if rel.id not in seen_rels:
                    seen_rels.add(rel.id)
                    all_rels.append(rel)

            if depth > 0:
                neighbors = self.get_neighbors(entity.id, depth=depth)
                for n in neighbors:
                    if n.id not in seen_neighbors:
                        seen_neighbors.add(n.id)
                        all_neighbors.append(n)

        return {
            "entities": entities,
            "relationships": all_rels,
            "neighbors": all_neighbors,
        }

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------

    def extract_from_text(
        self,
        text: str,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract entities and relationships from text using an LLM.

        Returns dict with 'entities' and 'relationships' lists.
        """
        client = self._get_llm_client()
        response = client.chat.completions.create(
            model=self._config.extraction_model,
            messages=[
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}

        # Create entities with resolution
        entity_map: dict[str, Entity] = {}
        for item in parsed.get("entities", []):
            if not isinstance(item, dict) or "name" not in item:
                continue
            entity = self.add_entity(
                name=item["name"],
                entity_type=item.get("type", "other"),
                description=item.get("description", ""),
                app_id=app_id,
                user_id=user_id,
            )
            entity_map[item["name"]] = entity

        # Create relationships
        created_rels: list[Relationship] = []
        for item in parsed.get("relationships", []):
            if not isinstance(item, dict):
                continue
            source = entity_map.get(item.get("source", ""))
            target = entity_map.get(item.get("target", ""))
            if source and target:
                rel = self.add_relationship(
                    source_id=source.id,
                    target_id=target.id,
                    relation_type=item.get("relation", "related_to"),
                    description=item.get("description", ""),
                    app_id=app_id,
                    user_id=user_id,
                )
                created_rels.append(rel)

        return {
            "entities": list(entity_map.values()),
            "relationships": created_rels,
        }

    def _get_llm_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
            self._llm_client = OpenAI(api_key=api_key)
        return self._llm_client
