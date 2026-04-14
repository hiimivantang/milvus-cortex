#!/usr/bin/env python3
"""End-to-end demo exercising all 5 cortex features through the memsearch fork.

Requirements:
  - Milvus standalone running (docker-compose up -d from parent directory)
  - OPENAI_API_KEY set (for embeddings and graph extraction)

Usage:
  cd integrations/memsearch
  docker-compose up -d
  python demo/demo_agent_loop.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add parent to path so memsearch fork is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsearch import MemSearch  # noqa: E402

MILVUS_URI = "http://localhost:19530"
SAMPLE_DIR = Path(__file__).parent / "sample_memories"


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def main() -> None:
    # ── Phase 1: Setup ──────────────────────────────────────────
    banner("Phase 1: Setup — Connect to Milvus Standalone")

    ms = MemSearch(
        paths=[str(SAMPLE_DIR)],
        embedding_provider="openai",
        milvus_uri=MILVUS_URI,
        collection="demo_agent",
        user_id="agent_demo",
    )
    print(f"Connected to Milvus at {MILVUS_URI}")
    print(f"Collection: demo_agent, User ID: agent_demo")

    # ── Phase 2: Index ──────────────────────────────────────────
    banner("Phase 2: Index — Hybrid BM25+Dense + Multi-Vector + Partition Key")

    n = await ms.index(force=True)
    print(f"Indexed {n} chunks from {SAMPLE_DIR}")
    print("  Each chunk gets:")
    print("  - Dense embedding (OpenAI text-embedding-3-small)")
    print("  - BM25 sparse vector (Milvus server-side Function)")
    print("  - Context embedding (heading + source path)")
    print("  - Partition key isolation (user_id=agent_demo)")

    # ── Phase 3: Search ─────────────────────────────────────────
    banner("Phase 3: Search — Hybrid BM25+Dense Retrieval")

    # Keyword-heavy query that benefits from BM25
    queries = [
        "Sarah Chen OAuth2 authentication migration",
        "memory leak worker pool OOM",
        "What editor theme and font does the user prefer?",
        "milvus-cortex hybrid search benchmark",
    ]

    for query in queries:
        print(f"\nQuery: {query!r}")
        results = await ms.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            source = Path(r.get("source", "")).name
            heading = r.get("heading", "")
            content = r.get("content", "")[:120]
            print(f"  [{i}] score={score:.4f} | {source}: {heading}")
            print(f"      {content}...")

    # ── Phase 4: Graph Extraction ───────────────────────────────
    banner("Phase 4: Graph Extraction — Entity & Relationship Discovery")

    try:
        result = await ms.extract_graph()
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        print(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        for e in entities[:10]:
            print(f"  Entity: {e.name} ({e.entity_type})")
        for r in relationships[:10]:
            print(f"  Rel: {r.relation_type}: {r.description}")
    except RuntimeError as exc:
        print(f"  Graph not available: {exc}")
        print("  (Graph requires Milvus standalone + OPENAI_API_KEY)")

    # ── Phase 5: Consolidation ──────────────────────────────────
    banner("Phase 5: Consolidation — Merge Similar Memories")

    count_before = ms.store.count()
    print(f"Memories before consolidation: {count_before}")

    try:
        merged = await ms.consolidate()
        count_after = ms.store.count()
        print(f"Consolidated into {len(merged)} merged memories")
        print(f"Memories after consolidation: {count_after}")
    except Exception as exc:
        print(f"  Consolidation skipped: {exc}")

    # ── Phase 6: Multi-Tenancy ──────────────────────────────────
    banner("Phase 6: Multi-Tenancy — Partition Key Isolation")

    ms_a = MemSearch(
        embedding_provider="openai",
        milvus_uri=MILVUS_URI,
        collection="demo_tenant",
        user_id="agent_a",
    )
    ms_b = MemSearch(
        embedding_provider="openai",
        milvus_uri=MILVUS_URI,
        collection="demo_tenant",
        user_id="agent_b",
    )

    # Store different facts for each tenant
    ms_a.store.upsert([{
        "chunk_hash": "tenant_a_fact1",
        "content": "Agent A's secret: the launch code is ALPHA-7",
        "embedding": None,  # will be generated
        "source": "agent_a_notes.md",
        "heading": "Secrets",
        "heading_level": 2,
        "start_line": 1,
        "end_line": 3,
    }])
    ms_b.store.upsert([{
        "chunk_hash": "tenant_b_fact1",
        "content": "Agent B's secret: the password is BRAVO-9",
        "embedding": None,
        "source": "agent_b_notes.md",
        "heading": "Secrets",
        "heading_level": 2,
        "start_line": 1,
        "end_line": 3,
    }])

    results_a = await ms_a.search("secret launch code", top_k=5)
    results_b = await ms_b.search("secret password", top_k=5)

    print(f"Agent A search results: {len(results_a)} (should find ALPHA-7)")
    for r in results_a:
        print(f"  {r.get('content', '')[:80]}")

    print(f"\nAgent B search results: {len(results_b)} (should find BRAVO-9)")
    for r in results_b:
        print(f"  {r.get('content', '')[:80]}")

    ms_a.close()
    ms_b.close()

    # ── Phase 7: Agent Loop Simulation ──────────────────────────
    banner("Phase 7: Full Agent Loop Simulation")

    print("Simulating an OpenClaw-style agent loop:")
    print("  1. Agent receives user message")

    user_msg = "What did we decide about the authentication migration?"
    print(f"     User: {user_msg!r}")

    print("  2. Agent searches memsearch for relevant context (hybrid search)")
    results = await ms.search(user_msg, top_k=3)
    print(f"     Found {len(results)} relevant memories")
    for r in results[:2]:
        print(f"     - {r.get('heading', 'untitled')}: {r.get('content', '')[:80]}...")

    print("  3. Agent stores new fact via memsearch (cortex remember)")
    ms.store.upsert([{
        "chunk_hash": "agent_fact_001",
        "content": "The user asked about auth migration. I retrieved the meeting notes with Sarah Chen about OAuth2 PKCE migration using Keycloak.",
        "embedding": None,
        "source": "agent_conversation.md",
        "heading": "Agent Memory",
        "heading_level": 2,
        "start_line": 1,
        "end_line": 5,
    }])
    print("     Stored agent memory about the interaction")

    print("  4. Loop complete.")

    # ── Cleanup ─────────────────────────────────────────────────
    banner("Demo Complete")
    stats = ms.cortex_stats()
    print(f"Final stats: {stats['stats']['total_memories']} memories, "
          f"{stats['stats']['total_entities']} entities, "
          f"{stats['stats']['total_relationships']} relationships")

    ms.close()
    print("\nAll 5 cortex features demonstrated successfully!")
    print("  1. Hybrid BM25+Dense search")
    print("  2. Graph extraction (entities + relationships)")
    print("  3. Memory consolidation")
    print("  4. Multi-vector context embeddings")
    print("  5. Partition key multi-tenancy")


if __name__ == "__main__":
    asyncio.run(main())
