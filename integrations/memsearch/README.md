# memsearch-cortex

A fork of [memsearch](https://github.com/nicholasgasior/memsearch) powered by [milvus-cortex](../../) as the storage backend. All 5 cortex features are available through memsearch's familiar API:

1. **Hybrid BM25+Dense search** — server-side BM25 via Milvus 2.5 Functions
2. **Graph extraction** — entity/relationship discovery from indexed content
3. **Memory consolidation** — cluster and merge similar memories
4. **Multi-vector context embeddings** — heading+source contextual vectors
5. **Partition key multi-tenancy** — user_id-based tenant isolation

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for Milvus standalone)
- OpenAI API key (for embeddings and graph extraction)

## Setup

### 1. Start Milvus standalone

```bash
cd integrations/memsearch
docker-compose up -d
```

Wait for Milvus to be healthy (~30 seconds):

```bash
curl http://localhost:9091/healthz
```

### 2. Install the package

From the repo root:

```bash
pip install -e integrations/memsearch
```

### 3. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

## Quick Start

### Run the demo

The demo script exercises all 5 cortex features end-to-end:

```bash
cd integrations/memsearch
python demo/demo_agent_loop.py
```

It runs 7 phases:
1. **Setup** — connect to Milvus standalone
2. **Index** — index sample markdown files (hybrid BM25+dense + multi-vector + partition key)
3. **Search** — hybrid retrieval with keyword-heavy queries
4. **Graph extraction** — discover entities and relationships
5. **Consolidation** — merge similar memories
6. **Multi-tenancy** — demonstrate partition key isolation between agents
7. **Agent loop** — simulate an OpenClaw-style agent memory workflow

### CLI usage

```bash
# Index markdown files
memsearch index ./demo/sample_memories --milvus-uri http://localhost:19530

# Hybrid search (BM25 + dense)
memsearch search "OAuth2 authentication migration" --milvus-uri http://localhost:19530

# Extract entities and relationships from indexed content
memsearch graph-extract --milvus-uri http://localhost:19530

# Consolidate similar memories
memsearch consolidate --milvus-uri http://localhost:19530

# Graph search
memsearch graph-search "people involved in auth migration" --milvus-uri http://localhost:19530

# Show cortex runtime stats
memsearch cortex-stats --milvus-uri http://localhost:19530

# Watch for file changes and auto-index
memsearch watch ./docs --milvus-uri http://localhost:19530
```

### Programmatic usage

```python
import asyncio
from memsearch import MemSearch

async def main():
    ms = MemSearch(
        paths=["./docs"],
        milvus_uri="http://localhost:19530",
        user_id="my_agent",
    )

    # Index markdown files
    await ms.index()

    # Hybrid search (BM25 + dense automatically)
    results = await ms.search("my query", top_k=5)
    for r in results:
        print(f"{r['score']:.4f} | {r['heading']}: {r['content'][:80]}")

    # Cortex-specific features
    graph = await ms.extract_graph()
    merged = await ms.consolidate()
    entities = ms.graph_search("auth migration")
    stats = ms.cortex_stats()

    ms.close()

asyncio.run(main())
```

## Multi-tenancy

Use `user_id` to isolate data between agents or users:

```python
ms_a = MemSearch(
    milvus_uri="http://localhost:19530",
    collection="shared_collection",
    user_id="agent_a",
)
ms_b = MemSearch(
    milvus_uri="http://localhost:19530",
    collection="shared_collection",
    user_id="agent_b",
)

# Each agent only sees its own data
results_a = await ms_a.search("secret")  # Only agent_a's memories
results_b = await ms_b.search("secret")  # Only agent_b's memories
```

## Architecture

```
OpenClaw Agent --> MemSearch (this fork) --> milvus-cortex MemoryRuntime --> Milvus Standalone
```

The `CortexStore` adapter replaces memsearch's original `MilvusStore`, delegating all storage operations to cortex's `MemoryRuntime`. Memsearch's public API (MemSearch class, CLI) is fully preserved.

Key files:
- `memsearch/store.py` — CortexStore adapter (drop-in MilvusStore replacement)
- `memsearch/cortex_bridge.py` — chunk-to-Memory mapping layer
- `memsearch/core.py` — MemSearch with cortex feature methods

## Running Tests

```bash
# Unit tests (no Docker required — uses Milvus Lite)
pytest integrations/memsearch/tests/

# All tests including cortex core
pytest tests/ && pytest integrations/memsearch/tests/
```

## Tear Down

```bash
cd integrations/memsearch
docker-compose down -v   # -v removes data volumes
```
