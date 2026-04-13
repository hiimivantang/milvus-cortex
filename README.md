# Milvus Cortex

Milvus-native memory runtime for agent systems. Handles ingestion, extraction, storage, retrieval, and lifecycle management so you can add memory to your agents without building the plumbing from scratch.

## Features

- **Ingest** conversations, events, tool outputs, and documents
- **Extract** durable memories from raw content (via LLM)
- **Store** memories in Milvus with automatic embedding and indexing
- **Search** by semantic similarity with scope filtering
- **Retrieve** assembled context bundles ready for prompt injection
- **Manage lifecycle** — TTL, expiry, deduplication, merge, explicit forget
- **Namespace** memories by app, user, session, agent, or workspace
- **Multiple memory types** — episodic, semantic, procedural, working

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
from milvus_cortex import CortexConfig, MemoryRuntime
from milvus_cortex.config import MilvusConfig, EmbeddingConfig

# Local dev — Milvus Lite (no server needed)
config = CortexConfig(
    milvus=MilvusConfig(uri="./my_memories.db"),
    embedding=EmbeddingConfig(model="text-embedding-3-small"),
)

with MemoryRuntime.from_config(config) as runtime:
    # Store a memory
    runtime.remember(
        content="User prefers concise answers",
        app_id="myapp",
        user_id="u1",
    )

    # Search
    results = runtime.search(
        query="How does the user like responses?",
        app_id="myapp",
        user_id="u1",
    )

    # Get context for prompt injection
    context = runtime.get_context(
        query="How should I respond?",
        app_id="myapp",
        user_id="u1",
    )
    print(context.to_text())
```

## API

### `MemoryRuntime.from_config(config)`

Build a fully-wired runtime. Connects to Milvus, initializes collections.

### `runtime.remember(content, *, app_id, user_id, ...)`

Store a single memory. Automatically embeds, deduplicates, and applies TTL.

### `runtime.ingest_messages(messages, *, app_id, user_id, ...)`

Ingest a conversation. Stores as episodic memory and optionally extracts durable semantic/procedural memories via LLM.

### `runtime.search(query, *, app_id, user_id, top_k=10)`

Vector similarity search scoped to the given namespace.

### `runtime.get_context(query, *, app_id, user_id, top_k=10)`

Search and assemble a `ContextBundle` with token estimates, ready for injection.

### `runtime.forget(memory_id=..., memory_ids=[...])`

Explicitly delete memories.

### `runtime.expire(**scope)`

Run an expiry sweep — removes memories past their TTL.

### `runtime.merge(memory_ids, merged_content)`

Merge multiple memories into one (e.g., after LLM summarization).

## Architecture

```
src/milvus_cortex/
├── runtime.py          # Main API — MemoryRuntime
├── models.py           # Memory, Message, SearchResult, ContextBundle
├── config.py           # CortexConfig, MilvusConfig, etc.
├── storage/
│   ├── base.py         # StorageBackend ABC
│   └── milvus.py       # Milvus adapter (pymilvus MilvusClient)
├── embedding/
│   ├── base.py         # EmbeddingProvider ABC
│   ├── openai.py       # OpenAI embeddings
│   └── fake.py         # Deterministic fake for testing
├── extraction/
│   ├── base.py         # MemoryExtractor ABC
│   └── llm.py          # LLM-based extraction
├── retrieval/
│   └── orchestrator.py # Search + filter + rank + assemble
└── lifecycle/
    └── manager.py      # Dedup, merge, expiry, forget
```

## Memory Types

| Type | Purpose | Default TTL |
|------|---------|-------------|
| `semantic` | Facts, preferences, knowledge | None (permanent) |
| `episodic` | Conversation history, events | None |
| `procedural` | Learned workflows, patterns | None |
| `working` | Short-lived session scratch | 1 hour |

## Configuration

```python
CortexConfig(
    milvus=MilvusConfig(
        uri="http://localhost:19530",  # Or "./local.db" for Milvus Lite
        token="",
        collection_prefix="cortex",
    ),
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
    ),
    extraction=ExtractionConfig(
        provider="llm",
        model="gpt-4o-mini",
    ),
    lifecycle=LifecycleConfig(
        default_ttl_seconds=None,
        working_memory_ttl_seconds=3600,
        dedup_threshold=0.95,
        auto_dedup=True,
    ),
)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0
