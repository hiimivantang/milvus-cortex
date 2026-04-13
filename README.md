# Milvus Cortex

Milvus-native memory runtime for agent systems. Goes deep on Milvus capabilities that generic memory frameworks treat as lowest-common-denominator — hybrid search, multi-vector representations, graph-on-Milvus, memory consolidation, and native observability.

## Why Milvus Cortex?

Generic memory frameworks (like mem0) support 25+ vector stores but treat each one identically. Milvus Cortex exploits Milvus-specific features for better retrieval quality and operational efficiency:

| Feature | Generic Framework | Milvus Cortex |
|---------|-------------------|---------------|
| Search | Dense vector only | **Hybrid dense+sparse (BM25 + RRF fusion)** |
| Vectors | Single embedding | **Multi-vector (content + context)** |
| Knowledge graph | Requires Neo4j | **Graph-on-Milvus (zero extra infra)** |
| Memory management | Per-fact dedup | **Cluster-based consolidation pipeline** |
| Observability | None | **Native collection stats + search diagnostics** |
| Multi-tenancy | Filter-based | **Partition key support (standalone/cloud)** |

## Retrieval Quality (LongMemEval-S Benchmark)

Benchmarked on [LongMemEval](https://github.com/xiaowu0162/longmemeval) (ICLR 2025), 465 questions, Milvus standalone 2.5+, `text-embedding-3-small`:

| Mode | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|------|----------|-----------|--------|---------|
| Dense only | 0.974 | 0.985 | 0.916 | 0.926 |
| **Hybrid (BM25 + Dense)** | **0.974** | **0.996** | **0.915** | **0.935** |

Hybrid search improves Recall@10 by +1.1% and NDCG@10 by +1.0% over dense-only. On Milvus standalone, hybrid uses server-side BM25 via `Function(FunctionType.BM25)` for maximum quality. On Milvus Lite, it falls back to a client-side BM25 tokenizer.

Reproduce: `docker-compose up -d && python benchmarks/longmemeval/run_benchmark.py --data path/to/longmemeval_s_cleaned.json`

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
from milvus_cortex import CortexConfig, MemoryRuntime
from milvus_cortex.config import MilvusConfig, EmbeddingConfig

config = CortexConfig(
    milvus=MilvusConfig(uri="./my_memories.db"),  # Milvus Lite for local dev
    embedding=EmbeddingConfig(model="text-embedding-3-small"),
)

with MemoryRuntime.from_config(config) as runtime:
    # Store a memory (auto-generates dense + sparse + context embeddings)
    runtime.remember(
        content="User prefers concise answers",
        app_id="myapp",
        user_id="u1",
        context="Settings discussion about response style",  # Multi-vector context
    )

    # Hybrid search (dense + BM25 sparse, fused via RRF)
    results = runtime.search(
        query="How does the user like responses?",
        app_id="myapp",
        user_id="u1",
        mode="hybrid",  # "dense", "hybrid", "multi_vector", "auto"
    )

    # Get assembled context for prompt injection
    context = runtime.get_context(
        query="How should I respond?",
        app_id="myapp",
        user_id="u1",
    )
    print(context.to_text())
```

## Features

### 1. Hybrid Search (Dense + Sparse/BM25)

Combines dense vector similarity with BM25 sparse keyword matching via Reciprocal Rank Fusion. Better recall than either approach alone.

```python
results = runtime.search(query="Python API authentication", mode="hybrid")
```

The sparse vectorizer is built-in (no external dependencies) — it uses BM25-style term frequency with hash-space mapping, compatible with Milvus `SPARSE_FLOAT_VECTOR`.

### 2. Multi-Vector Memory Representations

Store multiple embeddings per memory: **content** (what the memory says) and **context** (the situation when it was created). Search can leverage both.

```python
runtime.remember(
    content="User prefers TypeScript",
    context="Discussion about frontend framework choices",
)

# Search weighting both content and context relevance
results = runtime.search(
    query="language preferences",
    mode="multi_vector",
    context_query="frontend development",
)
```

### 3. Graph-on-Milvus (No Neo4j)

Entity and relationship memory stored directly in Milvus — zero additional infrastructure. Entity resolution via vector similarity, graph traversal via filtered search.

```python
config = CortexConfig(graph=GraphConfig(enabled=True))
runtime = MemoryRuntime.from_config(config)

alice = runtime.add_entity(name="Alice", entity_type="person", app_id="g")
python = runtime.add_entity(name="Python", entity_type="tool", app_id="g")
runtime.add_relationship(alice.id, python.id, "uses", app_id="g")

# Search entities and traverse relationships
result = runtime.graph_search(query="engineer", app_id="g", depth=2)
# Returns: entities, relationships, neighbors
```

### 4. Memory Consolidation Pipeline

Cluster related memories by vector proximity and merge them. Reduces redundancy without losing information.

```python
consolidated = runtime.consolidate(
    app_id="myapp",
    user_id="u1",
    similarity_threshold=0.85,
    min_cluster_size=3,
)
print(f"Merged {len(consolidated)} clusters")
```

### 5. Milvus-Native Observability

Collection stats, memory distribution, and search performance diagnostics.

```python
stats = runtime.stats(app_id="myapp", user_id="u1")
print(f"Total: {stats.total_memories}, Types: {stats.by_type}")
print(f"Avg importance: {stats.avg_importance:.2f}")

health = runtime.health()
print(f"Rows: {health.row_count}, Index: {health.index_status}")

diag = runtime.search_diagnostics()
print(f"P95 latency: {diag.get('p95_ms', 'N/A')}ms")
```

### 6. Partition Key Multi-Tenancy

Physical data isolation per user via Milvus partition keys. Zero filter overhead at query time. (Requires standalone/cloud Milvus — Milvus Lite falls back to filter-based scoping.)

```python
config = CortexConfig(
    milvus=MilvusConfig(uri="http://localhost:19530", use_partition_key=True),
)
```

## Full API

| Method | Description |
|--------|-------------|
| `remember(content, *, context, ...)` | Store memory with auto-embedding (dense + sparse + context) |
| `ingest_messages(messages, ...)` | Ingest conversation, optionally extract durable memories via LLM |
| `search(query, *, mode, ...)` | Search with `"dense"`, `"hybrid"`, `"multi_vector"`, or `"auto"` mode |
| `get_context(query, ...)` | Assemble `ContextBundle` for prompt injection |
| `forget(memory_id)` | Delete memories |
| `expire(**scope)` | TTL sweep for expired memories |
| `merge(memory_ids, content)` | Merge multiple memories into one |
| `consolidate(**scope)` | Cluster and merge related memories |
| `add_entity(name, type, ...)` | Add entity to knowledge graph |
| `add_relationship(src, tgt, ...)` | Add relationship between entities |
| `graph_search(query, ...)` | Search entities + traverse relationships |
| `stats(**scope)` | Memory statistics |
| `health()` | Collection health status |
| `search_diagnostics()` | Search latency percentiles |

## Architecture

```
src/milvus_cortex/
├── runtime.py              # Main API — MemoryRuntime
├── models.py               # Memory, Entity, Relationship, SearchResult, etc.
├── config.py               # CortexConfig with feature toggles
├── storage/
│   ├── base.py             # StorageBackend ABC
│   └── milvus.py           # Milvus adapter (hybrid, multi-vector, graph collections)
├── embedding/
│   ├── base.py             # EmbeddingProvider ABC
│   ├── openai.py           # OpenAI embeddings
│   ├── fake.py             # Deterministic fake for testing
│   └── sparse.py           # BM25-style sparse vectorizer (no deps)
├── extraction/
│   ├── base.py             # MemoryExtractor ABC
│   └── llm.py              # LLM-based extraction
├── retrieval/
│   └── orchestrator.py     # Hybrid/multi-vector search orchestration
├── graph/
│   └── engine.py           # Entity/relationship extraction and traversal
├── lifecycle/
│   └── manager.py          # Dedup, merge, expiry, consolidation
└── observability.py        # Stats, health, search diagnostics
```

## Configuration

```python
CortexConfig(
    milvus=MilvusConfig(
        uri="http://localhost:19530",  # Or "./local.db" for Milvus Lite
        collection_prefix="cortex",
        use_partition_key=False,       # Physical tenant isolation
    ),
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
    ),
    hybrid_search=HybridSearchConfig(
        enabled=True,                  # Dense + sparse search
        rrf_k=60,                      # RRF fusion constant
    ),
    multi_vector=MultiVectorConfig(
        enabled=True,                  # Content + context embeddings
    ),
    graph=GraphConfig(
        enabled=False,                 # Graph-on-Milvus
        similarity_threshold=0.85,     # Entity resolution threshold
    ),
    lifecycle=LifecycleConfig(
        auto_dedup=True,
        dedup_threshold=0.95,
        consolidation_threshold=0.85,
        consolidation_min_cluster=3,
    ),
)
```

## Development

```bash
pip install -e ".[dev]"
pytest  # 55 tests
```

## License

Apache-2.0
