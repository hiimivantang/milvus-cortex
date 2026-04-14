# Active Projects

## milvus-cortex v0.2

A Milvus-native memory runtime for agent systems. Key features:
- Hybrid BM25+dense search using Milvus 2.5 server-side Functions
- Graph-on-Milvus: entity and relationship storage with semantic traversal
- Memory consolidation: cluster and merge similar memories
- Multi-vector context embeddings for richer retrieval
- Partition key multi-tenancy for agent isolation

Current focus: integrating with memsearch to demonstrate all 5 features
through a real-world search tool used by OpenClaw agents.

## OpenClaw Agent Framework

Building autonomous coding agents that use memsearch for persistent memory.
The agents store conversation summaries, extracted facts, and code patterns
in markdown files that memsearch indexes and searches.

Key integration point: agents call `memsearch search` to retrieve relevant
context before generating code. Version 3.2.1 added support for the
progressive disclosure workflow (search -> expand -> transcript).

## LongMemEval Benchmark

Running the ICLR 2025 LongMemEval-S benchmark to measure retrieval quality.
Comparing dense-only vs hybrid (BM25+dense) vs hybrid+rerank across
recall@5, recall@10, and NDCG@5 metrics. Server-side BM25 should show
meaningful improvement due to global IDF statistics.
