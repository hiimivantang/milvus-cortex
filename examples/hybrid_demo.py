"""Hybrid search demo — shows BM25+dense finding memories that dense-only misses.

Requires Milvus standalone:
    docker-compose up -d
    export OPENAI_API_KEY=sk-...
    python examples/hybrid_demo.py

For local testing without OpenAI/Docker (Milvus Lite + fake embeddings):
    python examples/hybrid_demo.py --lite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milvus_cortex import CortexConfig, MemoryRuntime
from milvus_cortex.config import EmbeddingConfig, HybridSearchConfig, MilvusConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lite", action="store_true", help="Use Milvus Lite + fake embeddings (no Docker/API needed)")
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    args = parser.parse_args()

    if args.lite:
        config = CortexConfig(
            milvus=MilvusConfig(uri="./hybrid_demo.db"),
            embedding=EmbeddingConfig(provider="fake", dimensions=8),
            hybrid_search=HybridSearchConfig(enabled=True),
        )
    else:
        config = CortexConfig(
            milvus=MilvusConfig(uri=args.milvus_uri, collection_prefix="hybrid_demo"),
            embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=1536),
            hybrid_search=HybridSearchConfig(enabled=True),
        )

    with MemoryRuntime.from_config(config) as runtime:
        # Ingest diverse memories — some with specific keywords that
        # dense embeddings might not capture well
        memories = [
            "User's favorite programming language is Rust",
            "The deployment uses Kubernetes v1.28 on AWS EKS",
            "Database connection string uses port 5432 for PostgreSQL",
            "API rate limit is set to 1000 requests per minute per user",
            "The team uses Jira project key CORTEX-1234 for tracking",
            "Authentication uses OAuth 2.0 with Auth0 as the IdP",
            "User prefers dark mode and compact UI layout",
            "The ML model was trained on 2024-03-15 with accuracy 94.7%",
            "Budget approval was given by Sarah Chen, VP of Engineering",
            "The incident on 2024-01-22 was caused by a memory leak in the gRPC server",
            "Preferred communication channel is the #eng-platform Slack channel",
            "The SLA requires 99.95% uptime measured monthly",
            "CI/CD pipeline uses GitHub Actions with a 15-minute timeout",
            "The feature flag for dark-launch is FF_CORTEX_V2_ENABLED",
            "User mentioned they are allergic to shellfish during onboarding",
        ]

        print("Ingesting 15 memories...\n")
        for content in memories:
            runtime.remember(content=content, app_id="demo", user_id="u1")

        # Queries that benefit from keyword matching
        test_queries = [
            ("CORTEX-1234", "Exact Jira key — BM25 excels at exact matches"),
            ("Sarah Chen", "Person name — embeddings may not encode names well"),
            ("FF_CORTEX_V2_ENABLED", "Feature flag name — pure keyword match"),
            ("gRPC memory leak", "Technical terms — hybrid combines semantic + keyword"),
        ]

        for query, description in test_queries:
            print(f"Query: \"{query}\"")
            print(f"  ({description})")

            # Dense-only search
            dense_results = runtime.search(
                query=query, app_id="demo", user_id="u1",
                mode="dense", top_k=3,
            )

            # Hybrid search (BM25 + dense)
            hybrid_results = runtime.search(
                query=query, app_id="demo", user_id="u1",
                mode="hybrid", top_k=3,
            )

            print(f"\n  Dense-only top 3:")
            for i, r in enumerate(dense_results[:3], 1):
                print(f"    {i}. [{r.score:.3f}] {r.memory.content[:80]}")

            print(f"\n  Hybrid (BM25+Dense) top 3:")
            for i, r in enumerate(hybrid_results[:3], 1):
                print(f"    {i}. [{r.score:.3f}] {r.memory.content[:80]}")

            print()
            print("-" * 70)
            print()

    # Cleanup lite db
    if args.lite:
        import os
        os.remove("./hybrid_demo.db") if os.path.exists("./hybrid_demo.db") else None


if __name__ == "__main__":
    main()
