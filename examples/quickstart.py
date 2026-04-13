"""Quickstart example for Milvus Cortex.

This example uses Milvus Lite (local file) and fake embeddings,
so no external services are required.

For production, configure real Milvus and OpenAI embedding:

    config = CortexConfig(
        milvus=MilvusConfig(uri="http://localhost:19530"),
        embedding=EmbeddingConfig(model="text-embedding-3-small"),
    )
"""

from milvus_cortex import CortexConfig, MemoryRuntime, MemoryType
from milvus_cortex.config import EmbeddingConfig, MilvusConfig


def main():
    # 1. Configure — using local Milvus Lite and fake embeddings for the demo
    config = CortexConfig(
        milvus=MilvusConfig(uri="./demo_milvus.db"),
        embedding=EmbeddingConfig(provider="fake", dimensions=8),
    )

    with MemoryRuntime.from_config(config) as runtime:
        # 2. Store some memories
        runtime.remember(
            content="User prefers concise technical answers",
            app_id="demo",
            user_id="u1",
            memory_type="semantic",
        )

        runtime.remember(
            content="User is working on a RAG pipeline with LangChain",
            app_id="demo",
            user_id="u1",
            memory_type="semantic",
            importance=0.8,
        )

        runtime.remember(
            content="Always validate inputs before calling external APIs",
            app_id="demo",
            user_id="u1",
            memory_type="procedural",
        )

        # 3. Ingest a conversation (without LLM extraction in this demo)
        runtime.ingest_messages(
            messages=[
                {"role": "user", "content": "How do I connect to Milvus?"},
                {"role": "assistant", "content": "Use pymilvus: MilvusClient('http://localhost:19530')"},
                {"role": "user", "content": "Thanks, that worked!"},
            ],
            app_id="demo",
            user_id="u1",
            session_id="s1",
            extract=False,  # Skip LLM extraction for this demo
        )

        # 4. Search for relevant memories
        print("=== Search Results ===")
        results = runtime.search(
            query="What does the user prefer?",
            app_id="demo",
            user_id="u1",
            top_k=5,
        )
        for r in results:
            print(f"  [{r.score:.2f}] ({r.memory.memory_type.value}) {r.memory.content}")

        # 5. Get assembled context for an agent prompt
        print("\n=== Context Bundle ===")
        context = runtime.get_context(
            query="How should I respond to this user?",
            app_id="demo",
            user_id="u1",
        )
        print(context.to_text())
        print(f"\nEstimated tokens: {context.token_estimate}")

        # 6. Count and list
        print(f"\nTotal memories for u1: {runtime.count(app_id='demo', user_id='u1')}")

        # 7. Forget a memory
        if results:
            first_id = results[0].memory.id
            runtime.forget(memory_id=first_id)
            print(f"Forgot memory {first_id}")
            print(f"Remaining: {runtime.count(app_id='demo', user_id='u1')}")


if __name__ == "__main__":
    main()
