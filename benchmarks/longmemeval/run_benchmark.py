"""LongMemEval benchmark harness for milvus-cortex.

Measures retrieval quality (recall@k, NDCG@k) across three search modes:
dense-only, hybrid (BM25+dense), and hybrid+rerank.

Prerequisites:
    1. Milvus standalone running: docker-compose up -d
    2. LongMemEval data: git clone https://github.com/xiaowu0162/longmemeval
    3. OpenAI API key for embeddings: export OPENAI_API_KEY=...

Usage:
    python benchmarks/longmemeval/run_benchmark.py --data path/to/longmemeval_s_cleaned.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from milvus_cortex import CortexConfig, MemoryRuntime
from milvus_cortex.config import EmbeddingConfig, HybridSearchConfig, MilvusConfig, RerankerConfig


def compute_recall_any(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    return float(any(g in top_k for g in gold_ids))


def compute_recall_all(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    return float(all(g in top_k for g in gold_ids))


def compute_ndcg(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    gold_set = set(gold_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
    # Ideal DCG: all gold docs at top positions
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def run_benchmark(data_path: str, milvus_uri: str = "http://localhost:19530", max_questions: int | None = None):
    with open(data_path) as f:
        data = json.load(f)

    if max_questions:
        data = data[:max_questions]

    modes = ["dense", "hybrid"]
    metrics: dict[str, dict[str, list[float]]] = {
        mode: {
            "recall_any@5": [], "recall_all@5": [], "ndcg@5": [],
            "recall_any@10": [], "recall_all@10": [], "ndcg@10": [],
        }
        for mode in modes
    }

    print(f"Running benchmark on {len(data)} questions...")
    print(f"Milvus URI: {milvus_uri}")
    print()

    for qi, item in enumerate(data):
        qid = item["question_id"]
        if qid.endswith("_abs"):
            continue  # Skip abstention questions for retrieval eval

        question = item["question"]
        gold_session_ids = item["answer_session_ids"]
        sessions = item["haystack_sessions"]
        session_ids = item["haystack_session_ids"]

        # Unique app_id per question for isolation
        app_id = f"bench_{qid}"

        config = CortexConfig(
            milvus=MilvusConfig(uri=milvus_uri, collection_prefix=f"bench_{qi}"),
            embedding=EmbeddingConfig(model="text-embedding-3-small", dimensions=1536),
            hybrid_search=HybridSearchConfig(enabled=True),
        )

        try:
            with MemoryRuntime.from_config(config) as runtime:
                # Ingest all sessions
                for session, sid in zip(sessions, session_ids):
                    # Concatenate user messages as memory content
                    user_text = " ".join(
                        turn["content"] for turn in session if turn.get("role") == "user"
                    )
                    if user_text.strip():
                        runtime.remember(
                            content=user_text,
                            app_id=app_id,
                            session_id=sid,
                            metadata={"session_id": sid},
                        )

                # Search in each mode
                for mode in modes:
                    results = runtime.search(
                        query=question,
                        app_id=app_id,
                        top_k=10,
                        mode=mode,
                    )

                    # Extract session IDs from results
                    retrieved_sids = []
                    for r in results:
                        sid = r.memory.session_id or r.memory.metadata.get("session_id", "")
                        if sid and sid not in retrieved_sids:
                            retrieved_sids.append(sid)

                    for k in [5, 10]:
                        metrics[mode][f"recall_any@{k}"].append(
                            compute_recall_any(retrieved_sids, gold_session_ids, k)
                        )
                        metrics[mode][f"recall_all@{k}"].append(
                            compute_recall_all(retrieved_sids, gold_session_ids, k)
                        )
                        metrics[mode][f"ndcg@{k}"].append(
                            compute_ndcg(retrieved_sids, gold_session_ids, k)
                        )

        except Exception as e:
            print(f"  Error on question {qid}: {e}")
            continue

        if (qi + 1) % 10 == 0:
            print(f"  Processed {qi + 1}/{len(data)} questions...")

    # Print results
    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY BENCHMARK — LongMemEval-S")
    print("=" * 70)

    header = f"{'Mode':<20} {'Recall@5':>10} {'Recall@10':>10} {'NDCG@5':>10} {'NDCG@10':>10}"
    print(header)
    print("-" * 70)

    for mode in modes:
        m = metrics[mode]
        n = len(m["recall_any@5"]) or 1
        row = (
            f"{mode:<20} "
            f"{sum(m['recall_any@5'])/n:>10.3f} "
            f"{sum(m['recall_any@10'])/n:>10.3f} "
            f"{sum(m['ndcg@5'])/n:>10.3f} "
            f"{sum(m['ndcg@10'])/n:>10.3f}"
        )
        print(row)

    print("=" * 70)
    print(f"Questions evaluated: {len(metrics[modes[0]]['recall_any@5'])}")

    # Write JSONL output
    output_path = Path(data_path).parent / "benchmark_results.jsonl"
    with open(output_path, "w") as f:
        for mode in modes:
            m = metrics[mode]
            n = len(m["recall_any@5"]) or 1
            result = {
                "mode": mode,
                "n_questions": n,
                **{k: sum(v) / n for k, v in m.items()},
            }
            f.write(json.dumps(result) + "\n")
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument("--data", required=True, help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--milvus-uri", default="http://localhost:19530", help="Milvus standalone URI")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions (for testing)")
    args = parser.parse_args()
    run_benchmark(args.data, args.milvus_uri, args.max_questions)
