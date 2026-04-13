# LongMemEval Benchmark

Measures retrieval quality for milvus-cortex across search modes using the [LongMemEval](https://github.com/xiaowu0162/longmemeval) benchmark (ICLR 2025).

## Prerequisites

1. **Milvus standalone** (Docker):
   ```bash
   docker-compose up -d
   ```

2. **LongMemEval data**:
   ```bash
   git clone https://github.com/xiaowu0162/longmemeval
   ```

3. **OpenAI API key** (for embeddings):
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

4. **Install milvus-cortex**:
   ```bash
   pip install -e ".[dev]"
   ```

## Run

```bash
python benchmarks/longmemeval/run_benchmark.py \
    --data path/to/longmemeval/data/longmemeval_s_cleaned.json \
    --milvus-uri http://localhost:19530

# Quick test with 10 questions:
python benchmarks/longmemeval/run_benchmark.py \
    --data path/to/longmemeval/data/longmemeval_s_cleaned.json \
    --max-questions 10
```

## Output

Prints a comparison table and writes `benchmark_results.jsonl`:

```
Mode                 Recall@5  Recall@10    NDCG@5   NDCG@10
----------------------------------------------------------------------
dense                   X.XXX      X.XXX     X.XXX     X.XXX
hybrid                  X.XXX      X.XXX     X.XXX     X.XXX
```

## Metrics

- **Recall@k**: Did the top-k results contain at least one gold evidence session?
- **NDCG@k**: Normalized discounted cumulative gain — measures ranking quality.
