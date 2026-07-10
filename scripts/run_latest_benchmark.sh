#!/bin/bash
# Latest benchmark run: 10 models × 9 scenarios × 8 sectors × 3 prompt types × 5 replicates
# Total: 7,650 evaluations
# Estimated time: 1.5-2 hours
# Estimated cost: $780-990

set -e

PYTHON=".venv/bin/python"

echo "=========================================="
echo "SecDrift Latest Benchmark Run"
echo "10 models × 9 scenarios × 8 sectors × 3 prompt types × 5 replicates"
echo "Total: 7,650 evaluations"
echo "Estimated time: 1.5-2 hours"
echo "Estimated cost: ~\$780-990"
echo "=========================================="
echo ""

mkdir -p results/latest_run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Starting at: $(date)"
echo ""

# BATCH 1: Budget Models
echo "BATCH 1: Llama 4 Maverick, Llama 3.3 70B, Gemma 3 27B, Qwen3 32B"
$PYTHON -m secdrift.runner \
  --model llama-4-maverick --model llama-3-3-70b \
  --model gemma-3-27b --model qwen3-32b \
  --replicates 5 --workers 40 \
  --output results/latest_run/batch1_budget_${TIMESTAMP}.jsonl &
BATCH1_PID=$!

# BATCH 2: Mid-Tier Models
echo "BATCH 2: DeepSeek R1, GPT-OSS 120B, Mistral Large 3"
$PYTHON -m secdrift.runner \
  --model deepseek-r1 --model gpt-oss-120b --model mistral-large-3 \
  --replicates 5 --workers 30 \
  --output results/latest_run/batch2_midtier_${TIMESTAMP}.jsonl &
BATCH2_PID=$!

# BATCH 3: Premium Models
echo "BATCH 3: Claude Sonnet 4.6, Claude Opus 4.6, Kimi K2.5"
$PYTHON -m secdrift.runner \
  --model claude-sonnet-4-6 --model claude-opus-4-6 --model kimi-k2-5 \
  --replicates 5 --workers 30 \
  --output results/latest_run/batch3_premium_${TIMESTAMP}.jsonl &
BATCH3_PID=$!

echo "All 3 batches launched. Waiting..."
wait $BATCH1_PID $BATCH2_PID $BATCH3_PID
echo "All batches completed at: $(date)"

# Merge
cat results/latest_run/batch1_budget_${TIMESTAMP}.jsonl \
    results/latest_run/batch2_midtier_${TIMESTAMP}.jsonl \
    results/latest_run/batch3_premium_${TIMESTAMP}.jsonl \
    > results/latest_10models_5rep.jsonl

TOTAL=$(wc -l < results/latest_10models_5rep.jsonl)
echo "Merged ${TOTAL} evaluations into results/latest_10models_5rep.jsonl"
echo ""
echo "BENCHMARK COMPLETE at $(date)"
echo "Next: secdrift analyze results/latest_10models_5rep.jsonl"
