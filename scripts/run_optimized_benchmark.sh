#!/bin/bash
# Optimized benchmark run for 10 models with 5 replicates
# Target: Complete in under 2 hours
# Estimated cost: $545-695

set -e

echo "=========================================="
echo "SecDrift Optimized Benchmark Run"
echo "10 models × 8 sectors × 5 CWEs × 3 prompt types × 5 replicates"
echo "Total: 6,000 evaluations"
echo "Estimated time: 1.5-2 hours"
echo "Estimated cost: $545-695"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results/optimized_run

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting at: $(date)"
echo ""

# ============================================================================
# BATCH 1: Budget Models (4 models × 600 evals = 2,400 evals)
# ============================================================================
echo "=========================================="
echo "BATCH 1: Budget Models"
echo "Models: Llama 4 Maverick, Llama 3.3 70B, Gemma 3 27B, Qwen3 32B"
echo "Estimated time: 30-40 minutes"
echo "Estimated cost: $65-85"
echo "=========================================="

python -m secdrift.runner \
  --model llama-4-maverick \
  --model llama-3-3-70b \
  --model gemma-3-27b \
  --model qwen3-32b \
  --replicates 5 \
  --workers 40 \
  --output results/optimized_run/batch1_budget_${TIMESTAMP}.jsonl &

BATCH1_PID=$!
echo "Batch 1 started (PID: $BATCH1_PID)"
echo ""

# ============================================================================
# BATCH 2: Mid-Tier Models (3 models × 600 evals = 1,800 evals)
# ============================================================================
echo "=========================================="
echo "BATCH 2: Mid-Tier Models"
echo "Models: DeepSeek R1, GPT-OSS 120B, Mistral Large 3"
echo "Estimated time: 30-40 minutes"
echo "Estimated cost: $150-190"
echo "=========================================="

python -m secdrift.runner \
  --model deepseek-r1 \
  --model gpt-oss-120b \
  --model mistral-large-3 \
  --replicates 5 \
  --workers 30 \
  --output results/optimized_run/batch2_midtier_${TIMESTAMP}.jsonl &

BATCH2_PID=$!
echo "Batch 2 started (PID: $BATCH2_PID)"
echo ""

# Wait for first batch to complete
echo "Waiting for Batch 1 or Batch 2 to complete..."
wait -n
echo "First batch completed!"
echo ""

# ============================================================================
# BATCH 3: Premium Models (3 models × 600 evals = 1,800 evals)
# ============================================================================
echo "=========================================="
echo "BATCH 3: Premium Models"
echo "Models: Claude Sonnet 4.6, Claude Opus 4.6, Kimi K2.5"
echo "Estimated time: 40-50 minutes"
echo "Estimated cost: $330-420"
echo "=========================================="

python -m secdrift.runner \
  --model claude-sonnet-4-6 \
  --model claude-opus-4-6 \
  --model kimi-k2-5 \
  --replicates 5 \
  --workers 30 \
  --output results/optimized_run/batch3_premium_${TIMESTAMP}.jsonl

echo "Batch 3 completed!"
echo ""

# Wait for remaining batches
echo "Waiting for remaining batches to complete..."
wait

echo ""
echo "=========================================="
echo "All batches completed!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""

# ============================================================================
# Merge Results
# ============================================================================
echo "Merging results..."
cat results/optimized_run/batch1_budget_${TIMESTAMP}.jsonl \
    results/optimized_run/batch2_midtier_${TIMESTAMP}.jsonl \
    results/optimized_run/batch3_premium_${TIMESTAMP}.jsonl \
    > results/full_benchmark_10models_5rep_${TIMESTAMP}.jsonl

echo "Merged results saved to: results/full_benchmark_10models_5rep_${TIMESTAMP}.jsonl"
echo ""

# ============================================================================
# Generate Analysis
# ============================================================================
echo "=========================================="
echo "Generating analysis..."
echo "=========================================="

python -m secdrift.analysis results/full_benchmark_10models_5rep_${TIMESTAMP}.jsonl \
  > results/analysis_10models_5rep_${TIMESTAMP}.txt

echo "Analysis saved to: results/analysis_10models_5rep_${TIMESTAMP}.txt"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "BENCHMARK COMPLETE!"
echo "=========================================="
echo ""
echo "Results files:"
echo "  - Full results: results/full_benchmark_10models_5rep_${TIMESTAMP}.jsonl"
echo "  - Analysis: results/analysis_10models_5rep_${TIMESTAMP}.txt"
echo ""
echo "Next steps:"
echo "  1. Review analysis: cat results/analysis_10models_5rep_${TIMESTAMP}.txt"
echo "  2. Generate figures: python -m secdrift.visualize results/full_benchmark_10models_5rep_${TIMESTAMP}.jsonl -o figures/"
echo ""
echo "Total evaluations: 6,000"
echo "Models: 10"
echo "Replicates: 5 per condition"
echo "Prompt types: 3 (baseline, matched_baseline, industry)"
echo ""
