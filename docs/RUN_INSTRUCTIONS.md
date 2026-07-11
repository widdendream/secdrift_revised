# Run Instructions: 10 Models with 5 Replicates

## Summary

**Total Cost:** $545-695
**Total Time:** 1.5-2 hours (optimized)
**Total Evaluations:** 6,000

## What You're Running

- **10 models** (from your paper)
- **8 sectors** (all CISA critical infrastructure)
- **5 CWE scenarios** (currently defined)
- **3 prompt types** (baseline, matched_baseline, industry)
- **5 replicates** per condition

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
./run_optimized_benchmark.sh
```

This will:
1. Run 3 batches in parallel (staggered start)
2. Automatically merge results
3. Generate analysis report
4. Complete in ~1.5-2 hours

### Option 2: Manual Control

If you want more control, run batches individually:

**Batch 1 (Budget models - ~30 min, ~$65-85):**
```bash
python -m secdrift.runner \
  --model llama-4-maverick \
  --model llama-3-3-70b \
  --model gemma-3-27b \
  --model qwen3-32b \
  --replicates 5 \
  --workers 40 \
  --output results/batch1_budget.jsonl
```

**Batch 2 (Mid-tier - ~30 min, ~$150-190):**
```bash
python -m secdrift.runner \
  --model deepseek-r1 \
  --model gpt-oss-120b \
  --model mistral-large-3 \
  --replicates 5 \
  --workers 30 \
  --output results/batch2_midtier.jsonl
```

**Batch 3 (Premium - ~45 min, ~$330-420):**
```bash
python -m secdrift.runner \
  --model claude-sonnet-4-6 \
  --model claude-opus-4-6 \
  --model kimi-k2-5 \
  --replicates 5 \
  --workers 30 \
  --output results/batch3_premium.jsonl
```

## Cost Breakdown

| Tier | Models | Evaluations | Est. Cost | Time |
|------|--------|-------------|-----------|------|
| Budget | 4 models | 2,400 | $65-85 | 30 min |
| Mid-tier | 3 models | 1,800 | $150-190 | 30 min |
| Premium | 3 models | 1,800 | $330-420 | 45 min |
| **Total** | **10 models** | **6,000** | **$545-695** | **1.5-2 hrs** |

## Optimization Details

### What Changed
- `parallel_workers`: 15 → 40 (faster execution)
- `rate_limit_rpm`: 60 → 100 (higher throughput)
- Batched execution to avoid rate limits

### Why This Works
- AWS Bedrock allows ~100 RPM per model
- Running 3-4 models in parallel = 300-400 total RPM
- Staggered batches prevent hitting account-wide limits
- Each batch completes before next starts

## Monitoring Progress

The runner shows real-time progress:
```
Running evaluations: 45%|████████████▌              | 1080/2400 [15:23<17:12, 1.28it/s]
```

You can also check intermediate results:
```bash
# Count completed evaluations
wc -l results/optimized_run/batch1_budget_*.jsonl

# Check for errors
grep '"error"' results/optimized_run/batch1_budget_*.jsonl
```

## What You'll Get

### 1. Raw Results (JSONL)
```
results/full_benchmark_10models_5rep_TIMESTAMP.jsonl
```

Each line contains:
- Model name
- Sector
- Scenario (CWE)
- Prompt type (baseline/matched_baseline/industry)
- Replicate number
- Generated code
- Vulnerabilities detected
- Latency

### 2. Analysis Report (TXT)
```
results/analysis_10models_5rep_TIMESTAMP.txt
```

Shows:
- Three-way vulnerability rates per sector
- Decomposed drift (terminology + context effects)
- Statistical significance
- Effect sizes
- Per-model breakdowns

### 3. Example Output

```
healthcare:
  Baseline (generic): 45.0%
  Matched baseline (terminology): 42.0%
  Industry (terminology + context): 38.5%
  
  Total drift: -6.5pp *
    - Terminology effect: -3.0pp
    - Context effect: -3.5pp
  95% CI: [-8.2, -4.8]
  p-value: 0.0012 (corrected: 0.0096)
  Classification: protective
  N: 450 industry, 450 baseline, 450 matched
```

## After Completion

### 1. Review Analysis
```bash
cat results/analysis_10models_5rep_TIMESTAMP.txt
```

### 2. Generate Visualizations
```bash
python -m secdrift.visualize \
  results/full_benchmark_10models_5rep_TIMESTAMP.jsonl \
  -o figures/
```

### 3. Merge with Previous Results (Optional)
If you want to combine with old results:
```bash
cat results/benchmark_20260223_224221.jsonl \
    results/full_benchmark_10models_5rep_TIMESTAMP.jsonl \
    > results/combined_results.jsonl
```

## Troubleshooting

### If you hit rate limits:
```bash
# Reduce workers
python -m secdrift.runner --model MODEL --replicates 5 --workers 20
```

### If a batch fails:
Results are saved incrementally, so you can resume:
```bash
# Check what completed
wc -l results/optimized_run/batch1_budget_*.jsonl

# Rerun just the failed batch
python -m secdrift.runner --model MODEL --replicates 5 --output results/retry.jsonl
```

### If you want to test first:
Run a quick test with 1 model, 1 sector, 1 replicate:
```bash
python -m secdrift.runner \
  --model llama-4-maverick \
  --sectors healthcare \
  --replicates 1 \
  --output results/test.jsonl
```

## Ready to Start?

Just run:
```bash
./run_optimized_benchmark.sh
```

Or if you prefer manual control, start with Batch 1:
```bash
python -m secdrift.runner \
  --model llama-4-maverick \
  --model llama-3-3-70b \
  --model gemma-3-27b \
  --model qwen3-32b \
  --replicates 5 \
  --workers 40 \
  --output results/batch1_budget.jsonl
```

The script will handle everything automatically and give you a complete analysis at the end!
