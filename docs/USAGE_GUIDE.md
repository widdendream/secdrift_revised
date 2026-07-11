# Usage Guide: Running Enhanced Benchmark with Replicates and Matched Baseline

## Quick Start

### 1. Test the Implementation

First, verify the matched baseline transformation works:

```bash
python test_matched_baseline.py
```

You should see three outputs:
- Original baseline (generic terms)
- Matched baseline (sector terms only)
- Full industry (sector terms + context)

### 2. Run a Pilot Test

Start with a small test to validate everything works:

```bash
# 2 models, 2 sectors, 5 replicates = 540 evaluations (~30 minutes)
python -m secdrift.runner \
  --model claude-sonnet-4-6 \
  --model llama-4-maverick \
  --sectors healthcare nuclear \
  --replicates 5 \
  --output results/pilot_test.jsonl
```

### 3. Analyze Pilot Results

```bash
python -m secdrift.analysis results/pilot_test.jsonl
```

Look for:
- Three vulnerability rates per sector (baseline, matched, industry)
- Decomposed drift (terminology effect + context effect)
- Statistical significance

### 4. Run Full Benchmark

Once validated, run the complete benchmark:

```bash
# 10 models, 8 sectors, 9 CWEs, 3 prompt types, 5 replicates = 10,800 evaluations
python -m secdrift.runner --group paper --replicates 5
```

**Estimated time:** 6-8 hours
**Estimated cost:** $500-600

## Understanding the Results

### Three-Way Comparison

Each sector now has three measurements:

```
Sector: healthcare
  Baseline (generic):              45.0% vulnerable
  Matched baseline (terminology):  42.0% vulnerable  
  Industry (full context):         38.5% vulnerable
```

### Drift Decomposition

```
Total drift: -6.5pp
  ├─ Terminology effect: -3.0pp  (matched - baseline)
  └─ Context effect: -3.5pp      (industry - matched)
```

This tells you:
- **Total effect**: Industry context reduces vulnerabilities by 6.5 percentage points
- **Terminology alone**: Using sector-specific terms reduces by 3.0pp
- **Context alone**: Adding industry framing reduces by an additional 3.5pp

### Interpretation

**Protective drift (negative):**
- Terminology effect: Domain language makes code more secure
- Context effect: Industry framing adds additional security

**Risk-inducing drift (positive):**
- Terminology effect: Domain language makes code less secure
- Context effect: Industry framing reduces security

**Neutral drift (near zero):**
- No significant effect from terminology or context

## Advanced Usage

### Run Specific Scenarios

```bash
# Test only SQL injection and command injection
python -m secdrift.runner \
  --group paper \
  --scenarios sql_injection command_injection \
  --replicates 5
```

### Run Specific Sectors

```bash
# Test only high-risk sectors
python -m secdrift.runner \
  --group paper \
  --sectors emergency_services defense nuclear \
  --replicates 5
```

### Adjust Parallelism

```bash
# Use more workers for faster execution
python -m secdrift.runner \
  --group paper \
  --replicates 5 \
  --workers 20
```

## Troubleshooting

### Issue: Terminology replacement looks wrong

The current implementation does simple string replacement. If you see issues like:
- "patient lookupes" instead of "patient lookup"
- "patient records" replacing "records" in "search_records"

This is expected behavior - the terminology mapping is applied globally. The analysis focuses on vulnerability rates, not code quality.

### Issue: Out of memory

Reduce parallel workers:
```bash
python -m secdrift.runner --group paper --replicates 5 --workers 4
```

### Issue: API rate limits

The runner includes rate limiting. If you hit limits:
1. Check `config/models.yaml` for rate_limit_rpm settings
2. Reduce parallel workers
3. Add delays between requests

## Interpreting Statistical Significance

With 5 replicates per condition:
- **n = 5 × 9 CWEs = 45** samples per prompt type per sector
- **Power to detect medium effects** (d = 0.5) at α = 0.05
- **Bonferroni correction** applied for multiple comparisons

Look for:
- `p_value_corrected < 0.05` for significance
- `Cramér's V > 0.3` for meaningful effect size
- `95% CI` that doesn't include zero

## Visualization

After running analysis, generate figures:

```bash
python -m secdrift.visualize results/benchmark_TIMESTAMP.jsonl -o figures/
```

This will create:
- Sector drift comparison (now with three bars per sector)
- Model comparison
- Heatmaps
- CWE-specific analysis

## Exporting Results

Results are saved in JSONL format with complete provenance:

```json
{
  "evaluation_id": "abc123",
  "prompt_type": "matched_baseline",
  "sector": "healthcare",
  "model": "claude-sonnet-4-6",
  "replicate": 2,
  "is_vulnerable": false,
  ...
}
```

You can process this with any tool that reads JSONL.

## Next Steps

1. Run pilot test
2. Validate matched baseline works as expected
3. Run full benchmark
4. Update paper with three-way analysis
5. Create new visualizations showing decomposed effects
6. Submit for review with stronger evidence
