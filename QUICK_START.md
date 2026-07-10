# Quick Start

Two paths: **(A) reproduce the paper's analysis** from the shipped data (no API keys,
a few seconds), or **(B) re-run the benchmark** end-to-end (needs model access).

## A. Reproduce the paper (no credentials)

```bash
# 1. Install
pip install -e .

# 2. Regenerate the headline result + sensitivity analysis
python analysis/cwe_exclusion.py
```

Expected (from `results/latest_run/merged_results_5rep.jsonl`, code-only filter):

```
subset                      baseline  industry    drift           p        h
(a) All 9 CWEs                 14.0%     11.4%    -2.7pp      0.2368  -0.0799
(b) Excl. CWE-502               3.8%      2.4%    -1.4pp      0.2411  -0.0821
(c) Excl. CWE-502, CWE-22       2.2%      2.5%    +0.4pp      1.0000  +0.0246
```

The script hard-asserts the 14.0% / 11.4% / −2.7pp headline and stops if it does not
reproduce, so a clean run confirms the environment.

### Regenerate the rest of the paper's numbers

```bash
python analysis/leave_one_cwe_out.py       # leave-one-CWE-out robustness
python analysis/cramers_v.py               # omnibus association tests
python analysis/mixed_effects.py           # mixed-effects logistic regression
python analysis/length_confound.py         # prompt-length confound
python analysis/completeness.py            # per-model/condition completeness
python analysis/functional_correctness.py  # Tier-1 static functional quality
python analysis/functional_exec.py         # Tier-2 sandboxed functional pass@1
python analysis/contract_drift.py          # interface-drift audit
python analysis/analyze_control_pilot.py   # placebo (non-CISA) control
python analysis/aggregation_sensitivity.py # aggregation-rule sensitivity (Bandit AND Semgrep; Bandit@MEDIUM)
python analysis/firth_permutation.py       # Firth + permutation robustness on the condition effect
```

All read JSONL directly and print the values used in the paper. Cached CSV/JSON
outputs live in `results/extended_analysis/`.

## B. Re-run the benchmark (needs model access)

Generation defaults to AWS Bedrock.

```bash
export AWS_REGION=us-east-1        # credentials via AWS CLI or IAM role

# Pilot: 2 models, 2 sectors (fast sanity check)
python -m secdrift.runner \
  --model qwen3-32b --model llama-4-maverick \
  --sectors healthcare nuclear \
  --replicates 5 \
  --output results/pilot.jsonl

# Full study: the 7 paper models x 8 sectors x 9 CWEs x 3 conditions x 5 replicates
python -m secdrift.runner --group paper --replicates 5 \
  --output results/rerun.jsonl

# Analyze
python -m secdrift.analysis results/rerun.jsonl
```

Prompt conditions per scenario: `baseline` (neutral), `matched_baseline` (sector
terminology only), `industry` (full framing). Results stream to JSONL incrementally,
so a run can be interrupted and resumed.

## Troubleshooting

- **Import errors:** `pip install -e .` (installs the `secdrift` package).
- **Bandit/Semgrep missing:** `pip install bandit semgrep` (also pulled in by the install).
- **Rate limits / cost during a re-run:** lower `parallel_workers` and `rate_limit_rpm`
  in `config/models.yaml`, or restrict to a subset with `--model`/`--sectors`.
- **Reproduction mismatch:** confirm you are reading
  `results/latest_run/merged_results_5rep.jsonl`; `analysis/cwe_exclusion.py` will halt
  with a diff if the headline numbers do not match.
