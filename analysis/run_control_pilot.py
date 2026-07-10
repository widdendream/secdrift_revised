#!/usr/bin/env python3
"""Industry-only control-sector pilot runner.

Runs ONLY the industry condition for the two control sectors (ecommerce,
education) across the 7 paper models x 9 CWEs x 5 replicates = 630 evaluations,
writing to results/control_pilot.jsonl.

The stock ``BenchmarkRunner.run_benchmark`` always generates baseline +
matched_baseline + industry (which for these two sectors would be ~1,575
evaluations). This script constructs ONLY the 630 industry tasks so the pilot
is exactly industry-only, then reuses the runner's per-evaluation machinery
(model generation + Bandit/Semgrep detection).

Requires valid AWS Bedrock credentials for the live run.

Usage:
    # verify task construction only (no API calls, no credentials needed):
    python3 analysis/run_control_pilot.py --dry-run

    # live run (needs valid AWS Bedrock credentials):
    python3 analysis/run_control_pilot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCENARIOS_DIR = _REPO_ROOT / "scenarios"

PILOT_MODELS = [
    "qwen3-32b", "llama-4-maverick", "llama-3-3-70b", "gemma-3-27b",
    "mistral-large-3", "gpt-oss-120b", "deepseek-r1",
]
CONTROL_SECTORS = ["ecommerce", "education"]


def load_industry_prompts(sectors):
    out = {}
    for s in sectors:
        data = yaml.safe_load(open(SCENARIOS_DIR / f"industry_{s}.yaml"))
        out[s] = {
            sid: {"prompt": b["industry_prompt"], "cwe": b.get("cwe", "unknown")}
            for sid, b in data["scenarios"].items()
        }
    return out


def build_tasks(models, sectors, prompts, replicates):
    tasks = []
    for sector in sectors:
        for scenario_id, sd in prompts[sector].items():
            for model_name in models:
                for rep in range(replicates):
                    tasks.append({
                        "scenario_id": scenario_id,
                        "cwe": sd["cwe"],
                        "prompt_type": "industry",
                        "sector": sector,
                        "model_name": model_name,
                        "prompt_text": sd["prompt"],
                        "replicate": rep,
                    })
    return tasks


def main():
    ap = argparse.ArgumentParser(description="Industry-only control pilot")
    ap.add_argument("-o", "--output", default=str(_REPO_ROOT / "results" / "control_pilot.jsonl"))
    ap.add_argument("-r", "--replicates", type=int, default=5)
    ap.add_argument("-w", "--workers", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build tasks and report counts; make no API calls")
    args = ap.parse_args()

    prompts = load_industry_prompts(CONTROL_SECTORS)
    for s in CONTROL_SECTORS:
        n = len(prompts[s])
        assert n == 9, f"{s} has {n} scenarios (expected 9)"

    tasks = build_tasks(PILOT_MODELS, CONTROL_SECTORS, prompts, args.replicates)

    print(f"models: {len(PILOT_MODELS)}  sectors: {len(CONTROL_SECTORS)}  "
          f"cwes: 9  condition: industry-only  replicates: {args.replicates}")
    print(f"total tasks: {len(tasks)}")
    print("by sector:", dict(Counter(t["sector"] for t in tasks)))
    print("by model :", dict(Counter(t["model_name"] for t in tasks)))
    print("prompt_types:", sorted(set(t["prompt_type"] for t in tasks)))

    if args.dry_run:
        print("\nDRY RUN -- no API calls made.")
        print("sample prompt [ecommerce/sql_injection]:")
        print("-" * 60)
        print(prompts["ecommerce"]["sql_injection"]["prompt"].rstrip())
        return

    # ---- live run (requires valid AWS Bedrock credentials) ----
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataclasses import asdict
    from tqdm import tqdm
    from secdrift.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        models=PILOT_MODELS,
        parallel_workers=args.workers,
        replicates=args.replicates,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    results, errors = [], 0
    with ThreadPoolExecutor(max_workers=runner.parallel_workers) as ex:
        futs = {ex.submit(runner.run_evaluation, **t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="control pilot"):
            r = fut.result()
            results.append(r)
            if r.error:
                errors += 1
            with open(out, "a") as f:
                f.write(json.dumps(asdict(r)) + "\n")

    vuln = sum(1 for r in results if r.is_vulnerable)
    print(f"\ndone: {len(results)} evaluations, vulnerable={vuln}, errors={errors}")
    print(f"output: {out}")
    if errors:
        print("NOTE: errors > 0 -- check AWS Bedrock credentials/access if all failed.")


if __name__ == "__main__":
    main()
