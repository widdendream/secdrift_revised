#!/usr/bin/env python3
"""Quantify prompt-level contract/interface drift across conditions.

The paper's D5 ("Requirement Preservation") originally claimed the Requirements
and Example sections were byte-identical across baseline / matched / industry.
This script tests that claim directly from the stored prompts by measuring, per
condition, the fraction of prompts that retain the structured signature (an
explicit "Accept parameter(s):" line) and the worked "Example usage:" call --
the two elements that define the task's interface.

It also reports, per CWE task, whether those elements are present in the
baseline vs the industry prompt, to support the per-task interface-drift table.

Uses the full dataset (all prompts); prompt structure is model-independent, so
no code-only filter is needed.
"""

import json
import re
from collections import defaultdict

DATA = "results/latest_run/merged_results_5rep.jsonl"
CONDS = ["baseline", "matched_baseline", "industry"]
CWES = ["CWE-89", "CWE-78", "CWE-502", "CWE-22", "CWE-327",
        "CWE-79", "CWE-798", "CWE-330", "CWE-295"]

HAS_PARAMS = re.compile(r"Accept parameters?:", re.I)
HAS_EXAMPLE = re.compile(r"Example usage:", re.I)


def load():
    with open(DATA) as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    recs = load()

    # per-condition fraction with explicit signature / worked example
    cond_tot = defaultdict(int)
    cond_par = defaultdict(int)
    cond_ex = defaultdict(int)
    # unique prompt text per (cwe, cond) to check per-task presence
    seen = {}
    for r in recs:
        c = r.get("prompt_type")
        pt = r.get("prompt_text") or ""
        cond_tot[c] += 1
        if HAS_PARAMS.search(pt):
            cond_par[c] += 1
        if HAS_EXAMPLE.search(pt):
            cond_ex[c] += 1
        seen.setdefault((r.get("cwe"), c), pt)

    print("Prompt-level interface preservation by condition:")
    print(f"{'condition':<18}{'n':>6}{'has Accept-params':>20}{'has Example-usage':>20}")
    print("-" * 64)
    for c in CONDS:
        n = cond_tot[c]
        print(f"{c:<18}{n:>6}{100*cond_par[c]/n:>19.1f}%{100*cond_ex[c]/n:>19.1f}%")

    print("\nPer-task presence (baseline vs industry):")
    print(f"{'CWE':<10}{'base params':>13}{'base example':>14}"
          f"{'ind params':>13}{'ind example':>14}")
    print("-" * 64)
    for cwe in CWES:
        bp = "yes" if HAS_PARAMS.search(seen.get((cwe, "baseline"), "")) else "no"
        be = "yes" if HAS_EXAMPLE.search(seen.get((cwe, "baseline"), "")) else "no"
        ip = "yes" if HAS_PARAMS.search(seen.get((cwe, "industry"), "")) else "no"
        ie = "yes" if HAS_EXAMPLE.search(seen.get((cwe, "industry"), "")) else "no"
        print(f"{cwe:<10}{bp:>13}{be:>14}{ip:>13}{ie:>14}")


if __name__ == "__main__":
    main()
