#!/usr/bin/env python3
"""Cramer's V effect sizes for the central contingency tables.

The Methods section defines Cramer's V but the results tables never reported
it (a reviewer flagged this "specified but not applied" gap). This script
computes chi-square, Cramer's V, and its p-value for the tables that carry the
paper's central claims:

    (1) condition x outcome, 3x2  (baseline / matched / industry) x (vuln / not)
    (2) baseline vs industry, 2x2 (V == phi for a 2x2 table)
    (3) sector x outcome among industry prompts, 8x2
    (4) sector x outcome including baseline as a 9th group, 9x2

Applies the paper's code-only filter (drop gpt-oss-120b + empty generated_code).
"""

import json
import math
import os
import sys
from collections import defaultdict

from scipy.stats import chi2_contingency

DATA_CANDIDATES = [
    "data/merged_results_5rep.jsonl",
    "results/latest_run/merged_results_5rep.jsonl",
]


def find_data():
    for c in DATA_CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit("ERROR: merged_results_5rep.jsonl not found")


def load(path):
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def code_only(recs):
    out = []
    for r in recs:
        if r.get("model") == "gpt-oss-120b":
            continue
        if str(r.get("generated_code") or "").strip() == "":
            continue
        out.append(r)
    return out


def cramers_v(table):
    """Cramer's V for an r x c table (list of [pos, neg] rows)."""
    chi2, p, dof, _ = chi2_contingency(table, correction=False)
    n = sum(sum(row) for row in table)
    r = len(table)
    c = len(table[0])
    v = math.sqrt(chi2 / (n * min(r - 1, c - 1)))
    return chi2, p, dof, v, n


def outcome_row(recs, pred):
    v = sum(1 for r in recs if pred(r) and r.get("is_vulnerable"))
    n = sum(1 for r in recs if pred(r))
    return [v, n - v]


def report(label, table):
    chi2, p, dof, v, n = cramers_v(table)
    print(f"{label}")
    print(f"  table   : {table}")
    print(f"  n={n}  chi2={chi2:.3f}  dof={dof}  p={p:.4f}  Cramer's V={v:.4f}")
    print()


def main():
    recs = code_only(load(find_data()))

    # (1) condition x outcome (3x2)
    conds = ["baseline", "matched_baseline", "industry"]
    t1 = [outcome_row(recs, lambda r, c=c: r.get("prompt_type") == c) for c in conds]
    report("(1) condition x outcome  [baseline / matched / industry]", t1)

    # (2) baseline vs industry (2x2); V == |phi|
    t2 = [outcome_row(recs, lambda r: r.get("prompt_type") == "baseline"),
          outcome_row(recs, lambda r: r.get("prompt_type") == "industry")]
    report("(2) baseline vs industry (2x2)", t2)

    # (3) sector x outcome among industry prompts (8x2)
    sectors = sorted(set(r.get("sector") for r in recs
                         if r.get("prompt_type") == "industry" and r.get("sector")))
    t3 = [outcome_row(recs, lambda r, s=s: r.get("prompt_type") == "industry"
                      and r.get("sector") == s) for s in sectors]
    report(f"(3) sector x outcome, industry only ({len(sectors)}x2)  "
           f"sectors={sectors}", t3)

    # (4) sector x outcome incl baseline as 9th group (9x2)
    t4 = [outcome_row(recs, lambda r: r.get("prompt_type") == "baseline")] + t3
    report(f"(4) baseline + {len(sectors)} industry sectors x outcome "
           f"({len(sectors) + 1}x2)", t4)


if __name__ == "__main__":
    main()
