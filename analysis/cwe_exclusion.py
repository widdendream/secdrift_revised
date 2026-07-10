#!/usr/bin/env python3
"""CWE-exclusion sensitivity analysis for the security-drift benchmark.

Applies the paper's code-only filter, then computes vulnerability rates and
baseline-vs-industry drift under three CWE subsets:

    (a) all 9 CWEs
    (b) excluding CWE-502
    (c) excluding CWE-502 and CWE-22

For each subset a two-tailed Fisher's exact test is run on the baseline-vs-
industry 2x2 table, alongside Cohen's h and 95% Wilson score intervals.

A sanity check requires subset (a) to reproduce the paper's headline numbers
(baseline 14.0%, industry 11.4%, drift -2.7pp). If it does not, the script
stops and prints the discrepancy instead of continuing.
"""

import json
import math
import os
import sys
from collections import defaultdict

from scipy.stats import fisher_exact, norm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_CANDIDATES = [
    "data/merged_results_5rep.jsonl",
    "results/latest_run/merged_results_5rep.jsonl",
]

CONDITIONS = ["baseline", "matched_baseline", "industry"]

SUBSETS = [
    ("(a) all 9 CWEs", set()),
    ("(b) excl CWE-502", {"CWE-502"}),
    ("(c) excl CWE-502,CWE-22", {"CWE-502", "CWE-22"}),
]

# Expected headline values for subset (a), used as a hard sanity gate.
EXPECTED_A = {"baseline": 14.0, "industry": 11.4, "drift": -2.7}

Z = norm.ppf(0.975)  # 1.959963...


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def wilson_interval(successes, n, z=Z):
    """95% Wilson score interval for a binomial proportion, returned as %."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100.0 * (center - half), 100.0 * (center + half))


def cohens_h(p1, p2):
    """Cohen's h effect size between two proportions (p1 - p2), on [0,1] scale."""
    phi = lambda p: 2.0 * math.asin(math.sqrt(p))
    return phi(p1) - phi(p2)


# --------------------------------------------------------------------------- #
# Data loading + filtering
# --------------------------------------------------------------------------- #
def find_data_path():
    for c in DATA_CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit("ERROR: could not locate merged_results_5rep.jsonl")


def load_records(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def apply_code_only_filter(records):
    """Drop gpt-oss-120b and records with empty/whitespace generated_code."""
    kept = []
    for r in records:
        if r.get("model") == "gpt-oss-120b":
            continue
        if str(r.get("generated_code") or "").strip() == "":
            continue
        kept.append(r)
    return kept


# --------------------------------------------------------------------------- #
# Per-subset computation
# --------------------------------------------------------------------------- #
def compute_subset(records, excluded_cwes):
    """Return counts[condition] = [n_vulnerable, n_total] for one CWE subset."""
    counts = defaultdict(lambda: [0, 0])
    for r in records:
        if r.get("cwe") in excluded_cwes:
            continue
        pt = r.get("prompt_type")
        if pt not in CONDITIONS:
            continue
        counts[pt][0] += int(bool(r.get("is_vulnerable")))
        counts[pt][1] += 1
    return counts


def rate(counts, condition):
    v, n = counts[condition]
    return (100.0 * v / n) if n else float("nan")


def analyze_subset(counts):
    """Compute rates, drift, Fisher p, Cohen's h and Wilson CIs for a subset."""
    res = {"counts": counts, "rates": {}, "wilson": {}}
    for cond in CONDITIONS:
        v, n = counts[cond]
        res["rates"][cond] = rate(counts, cond)
        res["wilson"][cond] = wilson_interval(v, n)

    b_v, b_n = counts["baseline"]
    i_v, i_n = counts["industry"]

    # 2x2 table: rows = baseline / industry, cols = vulnerable / not-vulnerable
    table = [[b_v, b_n - b_v], [i_v, i_n - i_v]]
    _, p_value = fisher_exact(table, alternative="two-sided")

    p_base = b_v / b_n if b_n else float("nan")
    p_ind = i_v / i_n if i_n else float("nan")

    res["drift"] = res["rates"]["industry"] - res["rates"]["baseline"]
    res["p_value"] = p_value
    res["cohens_h"] = cohens_h(p_ind, p_base)  # industry - baseline
    res["table"] = table
    return res


# --------------------------------------------------------------------------- #
# Sanity gate
# --------------------------------------------------------------------------- #
def sanity_check(res_a):
    got = {
        "baseline": round(res_a["rates"]["baseline"], 1),
        "industry": round(res_a["rates"]["industry"], 1),
        "drift": round(res_a["drift"], 1),
    }
    mismatches = {k: (got[k], EXPECTED_A[k]) for k in EXPECTED_A if got[k] != EXPECTED_A[k]}
    return got, mismatches


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    path = find_data_path()
    records = load_records(path)
    filtered = apply_code_only_filter(records)

    print(f"Data file        : {path}")
    print(f"Total records    : {len(records)}")
    print(f"After code-only  : {len(filtered)} "
          f"(dropped {len(records) - len(filtered)}: gpt-oss-120b + empty code)")
    print()

    results = []
    for label, excluded in SUBSETS:
        counts = compute_subset(filtered, excluded)
        results.append((label, analyze_subset(counts)))

    # ---- sanity gate on subset (a) -------------------------------------- #
    label_a, res_a = results[0]
    got, mismatches = sanity_check(res_a)
    if mismatches:
        print("SANITY CHECK FAILED for subset (a): rounded values do not match.")
        print(f"  {'metric':<10} {'got':>8}   {'expected':>8}")
        for k, (g, e) in mismatches.items():
            print(f"  {k:<10} {g:>8}   {e:>8}")
        print("\nFull subset (a) values (unrounded):")
        print(f"  baseline rate : {res_a['rates']['baseline']:.4f}%")
        print(f"  industry rate : {res_a['rates']['industry']:.4f}%")
        print(f"  drift         : {res_a['drift']:.4f} pp")
        print(f"  counts        : {dict(res_a['counts'])}")
        sys.exit("Stopping: subset (a) did not reproduce 14.0% / 11.4% / -2.7pp.")

    print(f"Sanity check PASSED for {label_a}: "
          f"baseline {got['baseline']}% / industry {got['industry']}% / "
          f"drift {got['drift']}pp\n")

    # ---- summary table -------------------------------------------------- #
    print("=" * 84)
    print(f"{'subset':<26}{'baseline':>10}{'industry':>10}{'drift':>9}"
          f"{'p':>12}{'h':>9}")
    print("-" * 84)
    for label, res in results:
        print(f"{label:<26}"
              f"{res['rates']['baseline']:>9.1f}%"
              f"{res['rates']['industry']:>9.1f}%"
              f"{res['drift']:>8.1f}p"
              f"{res['p_value']:>12.4f}"
              f"{res['cohens_h']:>9.4f}")
    print("=" * 84)

    # ---- detail: matched_baseline rate + Wilson 95% CIs ----------------- #
    print("\nDetail per subset (rate [95% Wilson CI], n):")
    for label, res in results:
        print(f"\n{label}")
        for cond in CONDITIONS:
            v, n = res["counts"][cond]
            lo, hi = res["wilson"][cond]
            print(f"  {cond:<17} {res['rates'][cond]:6.2f}%  "
                  f"[{lo:5.2f}%, {hi:5.2f}%]  (vuln {v}/{n})")
        print(f"  drift (ind-base)  {res['drift']:+.2f} pp   "
              f"Fisher p={res['p_value']:.4f}   Cohen's h={res['cohens_h']:+.4f}")


if __name__ == "__main__":
    main()
