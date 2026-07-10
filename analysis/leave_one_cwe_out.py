#!/usr/bin/env python3
"""Leave-one-CWE-out sensitivity for the baseline-vs-industry drift.

Complements analysis/cwe_exclusion.py (which does the nested exclusion of the
two dominant categories). Here we drop EACH CWE in turn and recompute the
baseline vs industry rates, the drift, a two-tailed Fisher's exact p-value,
and Cohen's h. This directly answers the reviewer request for a
leave-one-CWE-out check.

Applies the paper's code-only filter (drop gpt-oss-120b + empty generated_code).
"""

import json
import math
import os
import sys
from collections import defaultdict

from scipy.stats import fisher_exact, norm

DATA_CANDIDATES = [
    "data/merged_results_5rep.jsonl",
    "results/latest_run/merged_results_5rep.jsonl",
]
Z = norm.ppf(0.975)


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


def cohens_h(p1, p2):
    phi = lambda p: 2 * math.asin(math.sqrt(p))
    return phi(p1) - phi(p2)


def rates(recs, exclude_cwe=None):
    c = defaultdict(lambda: [0, 0])
    for r in recs:
        if exclude_cwe is not None and r.get("cwe") == exclude_cwe:
            continue
        pt = r.get("prompt_type")
        if pt not in ("baseline", "industry"):
            continue
        c[pt][0] += int(bool(r.get("is_vulnerable")))
        c[pt][1] += 1
    return c


def analyze(c):
    bv, bn = c["baseline"]
    iv, ino = c["industry"]
    pb, pi = bv / bn, iv / ino
    _, p = fisher_exact([[bv, bn - bv], [iv, ino - iv]], alternative="two-sided")
    return {
        "b_rate": 100 * pb, "i_rate": 100 * pi,
        "drift": 100 * (pi - pb), "p": p, "h": cohens_h(pi, pb),
        "bn": bn, "ino": ino,
    }


def main():
    recs = code_only(load(find_data()))
    cwes = sorted(set(r.get("cwe") for r in recs))

    full = analyze(rates(recs))
    print(f"Full (all 9 CWEs): baseline {full['b_rate']:.1f}%  "
          f"industry {full['i_rate']:.1f}%  drift {full['drift']:+.1f}pp  "
          f"p={full['p']:.4f}  h={full['h']:+.4f}")
    print()
    print(f"{'dropped CWE':<12}{'base':>8}{'ind':>8}{'drift':>9}"
          f"{'p':>10}{'h':>9}{'base_n':>9}{'ind_n':>8}")
    print("-" * 73)
    for cwe in cwes:
        a = analyze(rates(recs, exclude_cwe=cwe))
        print(f"{cwe:<12}{a['b_rate']:>7.1f}%{a['i_rate']:>7.1f}%"
              f"{a['drift']:>8.1f}p{a['p']:>10.4f}{a['h']:>9.4f}"
              f"{a['bn']:>9}{a['ino']:>8}")


if __name__ == "__main__":
    main()
