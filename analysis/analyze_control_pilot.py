#!/usr/bin/env python3
"""Analyze the control-sector pilot (results/control_pilot.jsonl).

The pilot ran the INDUSTRY condition only for two placebo/control sectors
(e-commerce, education) that were length- and specificity-matched to the 8
CISA critical-infrastructure sectors. The question this answers:

    Is the apparent "industry drift" driven by CISA-sector *identity*, or by
    the generic operational-specificity of an industry-framed prompt?

Reference anchors from the main analysis (code-only filter, 5 reps):
    baseline         14.0%  (33/235)
    CISA industry    11.4%  (223/1959)     -> CISA drift = -2.7pp (Fisher p=0.24, ns)

Interpretation (confirmed with the user):
    * control industry rate ~ CISA industry (11.4%)  -> specificity drives the
      effect: an industry framing lowers the rate regardless of which sector,
      so the drift is NOT about CISA-sector identity.
    * control industry rate ~ baseline (14.0%), drift ~ 0, while CISA stays
      negative -> sector identity is real: only genuine CISA sectors move the
      rate.

Applies the paper's code-only filter (drop gpt-oss-120b + empty generated_code).
"""

import json
import math
import os
import sys
from collections import defaultdict

from scipy.stats import fisher_exact, norm

PILOT_PATH = "results/control_pilot.jsonl"

# Anchors from the main (CISA) analysis, code-only filter.
BASELINE_V, BASELINE_N = 33, 235          # 14.04%
CISA_IND_V, CISA_IND_N = 223, 1959        # 11.38%

Z = norm.ppf(0.975)


def wilson(v, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"))
    p = v / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100 * (center - half), 100 * (center + half))


def cohens_h(p1, p2):
    phi = lambda p: 2 * math.asin(math.sqrt(p))
    return phi(p1) - phi(p2)


def fisher_two_sided(v1, n1, v2, n2):
    table = [[v1, n1 - v1], [v2, n2 - v2]]
    _, p = fisher_exact(table, alternative="two-sided")
    return p


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def code_only(records):
    kept = []
    for r in records:
        if r.get("model") == "gpt-oss-120b":
            continue
        if str(r.get("generated_code") or "").strip() == "":
            continue
        kept.append(r)
    return kept


def rate_block(label, v, n):
    lo, hi = wilson(v, n)
    r = 100 * v / n if n else float("nan")
    return f"  {label:<22} {r:6.2f}%  [{lo:5.2f}%, {hi:5.2f}%]  (vuln {v}/{n})"


def main():
    if not os.path.exists(PILOT_PATH):
        sys.exit(f"ERROR: {PILOT_PATH} not found")

    raw = load(PILOT_PATH)
    errors = sum(1 for r in raw if r.get("error"))
    print(f"Pilot file        : {PILOT_PATH}")
    print(f"Records           : {len(raw)}  (expected 630)")
    print(f"Errors            : {errors}")

    # code-only filter
    filt = code_only(raw)
    print(f"After code-only   : {len(filt)} "
          f"(dropped {len(raw) - len(filt)}: gpt-oss-120b + empty code)")

    # sanity: everything should be industry condition
    conds = sorted(set(r.get("prompt_type") for r in filt))
    print(f"Conditions present: {conds}")
    print()

    # -- overall pooled control industry rate ----------------------------- #
    v = sum(1 for r in filt if r.get("is_vulnerable"))
    n = len(filt)
    print("Control-sector INDUSTRY vulnerability rate (pooled):")
    print(rate_block("control (pooled)", v, n))
    print(rate_block("baseline (anchor)", BASELINE_V, BASELINE_N))
    print(rate_block("CISA industry (anchor)", CISA_IND_V, CISA_IND_N))
    print()

    ctrl_rate = 100 * v / n
    drift_vs_base = ctrl_rate - 100 * BASELINE_V / BASELINE_N
    p_vs_base = fisher_two_sided(v, n, BASELINE_V, BASELINE_N)
    h_vs_base = cohens_h(v / n, BASELINE_V / BASELINE_N)
    p_vs_cisa = fisher_two_sided(v, n, CISA_IND_V, CISA_IND_N)
    h_vs_cisa = cohens_h(v / n, CISA_IND_V / CISA_IND_N)

    print("Comparisons (two-tailed Fisher, Cohen's h):")
    print(f"  control vs baseline    : drift {drift_vs_base:+.2f}pp   "
          f"p={p_vs_base:.4f}   h={h_vs_base:+.4f}")
    print(f"  control vs CISA industry: diff  "
          f"{ctrl_rate - 100 * CISA_IND_V / CISA_IND_N:+.2f}pp   "
          f"p={p_vs_cisa:.4f}   h={h_vs_cisa:+.4f}")
    print()

    # -- per control sector ---------------------------------------------- #
    by_sector = defaultdict(lambda: [0, 0])
    for r in filt:
        s = r.get("sector")
        by_sector[s][0] += int(bool(r.get("is_vulnerable")))
        by_sector[s][1] += 1
    print("Per control sector (industry):")
    for s in sorted(by_sector):
        sv, sn = by_sector[s]
        print(rate_block(s, sv, sn))
    print()

    # -- per CWE --------------------------------------------------------- #
    by_cwe = defaultdict(lambda: [0, 0])
    for r in filt:
        c = r.get("cwe")
        by_cwe[c][0] += int(bool(r.get("is_vulnerable")))
        by_cwe[c][1] += 1
    print("Per CWE (industry, control sectors pooled):")
    for c in sorted(by_cwe):
        cv, cn = by_cwe[c]
        print(rate_block(c, cv, cn))
    print()

    # -- verdict --------------------------------------------------------- #
    d_base = abs(drift_vs_base)
    d_cisa = abs(ctrl_rate - 100 * CISA_IND_V / CISA_IND_N)
    print("=" * 68)
    if d_cisa <= d_base:
        print("READING: control industry rate is closer to the CISA industry")
        print("rate than to baseline -> SPECIFICITY drives the effect (an")
        print("industry framing lowers the rate regardless of sector identity).")
    else:
        print("READING: control industry rate is closer to baseline than to the")
        print("CISA industry rate -> SECTOR IDENTITY is real (only genuine CISA")
        print("sectors move the rate; a matched generic industry framing does not).")
    print("Note: the CISA drift itself was not significant (p=0.24); treat this")
    print("as a descriptive placebo comparison, not a significance test.")
    print("=" * 68)


if __name__ == "__main__":
    main()
