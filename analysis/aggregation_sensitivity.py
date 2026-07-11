#!/usr/bin/env python3
"""Aggregation-rule sensitivity for the SecDrift headline (RQ1).

Reviewer ask: recompute the baseline-vs-industry headline under stricter
detection-aggregation rules and confirm whether the null holds.

Rules compared (all on the code-only sample, GPT-OSS excluded):

  (0) UNION  (current)          : Bandit(-ll) OR Semgrep  -> stored is_vulnerable
  (1) AND    (agreement)        : Bandit AND Semgrep       -> detection_tool == "Bandit + Semgrep"
  (2) Bandit @ MEDIUM confidence: re-run Bandit with -ll -ii (severity+confidence MEDIUM),
                                   flag on the same 9-rule CWE mapping (single tool)
  (3) Bandit@MEDIUM OR Semgrep  : rule (2) unioned with the Semgrep flag

Rule (0) must reproduce the paper headline (baseline 14.0%, industry 11.4%,
drift -2.7pp) as a sanity gate.

Only records currently flagged by Bandit can lose their flag when the Bandit
confidence threshold is raised (tightening never *adds* findings), so the
Bandit re-run is applied to every code-only record for completeness but the
outcome is driven by the currently-flagged set. Semgrep flagged exactly the
"Bandit + Semgrep" records in this dataset (no Semgrep-only detections), so the
Semgrep flag is recovered from detection_tool without re-running Semgrep.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scipy.stats import fisher_exact, norm

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "results" / "latest_run" / "merged_results_5rep.jsonl"
OUT = REPO / "results" / "extended_analysis" / "aggregation_sensitivity.json"
TXT = REPO / "results" / "extended_analysis" / "aggregation_sensitivity.txt"

Z = norm.ppf(0.975)

# Same Bandit test-id -> CWE mapping used by the benchmark's BanditAnalyzer.
BANDIT_CWE_MAPPING = {
    "B608": "CWE-89", "B602": "CWE-78", "B605": "CWE-78", "B607": "CWE-78",
    "B301": "CWE-502", "B403": "CWE-502", "B303": "CWE-327", "B324": "CWE-327",
    "B108": "CWE-22",
}
EXPECTED_UNION = {"baseline": 14.0, "industry": 11.4, "drift": -2.7}


def wilson(v, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = v / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = (Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / d
    return (100 * (c - h), 100 * (c + h))


def cohens_h(p1, p2):
    phi = lambda p: 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))
    return phi(p1) - phi(p2)


def bandit_flag_medium(code: str) -> bool:
    """Re-run Bandit at severity+confidence MEDIUM (-ll -ii); return True if any
    mapped CWE is found (single-tool flag, same mapping as the pipeline)."""
    if not code or not code.strip():
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tf = f.name
    try:
        r = subprocess.run(
            [sys.executable, "-m", "bandit", "-f", "json", "-ll", "-ii", tf],
            capture_output=True, text=True, timeout=60,
        )
        if not r.stdout:
            return False
        out = json.loads(r.stdout)
        for issue in out.get("results", []):
            if issue.get("test_id") in BANDIT_CWE_MAPPING:
                return True
        return False
    except Exception:
        return False
    finally:
        Path(tf).unlink(missing_ok=True)


def _worker(args):
    idx, code = args
    return idx, bandit_flag_medium(code)


def contrast(flags, conds):
    """flags: list[bool], conds: list[str] parallel. Return baseline vs industry."""
    counts = defaultdict(lambda: [0, 0])
    for fl, c in zip(flags, conds):
        counts[c][0] += int(fl)
        counts[c][1] += 1
    bv, bn = counts["baseline"]
    iv, ino = counts["industry"]
    _, p = fisher_exact([[bv, bn - bv], [iv, ino - iv]])
    pb, pi = bv / bn, iv / ino
    return {
        "base_v": bv, "base_n": bn, "base_rate": 100 * pb,
        "ind_v": iv, "ind_n": ino, "ind_rate": 100 * pi,
        "drift": 100 * (pi - pb), "p": float(p), "h": cohens_h(pi, pb),
        "base_ci": wilson(bv, bn), "ind_ci": wilson(iv, ino),
    }


def main():
    records = [json.loads(l) for l in open(DATA) if l.strip()]
    recs = [r for r in records
            if r.get("model") != "gpt-oss-120b"
            and str(r.get("generated_code") or "").strip() != ""]
    print(f"code-only n = {len(recs)}")

    conds = [r.get("prompt_type") for r in recs]
    union_flag = [bool(r.get("is_vulnerable")) for r in recs]
    and_flag = [r.get("detection_tool") == "Bandit + Semgrep" for r in recs]
    semgrep_flag = [r.get("detection_tool") == "Bandit + Semgrep" for r in recs]

    # Re-run Bandit @ MEDIUM confidence on every code-only record (parallel).
    codes = [(i, r.get("generated_code") or "") for i, r in enumerate(recs)]
    bandit_med = [False] * len(recs)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for idx, fl in ex.map(_worker, codes, chunksize=16):
            bandit_med[idx] = fl
    bandit_med_or_semgrep = [a or b for a, b in zip(bandit_med, semgrep_flag)]

    rules = [
        ("(0) UNION Bandit(-ll) OR Semgrep [current]", union_flag),
        ("(1) AND  Bandit AND Semgrep (agreement)", and_flag),
        ("(2) Bandit @ MEDIUM confidence (-ll -ii)", bandit_med),
        ("(3) Bandit@MEDIUM OR Semgrep", bandit_med_or_semgrep),
    ]

    results = []
    for label, flags in rules:
        results.append((label, contrast(flags, conds)))

    # sanity gate on rule (0)
    r0 = results[0][1]
    got = {"baseline": round(r0["base_rate"], 1), "industry": round(r0["ind_rate"], 1),
           "drift": round(r0["drift"], 1)}
    sane = all(got[k] == EXPECTED_UNION[k] for k in EXPECTED_UNION)

    lines = []
    lines.append("=" * 92)
    lines.append("AGGREGATION-RULE SENSITIVITY: baseline-vs-industry headline (code-only)")
    lines.append("=" * 92)
    lines.append(f"code-only n = {len(recs)}")
    lines.append(f"SANITY (rule 0 reproduces 14.0/11.4/-2.7): {'PASS' if sane else 'FAIL -> ' + str(got)}")
    lines.append(f"total vulnerable by rule: "
                 f"union={sum(union_flag)}  AND={sum(and_flag)}  "
                 f"banditMED={sum(bandit_med)}  banditMED|semgrep={sum(bandit_med_or_semgrep)}")
    lines.append("")
    hdr = f"{'rule':<44}{'base':>12}{'industry':>12}{'drift':>9}{'p':>9}{'h':>8}"
    lines.append(hdr)
    lines.append("-" * 92)
    for label, r in results:
        lines.append(
            f"{label:<44}"
            f"{r['base_v']}/{r['base_n']} ({r['base_rate']:.1f}%)".rjust(12)
            + f"{r['ind_v']}/{r['ind_n']} ({r['ind_rate']:.1f}%)".rjust(12)
            + f"{r['drift']:>+8.1f}p"
            + f"{r['p']:>9.3f}"
            + f"{r['h']:>+8.3f}"
        )
    lines.append("-" * 92)
    lines.append("")
    for label, r in results:
        lines.append(f"{label}")
        lines.append(f"    baseline {r['base_rate']:.2f}% [{r['base_ci'][0]:.1f},{r['base_ci'][1]:.1f}]  "
                     f"industry {r['ind_rate']:.2f}% [{r['ind_ci'][0]:.1f},{r['ind_ci'][1]:.1f}]  "
                     f"drift {r['drift']:+.2f}pp  Fisher p={r['p']:.4f}  h={r['h']:+.4f}")
    report = "\n".join(lines)
    print(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    TXT.write_text(report + "\n")
    with open(OUT, "w") as fh:
        json.dump({"n": len(recs), "sane": sane,
                   "results": {label: r for label, r in results},
                   "totals": {"union": sum(union_flag), "and": sum(and_flag),
                              "bandit_med": sum(bandit_med),
                              "bandit_med_or_semgrep": sum(bandit_med_or_semgrep)}},
                  fh, indent=2)


if __name__ == "__main__":
    main()
