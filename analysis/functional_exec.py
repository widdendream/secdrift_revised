#!/usr/bin/env python3
"""Tier-2 sandboxed functional testing (execution-based pass@1).

Complements the Tier-1 static check (analysis/functional_correctness.py) by
actually EXECUTING generated code against small per-task unit tests, so we can
report a functional pass rate rather than mere well-formedness.

IMPORTANT SCOPE (see analysis notes / paper Section on functional quality):
the 5-dimension transformation changes the callable contract across conditions
(function name, parameter names, arity, and sometimes task semantics), so a
fixed per-task oracle is only valid on the BASELINE condition, where each task
has a single clean contract. We therefore default to --condition baseline and
report the executable-subset pass@1 there. Network tasks (CWE-798, CWE-295) are
excluded because they cannot be validated offline.

Each sample runs in a separate hardened child (analysis/exec_runner.py):
python3 -I -B -E, fresh temp CWD, minimal environment, CPU/memory/file-size
rlimits, network disabled, and a wall-clock timeout enforced here. This
executes UNTRUSTED model output; the sandboxing is defense-in-depth, not a
guarantee. Run on a machine you are comfortable executing generated code on.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import ast
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
DEFAULT_DATA = _REPO / "results" / "latest_run" / "merged_results_5rep.jsonl"
DEFAULT_OUT = _REPO / "results" / "extended_analysis"
RUNNER = _HERE / "exec_runner.py"

EXECUTABLE_CWES = ["CWE-89", "CWE-78", "CWE-502", "CWE-22",
                   "CWE-327", "CWE-79", "CWE-330"]  # excludes CWE-798, CWE-295 (network)
TIMEOUT_S = 12
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n?|\n?```\s*$")


def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def code_only(recs):
    return [r for r in recs
            if r.get("model") != "gpt-oss-120b"
            and str(r.get("generated_code") or "").strip() != ""]


def strip_fences(code: str) -> str:
    t = str(code).strip()
    if t.startswith("```"):
        t = _FENCE.sub("", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t


def parses(code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def run_one(code: str, cwe: str) -> str:
    """Run one sample in a hardened subprocess; return pass/fail/error/timeout."""
    workdir = tempfile.mkdtemp(prefix="secdrift_exec_")
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C", "LANG": "C",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", "-E", str(RUNNER)],
            input=json.dumps({"code": code, "cwe": cwe}),
            text=True, capture_output=True, cwd=workdir, env=env,
            timeout=TIMEOUT_S,
        )
        out = proc.stdout.strip().splitlines()
        if not out:
            return "error"
        verdict = json.loads(out[-1])
        return verdict.get("status", "error")
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "error"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Tier-2 sandboxed functional pass@1")
    ap.add_argument("results", nargs="?", type=Path, default=DEFAULT_DATA)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--condition", default="baseline",
                    help="prompt_type to test (default: baseline; only baseline "
                         "has a clean per-task contract)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max samples per task (0 = all)")
    args = ap.parse_args()

    recs = [r for r in code_only(load(args.results))
            if r.get("prompt_type") == args.condition
            and r.get("cwe") in EXECUTABLE_CWES]

    by_task = defaultdict(list)
    for r in recs:
        by_task[r.get("cwe")].append(r)

    print(f"Condition: {args.condition}   executable tasks: {len(EXECUTABLE_CWES)}"
          f"   (CWE-798, CWE-295 excluded: network)")
    print(f"Samples (code-only): {sum(len(v) for v in by_task.values())}\n")

    header = f"{'CWE':<10}{'n':>5}{'pass':>7}{'fail':>7}{'error':>7}{'timeout':>9}{'pass@1':>9}"
    print(header); print("-" * len(header))

    rows = []
    tot = defaultdict(int)
    for cwe in EXECUTABLE_CWES:
        samples = by_task.get(cwe, [])
        if args.limit:
            samples = samples[:args.limit]
        c = defaultdict(int)
        for r in samples:
            code = strip_fences(str(r.get("generated_code")))
            if not parses(code):
                c["error"] += 1
                continue
            c[run_one(code, cwe)] += 1
        n = sum(c.values())
        p1 = 100.0 * c["pass"] / n if n else float("nan")
        print(f"{cwe:<10}{n:>5}{c['pass']:>7}{c['fail']:>7}{c['error']:>7}"
              f"{c['timeout']:>9}{p1:>8.1f}%")
        rows.append({"cwe": cwe, "n": n, "pass": c["pass"], "fail": c["fail"],
                     "error": c["error"], "timeout": c["timeout"],
                     "pass_at_1": round(p1, 2)})
        for k, v in c.items():
            tot[k] += v

    N = sum(tot.values())
    p1 = 100.0 * tot["pass"] / N if N else float("nan")
    print("-" * len(header))
    print(f"{'ALL':<10}{N:>5}{tot['pass']:>7}{tot['fail']:>7}{tot['error']:>7}"
          f"{tot['timeout']:>9}{p1:>8.1f}%")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"functional_exec_{args.condition}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cwe", "n", "pass", "fail", "error",
                                          "timeout", "pass_at_1"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
