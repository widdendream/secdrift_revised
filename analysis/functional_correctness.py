#!/usr/bin/env python3
"""Tier-1 static functional-quality check for SecDrift.

A static-analysis flag rate says nothing about whether the generated code is
*useful*: a reviewer noted that "safer-looking" code could simply be less
functional. This script measures a functional-quality signal WITHOUT executing
any code, so it is safe to run on the full corpus.

Design note (important): the 5-dimension transformation deliberately rewrites
the prompt, and in doing so it renames the requested function and even some
parameters across conditions/sectors (e.g. ``search_records`` ->
``lookup_subscriber_records`` -> ``lookup_subscribers``, with ``status`` ->
``account_status``). An exact function-name / fixed-arity contract would
therefore penalize the industry condition for the transformation's own
renaming rather than for any loss of functional quality. We consequently use
transformation-invariant structural criteria that do not depend on the
requested identifier:

  parses        : ast.parse succeeds after stripping Markdown fences
  defines_fn    : at least one function (any name) is defined
  nonstub_fn    : at least one defined function has a substantive body
                  (more than pass / a docstring / an Ellipsis)
  well_formed   : parses AND defines a non-stub function  (headline metric)

We report these per condition (and per model) alongside the vulnerability rate,
on the same code-only denominator, so functional quality and security can be
read together. If well-formedness is uniformly high across conditions, then a
lower industry flag rate is not an artifact of the industry condition emitting
less or emptier code.

Reused from completeness.py: the fence-stripping + ast.parse parseability
definition, so ``parses`` here matches the paper's completeness table.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = _REPO_ROOT / "results" / "latest_run" / "merged_results_5rep.jsonl"
DEFAULT_OUT = _REPO_ROOT / "results" / "extended_analysis"

CONDITION_ORDER = ["baseline", "matched_baseline", "industry"]
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n?|\n?```\s*$")


def load(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def code_only(recs):
    out = []
    for r in recs:
        if r.get("model") == "gpt-oss-120b":
            continue
        if str(r.get("generated_code") or "").strip() == "":
            continue
        out.append(r)
    return out


def strip_fences(code: str) -> str:
    text = str(code).strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text


def parse_code(code):
    if code is None or str(code).strip() == "":
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(strip_fences(str(code)))
    except (SyntaxError, ValueError):
        return None


def _is_docstring(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _is_ellipsis(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis)


def is_substantive(func):
    """A function is a non-stub if, after dropping a leading docstring, its body
    contains something other than a single pass / Ellipsis."""
    body = list(func.body)
    if body and _is_docstring(body[0]):
        body = body[1:]
    if not body:
        return False
    if len(body) == 1 and (isinstance(body[0], ast.Pass)
                           or _is_ellipsis(body[0])):
        return False
    return True


def classify(rec):
    """Return (parses, defines_fn, nonstub) booleans for one record."""
    tree = parse_code(rec.get("generated_code"))
    if tree is None:
        return (False, False, False)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return (True, False, False)
    return (True, True, any(is_substantive(f) for f in funcs))


def aggregate(recs, keyfn):
    agg = defaultdict(lambda: {"n": 0, "parses": 0, "defines": 0,
                               "nonstub": 0, "vuln": 0})
    for r in recs:
        p, d, ns = classify(r)
        a = agg[keyfn(r)]
        a["n"] += 1
        a["parses"] += int(p)
        a["defines"] += int(d)
        a["nonstub"] += int(ns)          # nonstub implies parses & defines
        a["vuln"] += int(bool(r.get("is_vulnerable")))
    return agg


def pct(x, n):
    return 100.0 * x / n if n else float("nan")


HEADER = (f"{'group':<18}{'n':>6}{'parses':>9}{'defines_fn':>12}"
          f"{'well_formed':>13}{'vuln':>9}")


def emit(label, a, rows, group):
    print(f"{label:<18}{a['n']:>6}{pct(a['parses'],a['n']):>8.1f}%"
          f"{pct(a['defines'],a['n']):>11.1f}%{pct(a['nonstub'],a['n']):>12.1f}%"
          f"{pct(a['vuln'],a['n']):>8.1f}%")
    rows.append({
        "group": group, "key": label, "n": a["n"],
        "parses_rate": round(pct(a["parses"], a["n"]), 2),
        "defines_fn_rate": round(pct(a["defines"], a["n"]), 2),
        "well_formed_rate": round(pct(a["nonstub"], a["n"]), 2),
        "vuln_rate": round(pct(a["vuln"], a["n"]), 2),
    })


def main():
    ap = argparse.ArgumentParser(description="Tier-1 static functional check")
    ap.add_argument("results", nargs="?", type=Path, default=DEFAULT_DATA)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    recs = code_only(load(args.results))
    print(f"Code-only responses (excl. gpt-oss + empty): {len(recs)}\n")
    rows = []

    by_cond = aggregate(recs, lambda r: r.get("prompt_type"))
    print("Functional-quality rates by condition (code-only):")
    print(HEADER); print("-" * len(HEADER))
    tot = {"n": 0, "parses": 0, "defines": 0, "nonstub": 0, "vuln": 0}
    for cond in CONDITION_ORDER:
        a = by_cond.get(cond)
        if not a:
            continue
        emit(cond, a, rows, "condition")
        for k in tot:
            tot[k] += a[k]
    emit("ALL", tot, rows, "overall")

    by_model = aggregate(recs, lambda r: r.get("model"))
    print("\nFunctional-quality rates by model (code-only):")
    print(HEADER); print("-" * len(HEADER))
    for m in sorted(by_model):
        emit(m, by_model[m], rows, "model")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "functional_correctness.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "key", "n", "parses_rate",
                                          "defines_fn_rate", "well_formed_rate",
                                          "vuln_rate"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
