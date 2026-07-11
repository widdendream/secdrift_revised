#!/usr/bin/env python3
"""Response-completeness analysis for SecDrift.

Runs on the FULL dataset -- no code-only filter, all 7 models including
gpt-oss-120b -- and characterizes, per model and per condition (prompt type):

  * n                     : number of responses
  * empty_rate            : fraction with empty/whitespace generated_code
  * mean_lines            : mean generated_code length in lines (empty = 0)
  * median_lines          : median generated_code length in lines
  * parseable_rate        : fraction whose code parses as Python (ast.parse)
  * non_parseable_rate    : fraction that is non-empty but does not parse

Outputs ``completeness.csv`` and a stacked-bar figure (``completeness.png`` /
``.pdf``) styled to match ``secdrift/visualize.py``.

The parseability criterion is ``ast.parse`` success: a response "contains a
parseable Python block" if, after stripping any Markdown code fences, the
extracted text parses without a SyntaxError. Empty responses are, by
definition, neither parseable nor counted toward code length beyond 0 lines.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Publication style (matches secdrift/visualize.py)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = _REPO_ROOT / "results" / "latest_run" / "merged_results_5rep.jsonl"
DEFAULT_OUT = _REPO_ROOT / "results" / "extended_analysis"

CONDITION_ORDER = ["baseline", "matched_baseline", "industry"]
CONDITION_LABEL = {"baseline": "B", "matched_baseline": "M", "industry": "I"}

# Colors consistent with visualize.py; empty flagged in red.
COLOR_PARSEABLE = "#2ecc71"      # green
COLOR_NONPARSE = "#f39c12"       # orange
COLOR_EMPTY = "#e74c3c"          # red

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n?|\n?```\s*$")


def load_results(path: Path) -> List[Dict]:
    """Load benchmark results from a JSONL file (full dataset, no filtering)."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def is_empty(code) -> bool:
    """True if the response produced no code (None / empty / whitespace)."""
    return code is None or str(code).strip() == ""


def n_lines(code) -> int:
    """Number of lines in the generated code; empty responses count as 0."""
    if is_empty(code):
        return 0
    return len(str(code).splitlines())


def _strip_fences(code: str) -> str:
    """Remove a single wrapping Markdown code fence if present."""
    text = str(code).strip()
    if text.startswith("```"):
        # Drop opening fence line and a trailing fence if present.
        text = _FENCE_RE.sub("", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text


def is_parseable(code) -> bool:
    """True if the (fence-stripped) response parses as Python via ast.parse."""
    if is_empty(code):
        return False
    candidate = _strip_fences(str(code))
    try:
        # Suppress SyntaxWarnings (e.g. invalid escape sequences) emitted while
        # parsing model-generated code; they do not affect parse success.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(candidate)
        return True
    except (SyntaxError, ValueError):
        return False


def compute_completeness(results: List[Dict]) -> List[Dict]:
    """Aggregate completeness metrics per (model, condition)."""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in results:
        groups[(r.get("model", "unknown"), r.get("prompt_type", "unknown"))].append(r)

    rows = []
    models = sorted({m for m, _ in groups})
    for model in models:
        for condition in CONDITION_ORDER:
            recs = groups.get((model, condition))
            if not recs:
                continue
            n = len(recs)
            codes = [r.get("generated_code") for r in recs]
            n_empty = sum(1 for c in codes if is_empty(c))
            n_parse = sum(1 for c in codes if is_parseable(c))
            lines = [n_lines(c) for c in codes]
            rows.append({
                "model": model,
                "condition": condition,
                "n": n,
                "empty_rate": round(n_empty / n, 4),
                "mean_lines": round(statistics.mean(lines), 2),
                "median_lines": round(statistics.median(lines), 1),
                "parseable_rate": round(n_parse / n, 4),
                "non_parseable_rate": round((n - n_empty - n_parse) / n, 4),
            })
    return rows


def write_csv(rows: List[Dict], out_path: Path) -> None:
    cols = ["model", "condition", "n", "empty_rate", "mean_lines",
            "median_lines", "parseable_rate", "non_parseable_rate"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_stacked(rows: List[Dict], out_dir: Path) -> None:
    """Grouped stacked bars: per model, one stack per condition, composed of
    parseable / non-parseable / empty shares (matches visualize.py style)."""
    models = sorted({r["model"] for r in rows})
    by_key = {(r["model"], r["condition"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    width = 0.26
    offsets = {"baseline": -width, "matched_baseline": 0.0, "industry": width}

    for condition in CONDITION_ORDER:
        pos = x + offsets[condition]
        parse = np.array([100 * by_key[(m, condition)]["parseable_rate"]
                          if (m, condition) in by_key else 0 for m in models])
        nonp = np.array([100 * by_key[(m, condition)]["non_parseable_rate"]
                         if (m, condition) in by_key else 0 for m in models])
        empty = np.array([100 * by_key[(m, condition)]["empty_rate"]
                          if (m, condition) in by_key else 0 for m in models])

        ax.bar(pos, parse, width, color=COLOR_PARSEABLE,
               edgecolor="black", linewidth=0.5)
        ax.bar(pos, nonp, width, bottom=parse, color=COLOR_NONPARSE,
               edgecolor="black", linewidth=0.5)
        ax.bar(pos, empty, width, bottom=parse + nonp, color=COLOR_EMPTY,
               edgecolor="black", linewidth=0.5)

        # Condition marker (B / M / I) beneath each sub-bar.
        for xi in pos:
            ax.text(xi, -4, CONDITION_LABEL[condition], ha="center", va="top",
                    fontsize=7, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Share of Responses (%)")
    ax.set_xlabel("Model  (B = baseline, M = matched baseline, I = industry)")
    ax.set_title("Response Completeness by Model and Condition\n"
                 "(full dataset, all 7 models)")
    ax.set_ylim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_PARSEABLE, edgecolor="black", label="Parseable Python"),
        mpatches.Patch(facecolor=COLOR_NONPARSE, edgecolor="black", label="Non-parseable content"),
        mpatches.Patch(facecolor=COLOR_EMPTY, edgecolor="black", label="Empty response"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "completeness.pdf")
    plt.savefig(out_dir / "completeness.png")
    plt.close()
    print("\u2713 Figure: completeness stacked bars")


def run(data_path: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT) -> List[Dict]:
    results = load_results(Path(data_path))
    print(f"Loaded {len(results)} evaluations (full dataset, no filtering)")

    rows = compute_completeness(results)
    csv_path = Path(out_dir) / "completeness.csv"
    write_csv(rows, csv_path)
    plot_stacked(rows, Path(out_dir))

    # Console summary.
    print(f"\nWrote {csv_path}")
    print(f"{'model':<18}{'cond':<18}{'n':>5}{'empty':>8}{'meanL':>8}{'parse':>8}")
    for r in rows:
        print(f"{r['model']:<18}{r['condition']:<18}{r['n']:>5}"
              f"{r['empty_rate']:>8.2f}{r['mean_lines']:>8.1f}{r['parseable_rate']:>8.2f}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecDrift response-completeness analysis")
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_DATA,
                        help="Path to full results JSONL")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                        help="Output directory for completeness.csv and figure")
    args = parser.parse_args()
    run(args.results, args.output)
