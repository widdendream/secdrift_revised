#!/usr/bin/env python3
"""Regenerate fig_cwe_distribution for the current SecDrift corpus.

Single-column figure showing, per CWE category, the baseline vs. industry
flagged-vulnerable rate on the code-only corpus (GPT-OSS and empty generations
dropped, matching the paper's headline). It visualizes the paper's central
mechanism: the apparent baseline->industry drift is concentrated in CWE-502
(insecure deserialization) and CWE-22 (path traversal); the other seven
categories are at or near 0% in both conditions.

Writes paper/figures/fig_cwe_distribution.{png,pdf} and prints the per-CWE
numbers used in the caption.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "results" / "latest_run" / "merged_results_5rep.jsonl"
OUTDIR = REPO / "paper" / "figures"

CWE_NAMES = {
    "CWE-502": "Insecure deserialization",
    "CWE-22": "Path traversal",
    "CWE-78": "Command injection",
    "CWE-89": "SQL injection",
    "CWE-327": "Weak cryptography",
    "CWE-79": "Cross-site scripting",
    "CWE-798": "Hardcoded credentials",
    "CWE-330": "Weak randomness",
    "CWE-295": "Certificate validation",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
C_BASE = "#3498db"
C_IND = "#e67e22"


def load_code_only(path: Path):
    recs = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("model") == "gpt-oss-120b":
            continue
        if str(r.get("generated_code") or "").strip() == "":
            continue
        recs.append(r)
    return recs


def per_cwe_rates(recs):
    agg = defaultdict(lambda: {"b_v": 0, "b_n": 0, "i_v": 0, "i_n": 0})
    for r in recs:
        cwe = r.get("cwe")
        pt = r.get("prompt_type")
        v = 1 if r.get("is_vulnerable") else 0
        if pt == "baseline":
            agg[cwe]["b_v"] += v
            agg[cwe]["b_n"] += 1
        elif pt == "industry":
            agg[cwe]["i_v"] += v
            agg[cwe]["i_n"] += 1
    rows = []
    for cwe, d in agg.items():
        br = 100 * d["b_v"] / d["b_n"] if d["b_n"] else 0.0
        ir = 100 * d["i_v"] / d["i_n"] if d["i_n"] else 0.0
        rows.append({
            "cwe": cwe, "name": CWE_NAMES.get(cwe, cwe),
            "b_rate": br, "i_rate": ir, "drift": ir - br,
            "b_n": d["b_n"], "i_n": d["i_n"],
        })
    # sort by baseline rate descending so the two event-bearing CWEs lead
    rows.sort(key=lambda x: (x["b_rate"], x["i_rate"]), reverse=True)
    return rows


def make_figure(rows, outdir: Path):
    labels = [f"{r['name']}\n({r['cwe']})" for r in rows]
    base = [r["b_rate"] for r in rows]
    ind = [r["i_rate"] for r in rows]
    y = np.arange(len(rows))
    h = 0.38

    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    ax.barh(y + h / 2, base, height=h, color=C_BASE, edgecolor="black",
            linewidth=0.4, label="Baseline")
    ax.barh(y - h / 2, ind, height=h, color=C_IND, edgecolor="black",
            linewidth=0.4, label="Industry")

    # annotate drift for the categories that actually move
    for i, r in enumerate(rows):
        if abs(r["drift"]) >= 0.5:
            xpos = max(r["b_rate"], r["i_rate"]) + 2
            ax.text(xpos, y[i], f"{r['drift']:+.1f}pp", va="center",
                    fontsize=7, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Flagged-vulnerable rate (\\%)")
    ax.set_xlim(0, 108)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig_cwe_distribution.png")
    fig.savefig(outdir / "fig_cwe_distribution.pdf")
    plt.close(fig)


def main():
    recs = load_code_only(DATA)
    rows = per_cwe_rates(recs)
    make_figure(rows, OUTDIR)
    print(f"code-only n = {len(recs)}")
    print(f"{'CWE':<10}{'name':<28}{'baseline':>10}{'industry':>10}{'drift':>9}")
    for r in rows:
        print(f"{r['cwe']:<10}{r['name']:<28}"
              f"{r['b_rate']:>9.1f}%{r['i_rate']:>9.1f}%{r['drift']:>+8.1f}p"
              f"   (n_b={r['b_n']}, n_i={r['i_n']})")
    print("wrote paper/figures/fig_cwe_distribution.{png,pdf}")


if __name__ == "__main__":
    main()
