#!/usr/bin/env python3
"""Build a human-validation sample for the zero-detection CWE categories.

Purpose (human validation)
--------------------------
Six of the nine CWE categories in the benchmark show a 0% vulnerability rate
under the automated SAST pipeline (Bandit + Semgrep). A 0% rate has two very
different possible explanations, and only a human can tell them apart:

  * the models genuinely wrote secure code, OR
  * the detector missed real vulnerabilities, OR
  * the prompt never actually asked for code that could exhibit that CWE.

This script draws a stratified random sample (15 records per zero-detection
CWE, 90 total) so that two human reviewers can independently label each
generated snippet. Both reviewers label all 90 records, and each record carries
a ``verdict_1`` (reviewer 1), a ``verdict_2`` (reviewer 2), and a reconciled
``consensus`` verdict. Every verdict is one of:

  * ``secure``                          -- code is genuinely safe; the 0% is real.
  * ``vulnerable_missed_by_SAST``       -- a real vulnerability the tools missed.
  * ``not_applicable_task_mismatch``    -- the prompt did not elicit code that
                                           could exhibit this CWE, so the 0% is a
                                           prompt-design artifact, not a detector
                                           result.

How the labels are used
-----------------------
  * Inter-rater reliability: both reviewers label all 90 records (full overlap),
    and we compute Cohen's kappa between ``verdict_1`` and ``verdict_2``.
  * Reconciliation: disagreements are resolved by discussion into ``consensus``;
    the deciding rationale is recorded in ``notes`` and in
    ``validation_reconciliation_notes.md``.
  * ``vulnerable_missed_by_SAST`` counts are reported honestly: they lower the
    measured precision of the SAST pipeline but strengthen the paper's
    credibility by bounding its false-negative rate.
  * ``not_applicable_task_mismatch`` counts indicate the prompt design (not the
    detector) produced the zero; those CWEs get redesigned prompts in the
    benchmark expansion.

Sampling method
---------------
Code-only filter (drop gpt-oss-120b and empty/whitespace generated_code, the
same filter used elsewhere). Within each zero-detection CWE, records are
grouped into (model, condition) strata and drawn round-robin across those
strata so the 15 picks are spread across models and conditions rather than
concentrated. Fully reproducible with ``seed=42``.

Output
------
``validation_sample.csv`` with columns:
    sample_id, cwe, model, condition, prompt_text, generated_code,
    verdict_1, verdict_2, consensus, notes
The last four (verdict_1, verdict_2, consensus, notes) are intentionally left
blank for the human reviewers to fill in and reconcile.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Make the project package importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from secdrift.analysis import _filter_code_only  # shared code-only filter

DEFAULT_DATA = _REPO_ROOT / "results" / "latest_run" / "merged_results_5rep.jsonl"
DEFAULT_OUT = _REPO_ROOT / "results" / "extended_analysis" / "validation_sample.csv"

SEED = 42
PER_CWE = 15

CSV_COLUMNS = [
    "sample_id",
    "cwe",
    "model",
    "condition",
    "prompt_text",
    "generated_code",
    "verdict_1",   # reviewer 1: secure / vulnerable_missed_by_SAST / not_applicable_task_mismatch
    "verdict_2",   # reviewer 2: same label set
    "consensus",   # reconciled verdict (== reviewers where they agree)
    "notes",       # reconciliation rationale on disagreed rows
]

# Valid verdicts (for reviewer reference / downstream validation).
VERDICTS = ("secure", "vulnerable_missed_by_SAST", "not_applicable_task_mismatch")


def load_records(data_path: Path) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in open(data_path) if line.strip()]
    if not records:
        raise ValueError(f"No results found in {data_path}")
    return records


def zero_detection_cwes(records: List[Dict[str, Any]]) -> List[str]:
    """CWE categories with zero detections in the (code-only) records."""
    total = defaultdict(int)
    vuln = defaultdict(int)
    for r in records:
        cwe = r.get("cwe")
        total[cwe] += 1
        if r.get("is_vulnerable"):
            vuln[cwe] += 1
    return sorted(cwe for cwe in total if vuln[cwe] == 0)


def stratified_sample_for_cwe(
    records: List[Dict[str, Any]],
    n_target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Round-robin sample of ``n_target`` records spread across (model,
    condition) strata. Deterministic given ``rng``."""
    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        strata[(r.get("model"), r.get("prompt_type"))].append(r)

    # Deterministic order, then seeded shuffle within and across strata.
    strata_keys = sorted(strata.keys())
    for key in strata_keys:
        strata[key].sort(key=lambda r: str(r.get("evaluation_id", "")))
        rng.shuffle(strata[key])
    rng.shuffle(strata_keys)

    selected: List[Dict[str, Any]] = []
    while len(selected) < n_target and any(strata[k] for k in strata_keys):
        for key in strata_keys:
            if strata[key]:
                selected.append(strata[key].pop())
                if len(selected) >= n_target:
                    break
    return selected


def build_sample(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = _filter_code_only(records)
    zero_cwes = zero_detection_cwes(filtered)

    if len(zero_cwes) != 6:
        print(f"WARNING: expected 6 zero-detection CWEs, found {len(zero_cwes)}: "
              f"{zero_cwes}", file=sys.stderr)

    rng = random.Random(SEED)
    rows: List[Dict[str, Any]] = []
    counter = 0

    for cwe in zero_cwes:
        cwe_records = [r for r in filtered if r.get("cwe") == cwe]
        picks = stratified_sample_for_cwe(cwe_records, PER_CWE, rng)
        if len(picks) < PER_CWE:
            print(f"WARNING: only {len(picks)} code-only records available for "
                  f"{cwe} (wanted {PER_CWE}).", file=sys.stderr)
        for r in picks:
            counter += 1
            rows.append({
                "sample_id": f"VAL-{counter:03d}",
                "cwe": cwe,
                "model": r.get("model"),
                "condition": r.get("prompt_type"),
                "prompt_text": r.get("prompt_text", ""),
                "generated_code": r.get("generated_code", ""),
                "verdict_1": "",
                "verdict_2": "",
                "consensus": "",
                "notes": "",
            })
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # QUOTE_ALL keeps multi-line code/prompt cells intact and unambiguous.
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    by_cwe = defaultdict(int)
    by_model = defaultdict(int)
    by_condition = defaultdict(int)
    for row in rows:
        by_cwe[row["cwe"]] += 1
        by_model[row["model"]] += 1
        by_condition[row["condition"]] += 1

    print(f"Total sampled: {len(rows)}")
    print("\nPer CWE:")
    for cwe in sorted(by_cwe):
        print(f"  {cwe:<10} {by_cwe[cwe]}")
    print("\nPer condition:")
    for cond in sorted(by_condition):
        print(f"  {cond:<18} {by_condition[cond]}")
    print("\nPer model:")
    for model in sorted(by_model):
        print(f"  {model:<18} {by_model[model]}")
    print("\nVerdict options for reviewers: " + " | ".join(VERDICTS))


def run(data_path: Path = DEFAULT_DATA, out_path: Path = DEFAULT_OUT) -> List[Dict[str, Any]]:
    records = load_records(Path(data_path))
    rows = build_sample(records)
    write_csv(rows, Path(out_path))
    print(f"Wrote {out_path}")
    print_summary(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a stratified human-validation sample for zero-detection CWEs.")
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_DATA,
                        help="Path to full results JSONL")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                        help="Output path for validation_sample.csv")
    args = parser.parse_args()
    run(args.results, args.output)
