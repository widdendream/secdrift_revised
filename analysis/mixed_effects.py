#!/usr/bin/env python3
"""Mixed-effects logistic regression for sector-conditioned security drift.

Model
-----
    is_vulnerable ~ condition + sector          (fixed effects)
    random intercepts for scenario_id, cwe, model

``condition`` is the prompt type (baseline / matched_baseline / industry).
This asks whether the ``condition`` effect survives once we account for the
clustering induced by which prompt/CWE/model produced each sample.

Estimator
---------
statsmodels' ``MixedLM`` is a *linear* mixed model, so it cannot fit a
logistic outcome. We therefore try, in order:

  1. ``statsmodels.BinomialBayesMixedGLM`` (variational Bayes) -- a true
     mixed-effects *logistic* model with random intercepts.
  2. If that fails / does not produce finite estimates, fall back to a
     ``GEE`` (Binomial family, logit link) with an exchangeable working
     correlation clustered on ``scenario_id``. GEE is a marginal model, not a
     mixed model, so it does not yield random-effect variances; the fallback
     is announced explicitly in the output.

Data filtering
--------------
Uses the project's code-only filter (drop the no-code model gpt-oss-120b and
any record whose generated_code is empty/whitespace) -- the same filter added
to ``secdrift/analysis.py`` -- so the modelled sample matches the paper's
reported rates.

Identification note (baseline has no sector)
--------------------------------------------
``baseline`` prompts are sector-independent (sector is null). In the additive
``condition + sector`` model, baseline is the reference cell: we code baseline
rows at the reference sector level so that all sector dummies are zero for
them. This keeps the design full rank without changing the meaning of the
condition coefficients (baseline-vs-matched and baseline-vs-industry
contrasts). scenario_id and cwe are 1:1 in this dataset, so their two random
intercepts are confounded and should be read as a single scenario/CWE effect.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project package importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from secdrift.analysis import _filter_code_only  # code-only filter (item 1)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DEFAULT_DATA = _REPO_ROOT / "results" / "latest_run" / "merged_results_5rep.jsonl"
DEFAULT_OUT = _REPO_ROOT / "results" / "extended_analysis" / "mixed_effects.json"

# Fixed reference levels for the additive design.
CONDITION_REF = "baseline"
SECTOR_REF = "communications"  # first sector alphabetically; choice is arbitrary
CONDITION_ORDER = ["baseline", "matched_baseline", "industry"]

Z = 1.959963984540054  # 97.5th percentile of the standard normal


def load_frame(data_path: Path) -> pd.DataFrame:
    """Load records, apply the code-only filter, and build the model frame."""
    records = [json.loads(l) for l in open(data_path) if l.strip()]
    records = _filter_code_only(records)

    rows = []
    for r in records:
        sector = r.get("sector")
        # Baseline is sector-less: place it at the reference sector so that
        # all sector dummies are zero for baseline (identification constraint).
        if sector is None or str(sector).strip() == "":
            sector = SECTOR_REF
        rows.append(
            {
                "vuln": int(bool(r.get("is_vulnerable"))),
                "condition": r.get("prompt_type"),
                "sector": sector,
                "scenario_id": r.get("scenario_id"),
                "cwe": r.get("cwe"),
                "model": r.get("model"),
            }
        )
    df = pd.DataFrame(rows)

    # Order categories so the intended reference level is dropped by patsy.
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER)
    sector_levels = [SECTOR_REF] + sorted(s for s in df["sector"].unique() if s != SECTOR_REF)
    df["sector"] = pd.Categorical(df["sector"], categories=sector_levels)
    return df


def _fixed_formula() -> str:
    return (
        "vuln ~ C(condition, Treatment('%s')) + C(sector, Treatment('%s'))"
        % (CONDITION_REF, SECTOR_REF)
    )


def _pretty_name(raw: str) -> str:
    """Shorten patsy coefficient names for readability."""
    return (
        raw.replace("C(condition, Treatment('%s'))" % CONDITION_REF, "condition")
        .replace("C(sector, Treatment('%s'))" % SECTOR_REF, "sector")
        .replace("[T.", "[")
    )


def fit_bayes_mixed(df: pd.DataFrame) -> dict:
    """Fit BinomialBayesMixedGLM with random intercepts for the three groups."""
    vc_formulas = {
        "scenario_id": "0 + C(scenario_id)",
        "cwe": "0 + C(cwe)",
        "model": "0 + C(model)",
    }
    model = BinomialBayesMixedGLM.from_formula(_fixed_formula(), vc_formulas, df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_vb(verbose=False)

    fe_names = list(model.exog_names)
    fe_mean = np.asarray(result.fe_mean, dtype=float)
    fe_sd = np.asarray(result.fe_sd, dtype=float)
    if not (np.all(np.isfinite(fe_mean)) and np.all(np.isfinite(fe_sd))):
        raise RuntimeError("BinomialBayesMixedGLM produced non-finite estimates")

    fixed = []
    for name, m, s in zip(fe_names, fe_mean, fe_sd):
        lo, hi = m - Z * s, m + Z * s
        fixed.append(
            {
                "term": _pretty_name(name),
                "coef": round(float(m), 4),
                "se": round(float(s), 4),
                "ci_low": round(float(lo), 4),
                "ci_high": round(float(hi), 4),
                "odds_ratio": round(float(np.exp(m)), 4),
                "or_ci_low": round(float(np.exp(lo)), 4),
                "or_ci_high": round(float(np.exp(hi)), 4),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        )

    # Variance components: vcp are on the log-SD scale (sd = exp(vcp)).
    vcp_names = list(getattr(model, "vcp_names", []))
    vcp_mean = np.asarray(result.vcp_mean, dtype=float)
    if len(vcp_names) != len(vcp_mean):
        vcp_names = [f"vc{i}" for i in range(len(vcp_mean))]
    random = []
    for name, logsd in zip(vcp_names, vcp_mean):
        sd = float(np.exp(logsd))
        random.append(
            {
                "group": name,
                "sd": round(sd, 4),
                "variance": round(sd ** 2, 4),
            }
        )

    return {"method": "BinomialBayesMixedGLM (variational Bayes)",
            "fixed": fixed, "random": random, "fell_back": False}


def fit_gee_fallback(df: pd.DataFrame, reason: str) -> dict:
    """GEE with exchangeable working correlation clustered on scenario_id."""
    model = smf.gee(
        _fixed_formula(),
        groups="scenario_id",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit()

    params = result.params
    conf = result.conf_int()
    fixed = []
    for name in params.index:
        m = float(params[name])
        lo, hi = float(conf.loc[name, 0]), float(conf.loc[name, 1])
        fixed.append(
            {
                "term": _pretty_name(name),
                "coef": round(m, 4),
                "se": round(float(result.bse[name]), 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "odds_ratio": round(float(np.exp(m)), 4),
                "or_ci_low": round(float(np.exp(lo)), 4),
                "or_ci_high": round(float(np.exp(hi)), 4),
                "p_value": round(float(result.pvalues[name]), 4),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        )

    alpha = None
    try:
        alpha = round(float(model.cov_struct.dep_params), 4)
    except Exception:
        pass

    return {
        "method": "GEE (Binomial, logit; exchangeable, clustered on scenario_id)",
        "fixed": fixed,
        "random": [],
        "fell_back": True,
        "fallback_reason": reason,
        "working_correlation_alpha": alpha,
    }


def _condition_verdict(result: dict) -> str:
    """Plain-language statement about the condition effect."""
    cond_terms = [f for f in result["fixed"] if f["term"].startswith("condition")]
    if not cond_terms:
        return "No condition terms were estimated."

    parts = []
    any_sig = False
    for t in cond_terms:
        level = t["term"].split("[", 1)[-1].rstrip("]").lstrip("T.")
        sig = t["excludes_zero"]
        any_sig = any_sig or sig
        ci = f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]"
        parts.append(
            f"{level} vs baseline: log-odds {t['coef']:+.3f} (OR {t['odds_ratio']:.2f}), "
            f"95% CI {ci} -> {'significant' if sig else 'NOT significant'}"
        )

    lead = (
        "After accounting for scenario/CWE/model clustering, the condition effect "
        + ("REMAINS statistically significant" if any_sig
           else "is NOT statistically significant")
        + " (industry vs baseline in particular)."
    )
    return lead + "\n    - " + "\n    - ".join(parts)


def format_report(result: dict, n_rows: int) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append("Mixed-effects logistic regression: is_vulnerable ~ condition + sector")
    lines.append("Random intercepts: scenario_id, cwe, model  |  code-only filter")
    lines.append("=" * 74)
    lines.append(f"Observations: {n_rows}")
    lines.append(f"Estimator   : {result['method']}")
    if result.get("fell_back"):
        lines.append(f"FALLBACK    : {result.get('fallback_reason')}")
        if result.get("working_correlation_alpha") is not None:
            lines.append(f"Exchangeable working correlation alpha = "
                         f"{result['working_correlation_alpha']}")
    lines.append("")

    lines.append("FIXED EFFECTS (log-odds; OR = odds ratio)")
    lines.append("-" * 74)
    header = f"{'term':<22}{'coef':>9}{'95% CI':>20}{'OR':>8}{'sig':>6}"
    lines.append(header)
    for f in result["fixed"]:
        ci = f"[{f['ci_low']:+.3f},{f['ci_high']:+.3f}]"
        sig = "*" if f["excludes_zero"] else ""
        lines.append(f"{f['term']:<22}{f['coef']:>9.3f}{ci:>20}{f['odds_ratio']:>8.2f}{sig:>6}")
    lines.append("")

    if result["random"]:
        lines.append("RANDOM-EFFECT VARIANCES (intercept variance per grouping)")
        lines.append("-" * 74)
        lines.append(f"{'group':<16}{'sd':>10}{'variance':>12}")
        for r in result["random"]:
            lines.append(f"{r['group']:<16}{r['sd']:>10.3f}{r['variance']:>12.3f}")
        lines.append("")
        lines.append("Note: scenario_id and cwe are 1:1 in this dataset; their two")
        lines.append("variance components are confounded (read as one scenario/CWE effect).")
        lines.append("")
    else:
        lines.append("RANDOM-EFFECT VARIANCES: not available under the GEE fallback")
        lines.append("(GEE is a marginal model; clustering is handled via the working")
        lines.append("correlation rather than estimated intercept variances).")
        lines.append("")

    lines.append("VERDICT")
    lines.append("-" * 74)
    lines.append(_condition_verdict(result))
    lines.append("=" * 74)
    return "\n".join(lines)


def run(data_path: Path = DEFAULT_DATA, out_path: Path = DEFAULT_OUT) -> dict:
    df = load_frame(Path(data_path))

    try:
        result = fit_bayes_mixed(df)
    except Exception as exc:  # noqa: BLE001 - report any failure and fall back
        result = fit_gee_fallback(
            df, reason=f"BinomialBayesMixedGLM failed to converge: {exc}"
        )

    report = format_report(result, len(df))
    print(report)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"n_observations": len(df), **result}, fh, indent=2)
    return result


if __name__ == "__main__":
    _data = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    _out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    run(_data, _out)
