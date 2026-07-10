#!/usr/bin/env python3
"""Separation-robust checks on the condition effect (RQ1 robustness).

The mixed-effects model (Section 6.7) reports a significant *conditional*
industry effect, but six of nine CWE scenarios have zero events and one is
near 100%, i.e. quasi-complete separation. Ordinary ML logistic regression
with CWE fixed effects is therefore unstable. We re-estimate the condition
effect two separation-robust ways:

  1. Firth bias-reduced (penalized-likelihood) logistic regression:
       is_vulnerable ~ industry + C(cwe)      (baseline vs industry rows)
     Firth's Jeffreys-prior penalty yields finite estimates under separation.
     We report the industry log-odds/OR with a Wald 95% CI and a penalized
     likelihood-ratio test (PLRT) of the industry coefficient, and contrast
     with the ordinary-ML fit (which is degenerate under separation).

  2. Permutation tests on the condition label (model-free):
       (a) marginal: permute baseline/industry over the pooled sample,
           statistic = drift (industry-rate - baseline-rate);
       (b) CWE-stratified: permute within each CWE (Cochran-Mantel-Haenszel
           numerator statistic), so zero-event strata contribute nothing and
           the separation cannot destabilize the test.

Code-only filter, GPT-OSS excluded, to match the paper's reported rates.
Seed = 42.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "results" / "latest_run" / "merged_results_5rep.jsonl"
OUT = REPO / "results" / "extended_analysis" / "firth_permutation.json"
TXT = REPO / "results" / "extended_analysis" / "firth_permutation.txt"
SEED = 42
N_PERM = 50000


def load():
    records = [json.loads(l) for l in open(DATA) if l.strip()]
    recs = [r for r in records
            if r.get("model") != "gpt-oss-120b"
            and str(r.get("generated_code") or "").strip() != ""]
    return recs


# --------------------------------------------------------------------------- #
# Firth logistic regression
# --------------------------------------------------------------------------- #
def firth_fit(X, y, max_iter=200, tol=1e-8):
    """Firth penalized-likelihood logistic regression.

    Returns dict with beta, cov (inverse info), penalized loglik, n_iter.
    """
    n, p = X.shape
    beta = np.zeros(p)
    for it in range(max_iter):
        eta = X @ beta
        pr = 1.0 / (1.0 + np.exp(-eta))
        pr = np.clip(pr, 1e-10, 1 - 1e-10)
        w = pr * (1 - pr)
        # Fisher information
        XW = X * w[:, None]
        info = X.T @ XW
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            info_inv = np.linalg.pinv(info)
        # hat diagonal: h_i = w_i * x_i^T info_inv x_i
        # compute via (X info_inv) elementwise X, times w
        XII = X @ info_inv
        h = w * np.einsum("ij,ij->i", XII, X)
        # penalized score
        U = X.T @ (y - pr + h * (0.5 - pr))
        step = info_inv @ U
        # damping for stability
        mstep = np.max(np.abs(step))
        if mstep > 5:
            step *= 5.0 / mstep
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # final quantities
    eta = X @ beta
    pr = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
    w = pr * (1 - pr)
    info = X.T @ (X * w[:, None])
    sign, logdet = np.linalg.slogdet(info)
    ll = np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    pen_ll = ll + 0.5 * logdet
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    return {"beta": beta, "cov": cov, "pen_ll": float(pen_ll), "n_iter": it + 1}


def build_design(recs, cwe_ref):
    """baseline+industry rows -> (X, y, colnames). Columns: intercept, industry,
    then CWE dummies (ref dropped)."""
    rows = [r for r in recs if r.get("prompt_type") in ("baseline", "industry")]
    cwes = sorted(set(r.get("cwe") for r in rows))
    cwes = [c for c in cwes if c != cwe_ref]
    cols = ["intercept", "industry"] + [f"cwe[{c}]" for c in cwes]
    X = np.zeros((len(rows), len(cols)))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        X[i, 0] = 1.0
        X[i, 1] = 1.0 if r.get("prompt_type") == "industry" else 0.0
        c = r.get("cwe")
        if c in cwes:
            X[i, 2 + cwes.index(c)] = 1.0
        y[i] = 1.0 if r.get("is_vulnerable") else 0.0
    return X, y, cols


def firth_industry(recs):
    """Firth on baseline vs industry with CWE fixed effects + PLRT on industry."""
    # choose a CWE reference that HAS events so the intercept is finite-ish
    X, y, cols = build_design(recs, cwe_ref="CWE-502")
    full = firth_fit(X, y)
    j = cols.index("industry")
    beta_ind = full["beta"][j]
    se_ind = float(np.sqrt(full["cov"][j, j]))
    ci = (beta_ind - 1.959963985 * se_ind, beta_ind + 1.959963985 * se_ind)

    # PLRT: drop the industry column, refit, compare penalized loglik
    keep = [k for k in range(X.shape[1]) if k != j]
    red = firth_fit(X[:, keep], y)
    lr = 2.0 * (full["pen_ll"] - red["pen_ll"])
    p_plrt = float(chi2.sf(lr, df=1))

    # ordinary ML for contrast (expected to be unstable under separation)
    ml_note = ""
    try:
        import statsmodels.api as sm
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(disp=0, maxiter=100)
        ml_beta = res.params[j]
        ml_se = res.bse[j]
        ml_note = f"ML industry beta={ml_beta:.3f} se={ml_se:.3f} (unstable if huge)"
    except Exception as e:
        ml_note = f"ML did not converge under separation: {type(e).__name__}: {e}"

    return {
        "n_rows": int(X.shape[0]),
        "cwe_ref": "CWE-502",
        "industry_logodds": float(beta_ind),
        "industry_se": se_ind,
        "industry_OR": float(np.exp(beta_ind)),
        "industry_OR_ci": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
        "industry_ci_logodds": [float(ci[0]), float(ci[1])],
        "plrt_stat": float(lr),
        "plrt_p": p_plrt,
        "ml_note": ml_note,
    }


# --------------------------------------------------------------------------- #
# Permutation tests
# --------------------------------------------------------------------------- #
def perm_marginal(recs, rng):
    rows = [r for r in recs if r.get("prompt_type") in ("baseline", "industry")]
    y = np.array([1.0 if r.get("is_vulnerable") else 0.0 for r in rows])
    is_ind = np.array([1 if r.get("prompt_type") == "industry" else 0 for r in rows])
    n_ind = int(is_ind.sum())
    n_base = len(rows) - n_ind

    def drift(mask_ind):
        iv = y[mask_ind == 1].mean()
        bv = y[mask_ind == 0].mean()
        return iv - bv

    obs = drift(is_ind)
    idx = np.arange(len(rows))
    count = 0
    for _ in range(N_PERM):
        perm = rng.permutation(idx)
        mask = np.zeros(len(rows), dtype=int)
        mask[perm[:n_ind]] = 1
        if abs(drift(mask)) >= abs(obs) - 1e-12:
            count += 1
    p = (count + 1) / (N_PERM + 1)
    return {"obs_drift_pp": float(100 * obs), "p_two_sided": float(p),
            "n_ind": n_ind, "n_base": n_base}


def perm_stratified(recs, rng):
    """CMH-numerator permutation, permuting condition within each CWE stratum."""
    rows = [r for r in recs if r.get("prompt_type") in ("baseline", "industry")]
    cwes = sorted(set(r.get("cwe") for r in rows))
    strata = []  # (y_vec, n_ind) per stratum
    for c in cwes:
        sub = [r for r in rows if r.get("cwe") == c]
        y = np.array([1.0 if r.get("is_vulnerable") else 0.0 for r in sub])
        n_ind = sum(1 for r in sub if r.get("prompt_type") == "industry")
        strata.append((y, n_ind))

    def stat(assign_list):
        # assign_list[k] = boolean mask (industry) for stratum k
        s = 0.0
        for (y, n_ind), mask in zip(strata, assign_list):
            n = len(y)
            tot_v = y.sum()
            a = y[mask].sum()             # industry vulnerable
            expected = n_ind * tot_v / n if n else 0.0
            s += (a - expected)
        return s

    # observed
    obs_masks = []
    for c, (y, n_ind) in zip(cwes, strata):
        sub = [r for r in rows if r.get("cwe") == c]
        mask = np.array([r.get("prompt_type") == "industry" for r in sub])
        obs_masks.append(mask)
    obs = stat(obs_masks)

    count = 0
    for _ in range(N_PERM):
        perm_masks = []
        for (y, n_ind) in strata:
            n = len(y)
            m = np.zeros(n, dtype=bool)
            if n_ind > 0 and n_ind < n:
                sel = rng.choice(n, size=n_ind, replace=False)
                m[sel] = True
            elif n_ind == n:
                m[:] = True
            perm_masks.append(m)
        if abs(stat(perm_masks)) >= abs(obs) - 1e-9:
            count += 1
    p = (count + 1) / (N_PERM + 1)
    return {"obs_statistic": float(obs), "p_two_sided": float(p),
            "n_strata": len(cwes),
            "event_bearing_strata": int(sum(1 for (y, _) in strata if y.sum() > 0))}


def main():
    recs = load()
    rng = np.random.default_rng(SEED)
    print(f"code-only n = {len(recs)}")

    firth = firth_industry(recs)
    pm = perm_marginal(recs, rng)
    ps = perm_stratified(recs, rng)

    lines = []
    lines.append("=" * 80)
    lines.append("SEPARATION-ROBUST CHECKS ON THE CONDITION EFFECT (baseline vs industry)")
    lines.append("=" * 80)
    lines.append(f"code-only n = {len(recs)};  permutations = {N_PERM};  seed = {SEED}")
    lines.append("")
    lines.append("1) FIRTH bias-reduced logistic: is_vulnerable ~ industry + C(cwe)")
    lines.append(f"   rows (baseline+industry)   : {firth['n_rows']}  (CWE ref = {firth['cwe_ref']})")
    lines.append(f"   industry log-odds          : {firth['industry_logodds']:+.3f} "
                 f"(SE {firth['industry_se']:.3f})")
    lines.append(f"   industry OR [95% Wald CI]  : {firth['industry_OR']:.3f} "
                 f"[{firth['industry_OR_ci'][0]:.3f}, {firth['industry_OR_ci'][1]:.3f}]")
    lines.append(f"   penalized LR test industry : chi2(1)={firth['plrt_stat']:.3f}, "
                 f"p={firth['plrt_p']:.4f}")
    lines.append(f"   ordinary ML contrast       : {firth['ml_note']}")
    lines.append("")
    lines.append("2) PERMUTATION TEST -- marginal (pooled baseline vs industry)")
    lines.append(f"   observed drift             : {pm['obs_drift_pp']:+.2f} pp "
                 f"(n_base={pm['n_base']}, n_ind={pm['n_ind']})")
    lines.append(f"   two-sided permutation p    : {pm['p_two_sided']:.4f}")
    lines.append("")
    lines.append("3) PERMUTATION TEST -- CWE-stratified (CMH numerator; within-stratum)")
    lines.append(f"   strata                     : {ps['n_strata']} "
                 f"({ps['event_bearing_strata']} event-bearing)")
    lines.append(f"   observed statistic         : {ps['obs_statistic']:+.3f}")
    lines.append(f"   two-sided permutation p    : {ps['p_two_sided']:.4f}")
    lines.append("=" * 80)
    report = "\n".join(lines)
    print(report)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    TXT.write_text(report + "\n")
    with open(OUT, "w") as fh:
        json.dump({"n": len(recs), "n_perm": N_PERM, "seed": SEED,
                   "firth": firth, "perm_marginal": pm, "perm_stratified": ps},
                  fh, indent=2)


if __name__ == "__main__":
    main()
