"""Statistical analysis module for SecDrift.

Provides rigorous statistical analysis for measuring sector-conditioned
security drift in LLM-generated code.

Key Features:
- Chi-square tests with Bonferroni correction
- Effect size calculations (Cramér's V, Cohen's h)
- Confidence intervals for proportions
- Post-hoc power analysis
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of a drift analysis between different prompt types."""

    sector: str
    
    # Baseline (generic terminology, no context)
    baseline_vuln_rate: float
    baseline_n: int
    
    # Industry (sector terminology + full context)
    industry_vuln_rate: float
    industry_n: int
    
    # Drift measurements
    drift: float  # Industry - Baseline (percentage points)
    relative_drift: float  # (Industry - Baseline) / Baseline
    
    # Statistical tests
    chi_square: float
    p_value: float
    p_value_corrected: float  # Bonferroni corrected
    cramers_v: float  # Effect size
    cohens_h: float  # Effect size for proportions
    significant: bool  # p_value_corrected < alpha
    classification: str  # "protective", "neutral", "risk-inducing"
    
    confidence_interval: Tuple[float, float]  # 95% CI for drift

    # Matched baseline (sector terminology, no context) - optional fields last
    matched_baseline_vuln_rate: Optional[float] = None
    matched_baseline_n: Optional[int] = None
    terminology_drift: Optional[float] = None  # Matched_baseline - Baseline
    context_drift: Optional[float] = None  # Industry - Matched_baseline


@dataclass
class AnalysisResults:
    """Complete analysis results for a benchmark run."""

    # Summary statistics
    total_evaluations: int
    models_analyzed: List[str]
    sectors_analyzed: List[str]
    cwes_analyzed: List[str]

    # Baseline metrics
    baseline_vuln_rate: float
    baseline_vuln_count: int
    baseline_total: int

    # Per-sector drift results
    sector_results: Dict[str, DriftResult]

    # Per-model results
    model_results: Dict[str, Dict[str, DriftResult]]

    # Statistical summary
    significant_drifts: int
    mean_drift: float
    overall_p_value: float

    # Metadata
    alpha: float = 0.05
    correction_method: str = "bonferroni"


def wilson_confidence_interval(
    successes: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion.

    Better than normal approximation for small samples and extreme proportions.
    """
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - margin), min(1, center + margin))


def cramers_v(chi2: float, n: int, min_dim: int) -> float:
    """Calculate Cramér's V effect size from chi-square statistic.

    Args:
        chi2: Chi-square statistic
        n: Total sample size
        min_dim: min(rows - 1, cols - 1) for contingency table

    Returns:
        Cramér's V (0-1 scale)
    """
    if n == 0 or min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h effect size for comparing two proportions.

    Args:
        p1: First proportion
        p2: Second proportion

    Returns:
        Cohen's h (can be negative)
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1)) if p1 > 0 else 0
    phi2 = 2 * np.arcsin(np.sqrt(p2)) if p2 > 0 else 0
    return phi1 - phi2


def interpret_effect_size(v: float) -> str:
    """Interpret Cramér's V effect size."""
    if abs(v) < 0.1:
        return "negligible"
    elif abs(v) < 0.3:
        return "small"
    elif abs(v) < 0.5:
        return "medium"
    else:
        return "large"


def classify_drift(drift_pp: float, significant: bool) -> str:
    """Classify drift as protective, neutral, or risk-inducing.

    Args:
        drift_pp: Drift in percentage points
        significant: Whether the drift is statistically significant
    """
    if not significant:
        return "neutral"
    elif drift_pp < -2.0:  # More than 2pp reduction
        return "protective"
    elif drift_pp > 2.0:  # More than 2pp increase
        return "risk-inducing"
    else:
        return "neutral"


def analyze_drift(
    baseline_results: List[Dict[str, Any]],
    industry_results: List[Dict[str, Any]],
    sector: str,
    alpha: float = 0.05,
    n_comparisons: int = 1,
    matched_baseline_results: Optional[List[Dict[str, Any]]] = None,
) -> DriftResult:
    """Analyze security drift between baseline and industry prompts.

    Args:
        baseline_results: List of baseline evaluation results
        industry_results: List of industry evaluation results
        sector: Sector name for this comparison
        alpha: Significance level
        n_comparisons: Number of comparisons for Bonferroni correction
        matched_baseline_results: Optional list of matched baseline results (terminology only)

    Returns:
        DriftResult with statistical analysis
    """
    # Count vulnerabilities - baseline
    baseline_vuln = sum(1 for r in baseline_results if r.get("is_vulnerable", False))
    baseline_total = len(baseline_results)
    baseline_rate = baseline_vuln / baseline_total if baseline_total > 0 else 0

    # Count vulnerabilities - industry
    industry_vuln = sum(1 for r in industry_results if r.get("is_vulnerable", False))
    industry_total = len(industry_results)
    industry_rate = industry_vuln / industry_total if industry_total > 0 else 0

    # Count vulnerabilities - matched baseline (if provided)
    matched_baseline_rate = None
    matched_baseline_total = None
    terminology_drift_pp = None
    context_drift_pp = None
    
    if matched_baseline_results:
        matched_baseline_vuln = sum(1 for r in matched_baseline_results if r.get("is_vulnerable", False))
        matched_baseline_total = len(matched_baseline_results)
        matched_baseline_rate = matched_baseline_vuln / matched_baseline_total if matched_baseline_total > 0 else 0
        
        # Calculate decomposed drifts
        terminology_drift_pp = (matched_baseline_rate - baseline_rate) * 100  # Effect of terminology
        context_drift_pp = (industry_rate - matched_baseline_rate) * 100  # Effect of context

    # Total drift calculations
    drift_pp = (industry_rate - baseline_rate) * 100  # Percentage points
    relative_drift = (drift_pp / (baseline_rate * 100)) if baseline_rate > 0 else 0

    # Chi-square test (baseline vs industry)
    contingency = np.array([
        [baseline_vuln, baseline_total - baseline_vuln],
        [industry_vuln, industry_total - industry_vuln]
    ])

    # Use Fisher's exact test if any expected count < 5
    expected = stats.contingency.expected_freq(contingency)
    if np.any(expected < 5):
        _, p_value = stats.fisher_exact(contingency)
        chi2 = 0  # Not applicable for Fisher's
    else:
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    # Bonferroni correction
    p_value_corrected = min(1.0, p_value * n_comparisons)

    # Effect sizes
    total_n = baseline_total + industry_total
    v = cramers_v(chi2, total_n, 1) if chi2 > 0 else 0
    h = cohens_h(industry_rate, baseline_rate)

    # Significance and classification
    significant = p_value_corrected < alpha
    classification = classify_drift(drift_pp, significant)

    # Confidence interval for drift (using delta method approximation)
    se_diff = np.sqrt(
        baseline_rate * (1 - baseline_rate) / baseline_total +
        industry_rate * (1 - industry_rate) / industry_total
    ) if baseline_total > 0 and industry_total > 0 else 0

    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = (drift_pp - z * se_diff * 100)
    ci_upper = (drift_pp + z * se_diff * 100)

    return DriftResult(
        sector=sector,
        baseline_vuln_rate=baseline_rate * 100,
        baseline_n=baseline_total,
        matched_baseline_vuln_rate=matched_baseline_rate * 100 if matched_baseline_rate is not None else None,
        matched_baseline_n=matched_baseline_total,
        industry_vuln_rate=industry_rate * 100,
        industry_n=industry_total,
        drift=drift_pp,
        terminology_drift=terminology_drift_pp,
        context_drift=context_drift_pp,
        relative_drift=relative_drift,
        chi_square=chi2,
        p_value=p_value,
        p_value_corrected=p_value_corrected,
        cramers_v=v,
        cohens_h=h,
        significant=significant,
        classification=classification,
        confidence_interval=(ci_lower, ci_upper),
    )


def analyze_benchmark_results(
    results_path: Path,
    alpha: float = 0.05,
) -> AnalysisResults:
    """Analyze complete benchmark results from JSONL file.

    Args:
        results_path: Path to results JSONL file
        alpha: Significance level

    Returns:
        Complete AnalysisResults
    """
    # Load results
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        raise ValueError(f"No results found in {results_path}")

    # Separate by prompt type
    baseline_results = [r for r in results if r.get("prompt_type") == "baseline"]
    matched_baseline_results = [r for r in results if r.get("prompt_type") == "matched_baseline"]
    industry_results = [r for r in results if r.get("prompt_type") == "industry"]

    # Get unique sectors, models, CWEs
    sectors = list(set(r.get("sector") for r in industry_results if r.get("sector")))
    models = list(set(r.get("model", "unknown") for r in results))
    cwes = list(set(r.get("cwe", "unknown") for r in results))

    n_comparisons = len(sectors)  # For Bonferroni correction

    # Baseline statistics
    baseline_vuln = sum(1 for r in baseline_results if r.get("is_vulnerable", False))
    baseline_total = len(baseline_results)
    baseline_rate = baseline_vuln / baseline_total * 100 if baseline_total > 0 else 0

    # Analyze each sector
    sector_results = {}
    for sector in sectors:
        sector_industry = [r for r in industry_results if r.get("sector") == sector]
        sector_matched = [r for r in matched_baseline_results if r.get("sector") == sector]
        
        if sector_industry:
            sector_results[sector] = analyze_drift(
                baseline_results,
                sector_industry,
                sector,
                alpha,
                n_comparisons,
                matched_baseline_results=sector_matched if sector_matched else None,
            )

    # Analyze per model
    model_results = {}
    for model in models:
        model_baseline = [r for r in baseline_results if r.get("model") == model]
        model_results[model] = {}

        for sector in sectors:
            model_industry = [
                r for r in industry_results
                if r.get("model") == model and r.get("sector") == sector
            ]
            model_matched = [
                r for r in matched_baseline_results
                if r.get("model") == model and r.get("sector") == sector
            ]
            
            if model_baseline and model_industry:
                model_results[model][sector] = analyze_drift(
                    model_baseline,
                    model_industry,
                    sector,
                    alpha,
                    n_comparisons,
                    matched_baseline_results=model_matched if model_matched else None,
                )

    # Summary statistics
    significant_count = sum(1 for r in sector_results.values() if r.significant)
    drifts = [r.drift for r in sector_results.values()]
    mean_drift = np.mean(drifts) if drifts else 0

    # Overall test (any sector shows drift?)
    overall_vuln = sum(1 for r in industry_results if r.get("is_vulnerable", False))
    overall_total = len(industry_results)

    overall_contingency = np.array([
        [baseline_vuln, baseline_total - baseline_vuln],
        [overall_vuln, overall_total - overall_vuln]
    ])

    if np.any(stats.contingency.expected_freq(overall_contingency) < 5):
        _, overall_p = stats.fisher_exact(overall_contingency)
    else:
        _, overall_p, _, _ = stats.chi2_contingency(overall_contingency)

    return AnalysisResults(
        total_evaluations=len(results),
        models_analyzed=models,
        sectors_analyzed=sectors,
        cwes_analyzed=cwes,
        baseline_vuln_rate=baseline_rate,
        baseline_vuln_count=baseline_vuln,
        baseline_total=baseline_total,
        sector_results=sector_results,
        model_results=model_results,
        significant_drifts=significant_count,
        mean_drift=mean_drift,
        overall_p_value=overall_p,
        alpha=alpha,
        correction_method="bonferroni",
    )


def format_analysis_report(results: AnalysisResults) -> str:
    """Format analysis results as a readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SecDrift Analysis Report")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total evaluations: {results.total_evaluations}")
    lines.append(f"Models: {', '.join(results.models_analyzed)}")
    lines.append(f"Sectors: {', '.join(results.sectors_analyzed)}")
    lines.append(f"CWEs: {', '.join(results.cwes_analyzed)}")
    lines.append("")

    # Baseline
    lines.append("BASELINE")
    lines.append("-" * 40)
    lines.append(f"Vulnerability rate: {results.baseline_vuln_rate:.1f}%")
    lines.append(f"Vulnerable: {results.baseline_vuln_count}/{results.baseline_total}")
    lines.append("")

    # Per-sector results
    lines.append("SECTOR DRIFT ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Alpha: {results.alpha} (Bonferroni corrected)")
    lines.append("")

    for sector, result in sorted(results.sector_results.items()):
        sig_marker = "*" if result.significant else ""
        lines.append(f"{sector}:")
        lines.append(f"  Baseline (generic): {result.baseline_vuln_rate:.1f}%")
        
        # Show matched baseline if available
        if result.matched_baseline_vuln_rate is not None:
            lines.append(f"  Matched baseline (terminology): {result.matched_baseline_vuln_rate:.1f}%")
            lines.append(f"  Industry (terminology + context): {result.industry_vuln_rate:.1f}%")
            lines.append(f"")
            lines.append(f"  Total drift: {result.drift:+.1f}pp {sig_marker}")
            if result.terminology_drift is not None:
                lines.append(f"    - Terminology effect: {result.terminology_drift:+.1f}pp")
            if result.context_drift is not None:
                lines.append(f"    - Context effect: {result.context_drift:+.1f}pp")
        else:
            lines.append(f"  Industry: {result.industry_vuln_rate:.1f}%")
            lines.append(f"  Drift: {result.drift:+.1f}pp {sig_marker}")
        
        lines.append(f"  95% CI: [{result.confidence_interval[0]:.1f}, "
                    f"{result.confidence_interval[1]:.1f}]")
        lines.append(f"  p-value: {result.p_value:.4f} "
                    f"(corrected: {result.p_value_corrected:.4f})")
        lines.append(f"  Effect size (Cramér's V): {result.cramers_v:.3f} "
                    f"({interpret_effect_size(result.cramers_v)})")
        lines.append(f"  Classification: {result.classification}")
        
        n_info = f"  N: {result.industry_n} industry, {result.baseline_n} baseline"
        if result.matched_baseline_n:
            n_info += f", {result.matched_baseline_n} matched"
        lines.append(n_info)
        lines.append("")

    # Summary statistics
    lines.append("STATISTICAL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Significant drifts: {results.significant_drifts}/{len(results.sector_results)}")
    lines.append(f"Mean drift: {results.mean_drift:+.1f}pp")
    lines.append(f"Overall p-value: {results.overall_p_value:.4f}")
    lines.append("")
    lines.append("* = significant at corrected alpha")
    lines.append("=" * 70)

    return "\n".join(lines)


def export_results_json(results: AnalysisResults, output_path: Path):
    """Export analysis results to JSON."""
    data = {
        "summary": {
            "total_evaluations": results.total_evaluations,
            "models": results.models_analyzed,
            "sectors": results.sectors_analyzed,
            "cwes": results.cwes_analyzed,
            "alpha": results.alpha,
            "correction_method": results.correction_method,
        },
        "baseline": {
            "vulnerability_rate": results.baseline_vuln_rate,
            "vulnerable_count": results.baseline_vuln_count,
            "total": results.baseline_total,
        },
        "sectors": {
            sector: {
                "vulnerability_rate": r.industry_vuln_rate,
                "drift_pp": r.drift,
                "relative_drift": r.relative_drift,
                "p_value": r.p_value,
                "p_value_corrected": r.p_value_corrected,
                "cramers_v": r.cramers_v,
                "cohens_h": r.cohens_h,
                "significant": r.significant,
                "classification": r.classification,
                "confidence_interval": list(r.confidence_interval),
                "n": r.industry_n,
            }
            for sector, r in results.sector_results.items()
        },
        "overall": {
            "significant_drifts": results.significant_drifts,
            "mean_drift": results.mean_drift,
            "p_value": results.overall_p_value,
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Extended inferential statistics (added; no existing signatures changed)
# =============================================================================
#
# The functions above report several quantities as point estimates only:
#   * per-sector drift (industry vs baseline),
#   * overall drift and its terminology/context decomposition,
#   * per-model vulnerability rates.
#
# The helpers below attach inference to each of those comparisons and export
# one CSV per table. To reproduce the point estimates that are actually
# reported in the paper (e.g. baseline 14.0%, industry 11.4%, DeepSeek 1.8%),
# they apply the paper's "code-only" filter by default: drop the model that
# produced no analyzable code (gpt-oss-120b) and any record whose
# generated_code is empty or whitespace.
#
# Every emitted row uses the shared schema:
#   comparison, n1, n2, rate1, rate2, diff, p_raw, p_bonferroni,
#   significant, cohens_h, ci_low, ci_high
#
# Rates, diff and CI bounds are proportions in [0, 1]; diff = rate2 - rate1;
# cohens_h is signed to match diff. p_raw is the two-tailed Fisher's exact
# p-value on the [[vuln1, ok1], [vuln2, ok2]] table; p_bonferroni multiplies
# p_raw by the number of comparisons in that table (clamped to 1.0).
# ci_low/ci_high hold the Wilson 95% interval for the row's outcome rate
# (rate2 for two-group comparisons, rate1 for single-rate model rows).

EXTENDED_STATS_COLUMNS: List[str] = [
    "comparison",
    "n1",
    "n2",
    "rate1",
    "rate2",
    "diff",
    "p_raw",
    "p_bonferroni",
    "significant",
    "cohens_h",
    "ci_low",
    "ci_high",
]

# Model excluded by the code-only filter (produced no analyzable code).
CODE_ONLY_EXCLUDED_MODEL: str = "gpt-oss-120b"


def _load_result_records(results_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation records from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No results found in {results_path}")
    return records


def _filter_code_only(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the paper's code-only filter.

    Drops the excluded model and any record whose ``generated_code`` field is
    empty or whitespace.
    """
    kept: List[Dict[str, Any]] = []
    for r in records:
        if r.get("model") == CODE_ONLY_EXCLUDED_MODEL:
            continue
        if str(r.get("generated_code") or "").strip() == "":
            continue
        kept.append(r)
    return kept


def _vuln_count(records: List[Dict[str, Any]]) -> int:
    """Count records flagged vulnerable."""
    return sum(1 for r in records if r.get("is_vulnerable", False))


def _proportion(successes: int, n: int) -> float:
    """Safe proportion helper."""
    return successes / n if n > 0 else 0.0


def _fisher_two_sided_p(vuln1: int, n1: int, vuln2: int, n2: int) -> float:
    """Two-tailed Fisher's exact p-value on a 2x2 vulnerable/ok table."""
    if n1 == 0 or n2 == 0:
        return 1.0
    table = [[vuln1, n1 - vuln1], [vuln2, n2 - vuln2]]
    _, p_value = stats.fisher_exact(table)  # alternative="two-sided" default
    return float(p_value)


def _build_comparison_row(
    comparison: str,
    vuln1: int,
    n1: int,
    vuln2: int,
    n2: int,
    n_comparisons: int,
    alpha: float = 0.05,
    ndigits: int = 6,
) -> Dict[str, Any]:
    """Build one two-group comparison row with full inference.

    ``rate1``/``rate2`` are the two proportions being compared; the Wilson
    interval reported is for ``rate2`` (the outcome group). Cohen's ``h`` is
    signed to match ``diff = rate2 - rate1``.
    """
    rate1 = _proportion(vuln1, n1)
    rate2 = _proportion(vuln2, n2)
    diff = rate2 - rate1

    p_raw = _fisher_two_sided_p(vuln1, n1, vuln2, n2)
    p_bonferroni = min(1.0, p_raw * n_comparisons)
    significant = bool(p_bonferroni < alpha)

    # Reuse the module's existing helpers (signed so h matches diff).
    h = cohens_h(rate2, rate1)
    ci_low, ci_high = wilson_confidence_interval(vuln2, n2)

    return {
        "comparison": comparison,
        "n1": n1,
        "n2": n2,
        "rate1": round(rate1, ndigits),
        "rate2": round(rate2, ndigits),
        "diff": round(diff, ndigits),
        "p_raw": round(p_raw, ndigits),
        "p_bonferroni": round(p_bonferroni, ndigits),
        "significant": significant,
        "cohens_h": round(h, ndigits),
        "ci_low": round(ci_low, ndigits),
        "ci_high": round(ci_high, ndigits),
    }


def compute_sector_stats(
    records: List[Dict[str, Any]],
    alpha: float = 0.05,
    code_only: bool = True,
    ndigits: int = 6,
) -> List[Dict[str, Any]]:
    """Per-sector drift (industry vs baseline) with Fisher's exact,
    Bonferroni over the sectors, Cohen's h, and Wilson 95% CIs.

    Bonferroni uses ``alpha / n_sectors`` (i.e. 0.05/8 for the standard
    8-sector design), realized as ``p_bonferroni = p_raw * n_sectors``.
    """
    recs = _filter_code_only(records) if code_only else list(records)
    baseline = [r for r in recs if r.get("prompt_type") == "baseline"]
    industry = [r for r in recs if r.get("prompt_type") == "industry"]

    sectors = sorted({r.get("sector") for r in industry if r.get("sector")})
    n_comparisons = max(1, len(sectors))

    base_v, base_n = _vuln_count(baseline), len(baseline)

    rows: List[Dict[str, Any]] = []
    for sector in sectors:
        sec_records = [r for r in industry if r.get("sector") == sector]
        sec_v, sec_n = _vuln_count(sec_records), len(sec_records)
        rows.append(
            _build_comparison_row(
                comparison=f"{sector}: industry_vs_baseline",
                vuln1=base_v,
                n1=base_n,
                vuln2=sec_v,
                n2=sec_n,
                n_comparisons=n_comparisons,
                alpha=alpha,
                ndigits=ndigits,
            )
        )
    return rows


def compute_decomposition_stats(
    records: List[Dict[str, Any]],
    alpha: float = 0.05,
    code_only: bool = True,
    ndigits: int = 6,
) -> List[Dict[str, Any]]:
    """Overall drift and its decomposition, each with inference.

    Three comparisons (Bonferroni factor = 3):
      * overall:     baseline vs industry
      * terminology: baseline vs matched_baseline
      * context:     matched_baseline vs industry
    """
    recs = _filter_code_only(records) if code_only else list(records)
    baseline = [r for r in recs if r.get("prompt_type") == "baseline"]
    matched = [r for r in recs if r.get("prompt_type") == "matched_baseline"]
    industry = [r for r in recs if r.get("prompt_type") == "industry"]

    base_v, base_n = _vuln_count(baseline), len(baseline)
    match_v, match_n = _vuln_count(matched), len(matched)
    ind_v, ind_n = _vuln_count(industry), len(industry)

    n_comparisons = 3
    rows = [
        _build_comparison_row(
            "overall: baseline_vs_industry",
            base_v, base_n, ind_v, ind_n, n_comparisons, alpha, ndigits,
        ),
        _build_comparison_row(
            "terminology: baseline_vs_matched_baseline",
            base_v, base_n, match_v, match_n, n_comparisons, alpha, ndigits,
        ),
        _build_comparison_row(
            "context: matched_baseline_vs_industry",
            match_v, match_n, ind_v, ind_n, n_comparisons, alpha, ndigits,
        ),
    ]
    return rows


def compute_model_stats(
    records: List[Dict[str, Any]],
    code_only: bool = True,
    confidence: float = 0.95,
    ndigits: int = 6,
) -> List[Dict[str, Any]]:
    """Per-model overall vulnerability rate with a Wilson CI.

    Each row is a single rate (no second group), so the comparison-only
    columns (n2, rate2, diff, p_raw, p_bonferroni, significant, cohens_h) are
    left empty and ci_low/ci_high hold the Wilson interval for rate1.
    """
    recs = _filter_code_only(records) if code_only else list(records)
    models = sorted({r.get("model", "unknown") for r in recs})

    rows: List[Dict[str, Any]] = []
    for model in models:
        model_records = [r for r in recs if r.get("model", "unknown") == model]
        vuln, n = _vuln_count(model_records), len(model_records)
        rate = _proportion(vuln, n)
        ci_low, ci_high = wilson_confidence_interval(vuln, n, confidence)
        rows.append(
            {
                "comparison": f"{model}: overall_rate",
                "n1": n,
                "n2": None,
                "rate1": round(rate, ndigits),
                "rate2": None,
                "diff": None,
                "p_raw": None,
                "p_bonferroni": None,
                "significant": None,
                "cohens_h": None,
                "ci_low": round(ci_low, ndigits),
                "ci_high": round(ci_high, ndigits),
            }
        )
    return rows


def _write_stats_csv(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to CSV using the shared schema; None becomes an empty cell."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXTENDED_STATS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {col: ("" if row.get(col) is None else row.get(col))
                 for col in EXTENDED_STATS_COLUMNS}
            )


def export_extended_stats_csvs(
    results_path: Path,
    output_dir: Path,
    alpha: float = 0.05,
    code_only: bool = True,
) -> Dict[str, Path]:
    """Compute all three inference tables and write them to CSV.

    Writes ``sector_stats.csv``, ``decomposition_stats.csv`` and
    ``model_stats.csv`` into ``output_dir``. Returns a mapping of table name
    to the written file path.
    """
    records = _load_result_records(Path(results_path))
    output_dir = Path(output_dir)

    sector_rows = compute_sector_stats(records, alpha=alpha, code_only=code_only)
    decomposition_rows = compute_decomposition_stats(records, alpha=alpha, code_only=code_only)
    model_rows = compute_model_stats(records, code_only=code_only)

    paths = {
        "sector_stats": output_dir / "sector_stats.csv",
        "decomposition_stats": output_dir / "decomposition_stats.csv",
        "model_stats": output_dir / "model_stats.csv",
    }
    _write_stats_csv(paths["sector_stats"], sector_rows)
    _write_stats_csv(paths["decomposition_stats"], decomposition_rows)
    _write_stats_csv(paths["model_stats"], model_rows)

    logger.info("Wrote extended stats CSVs to %s", output_dir)
    return paths


if __name__ == "__main__":
    import sys

    _default_data = Path("results/latest_run/merged_results_5rep.jsonl")
    _results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_data
    _output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/extended_analysis")

    _written = export_extended_stats_csvs(_results_path, _output_dir)
    for _name, _path in _written.items():
        print(f"{_name}: {_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze SecDrift benchmark results (JSONL)"
    )
    parser.add_argument("results_file", type=Path, help="Path to results JSONL")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Optional output path (.json exports JSON, otherwise a text report)"
    )
    parser.add_argument("-a", "--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    _args = parser.parse_args()

    _results = analyze_benchmark_results(_args.results_file, alpha=_args.alpha)
    _report = format_analysis_report(_results)
    print(_report)
    if _args.output:
        if _args.output.suffix == ".json":
            export_results_json(_results, _args.output)
        else:
            _args.output.write_text(_report)
        print(f"\nWrote {_args.output}")
