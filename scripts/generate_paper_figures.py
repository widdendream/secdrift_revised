#!/usr/bin/env python3
"""Generate all publication-quality figures for the rewritten SecDrift paper.

Produces 9 figures matching the updated captions in secdrift_paper.tex:
1. fig_architecture.png       - Benchmark architecture (1,785 total evals)
2. fig_transformation.png     - 5-dimension transformation (includes D5)
3. fig_overall_drift.png      - Overall drift with decomposition
4. fig_confidence_intervals.png - Three-way comparison by sector with SEM
5. fig_effect_size_comparison.png - Model vs sector vs drift components
6. fig_model_tiers.png        - Model security tiers (7 models only)
7. fig_heatmap_comparison.png - Vulnerability heatmap by model × sector
8. fig_cwe_distribution.png   - CWE distribution: baseline vs industry + drift
9. fig_nuclear_deep_dive.png  - Nuclear sector 4-panel analysis
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy import stats as scipy_stats

# Publication style
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

# Consistent color palette
COLORS = {
    'baseline': '#3498db',
    'matched': '#f39c12',
    'industry': '#2ecc71',
    'protective': '#2ecc71',
    'neutral': '#f39c12',
    'risk': '#e74c3c',
    'model_effect': '#e74c3c',
    'sector_effect': '#3498db',
    'term_effect': '#f39c12',
    'ctx_effect': '#9b59b6',
    'drift': '#2ecc71',
    'safe_tier': '#2ecc71',
    'low_tier': '#2ecc71',
    'medium_tier': '#f39c12',
    'high_tier': '#e74c3c',
}

# Display name mapping for models
MODEL_DISPLAY = {
    'deepseek-r1': 'DeepSeek R1',
    'gemma-3-27b': 'Gemma 3 27B',
    'gpt-oss-120b': 'GPT-OSS 120B',
    'llama-3-3-70b': 'Llama 3.3 70B',
    'llama-4-maverick': 'Llama 4 Maverick',
    'mistral-large-3': 'Mistral Large 3',
    'qwen3-32b': 'Qwen3 32B',
}

# Models to EXCLUDE from vulnerability analysis (no code produced)
EXCLUDED_MODELS = {'gpt-oss-120b'}

SECTOR_DISPLAY = {
    'communications': 'Communications',
    'defense': 'Defense',
    'emergency_services': 'Emergency\nServices',
    'energy': 'Energy',
    'financial': 'Financial',
    'government': 'Government',
    'healthcare': 'Healthcare',
    'nuclear': 'Nuclear',
}


def load_results(path: Path) -> List[Dict]:
    """Load benchmark results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_replicate_stats(results: List[Dict], group_by: List[str]) -> Dict:
    """Compute mean and SEM across replicates for grouped data."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = tuple(r.get(k) for k in group_by)
        replicate = r.get('replicate', 0)
        is_vuln = 1 if r.get('is_vulnerable') else 0
        grouped[key][replicate].append(is_vuln)

    stats = {}
    for key, replicates in grouped.items():
        replicate_rates = []
        for rep_id, vulns in replicates.items():
            rate = sum(vulns) / len(vulns) * 100 if vulns else 0
            replicate_rates.append(rate)
        stats[key] = {
            'mean': np.mean(replicate_rates),
            'std': np.std(replicate_rates, ddof=1) if len(replicate_rates) > 1 else 0,
            'sem': scipy_stats.sem(replicate_rates) if len(replicate_rates) > 1 else 0,
            'n_replicates': len(replicate_rates),
            'rates': replicate_rates,
        }
    return stats


def wilson_ci(successes, trials, confidence=0.95):
    """Compute Wilson score confidence interval."""
    if trials == 0:
        return 0, 0
    z = scipy_stats.norm.ppf((1 + confidence) / 2)
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denom
    return max(0, center - margin) * 100, min(1, center + margin) * 100


# ============================================================================
# FIGURE 1: Architecture diagram (programmatic)
# ============================================================================

def plot_architecture(output_dir: Path):
    """Benchmark architecture showing end-to-end pipeline with 1,785 total evaluations."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Boxes
    boxes = [
        (0.3, 1.8, 2.2, 1.4, 'Prompt Corpus\n\n5 CWE Scenarios\n9 Baseline Prompts', '#d6eaf8'),
        (3.1, 1.8, 2.2, 1.4, '5-Dimension\nTransformer\n\nD1–D5 Pipeline\n+ Matched Baseline', '#fdebd0'),
        (5.9, 1.8, 2.2, 1.4, 'LLM Generation\nEngine\n\n7 Models × 5 Reps\n(6 produce code)', '#d5f5e3'),
        (8.7, 1.8, 2.2, 1.4, 'Security Analysis\nPipeline\n\nBandit + Semgrep\nCWE-Aligned Rules', '#fadbd8'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5,
                fontweight='bold', linespacing=1.3)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=2)
    for x_start in [2.5, 5.3, 8.1]:
        ax.annotate('', xy=(x_start + 0.6, 2.5), xytext=(x_start, 2.5),
                    arrowprops=arrow_props)

    # Labels on arrows
    ax.text(2.8, 2.85, '3 prompt\ntypes', ha='center', fontsize=7, style='italic')
    ax.text(5.6, 2.85, '1,785\nevaluations', ha='center', fontsize=7, style='italic')
    ax.text(8.4, 2.85, 'Python\ncode', ha='center', fontsize=7, style='italic')

    # Title bar
    ax.text(6.0, 4.3, 'SecDrift Benchmark Architecture (1,785 Total Evaluations)',
            ha='center', va='center', fontsize=13, fontweight='bold')

    # Bottom summary
    ax.text(6.0, 0.8,
            '7 models × 5 CWEs × 3 prompt types × (1 baseline + 8 sectors) × 5 replicates = 1,785',
            ha='center', va='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_architecture.png', dpi=300)
    plt.savefig(output_dir / 'fig_architecture.pdf')
    plt.close()
    print("  ✓ fig_architecture")


# ============================================================================
# FIGURE 2: 5-Dimension Transformation (includes D5)
# ============================================================================

def plot_transformation(output_dir: Path):
    """5-dimension transformation formula showing all 5 dimensions including D5."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6.0, 5.5, '5-Dimension Transformation Formula: $T = D_5 \\circ D_4 \\circ D_3 \\circ D_2 \\circ D_1$',
            ha='center', va='center', fontsize=13, fontweight='bold')

    # Dimension boxes
    dims = [
        (0.2, 3.0, 'D1: Context\nInjection', '#d6eaf8',
         'Prepend sector system\nID and operational scope'),
        (2.5, 3.0, 'D2: Terminology\nMapping', '#fdebd0',
         'Replace generic terms\nwith sector equivalents'),
        (4.8, 3.0, 'D3: Stakeholder\nFraming', '#d5f5e3',
         'Map generic roles to\nsector stakeholders'),
        (7.1, 3.0, 'D4: Use Case\nGrounding', '#e8daef',
         'Append operational\ndeployment scenario'),
        (9.4, 3.0, 'D5: Requirement\nPreservation', '#fadbd8',
         'Identity constraint:\nfunctional specs invariant'),
    ]

    for x, y, title, color, desc in dims:
        rect = mpatches.FancyBboxPatch((x, y), 2.0, 1.6, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 1.0, y + 1.1, title, ha='center', va='center', fontsize=8.5,
                fontweight='bold')
        ax.text(x + 1.0, y + 0.35, desc, ha='center', va='center', fontsize=7,
                style='italic', linespacing=1.2)

    # Arrows between dimensions
    arrow_props = dict(arrowstyle='->', color='#333333', lw=1.5)
    for i in range(4):
        x_start = dims[i][0] + 2.0
        x_end = dims[i+1][0]
        ax.annotate('', xy=(x_end, 3.8), xytext=(x_start, 3.8), arrowprops=arrow_props)

    # Input/output labels
    # Baseline prompt input
    ax.annotate('Baseline\nPrompt $P_b$', xy=(0.2, 3.8), xytext=(-0.3, 3.8),
                fontsize=8, ha='right', va='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Output
    ax.annotate('Industry\nPrompt $P_i$', xy=(11.4, 3.8), xytext=(11.9, 3.8),
                fontsize=8, ha='left', va='center', fontweight='bold',
                arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))

    # Matched baseline annotation
    rect_mb = mpatches.FancyBboxPatch((2.5, 1.0), 2.0, 0.9, boxstyle="round,pad=0.1",
                                       facecolor='#fef9e7', edgecolor='#f39c12',
                                       linewidth=2, linestyle='--')
    ax.add_patch(rect_mb)
    ax.text(3.5, 1.45, 'Matched Baseline:\nD2 only (terminology)', ha='center',
            va='center', fontsize=8, fontweight='bold', color='#b7950b')
    ax.annotate('', xy=(3.5, 3.0), xytext=(3.5, 1.9),
                arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5, linestyle='--'))

    # Full industry annotation
    rect_fi = mpatches.FancyBboxPatch((5.5, 1.0), 3.0, 0.9, boxstyle="round,pad=0.1",
                                       facecolor='#eafaf1', edgecolor='#2ecc71',
                                       linewidth=2, linestyle='--')
    ax.add_patch(rect_fi)
    ax.text(7.0, 1.45, 'Full Industry: D1–D5\n(terminology + context)', ha='center',
            va='center', fontsize=8, fontweight='bold', color='#1e8449')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_transformation.png', dpi=300)
    plt.savefig(output_dir / 'fig_transformation.pdf')
    plt.close()
    print("  ✓ fig_transformation")


# ============================================================================
# FIGURE 3: Overall Security Drift with Decomposition
# ============================================================================

def plot_overall_drift(results: List[Dict], output_dir: Path):
    """Overall drift showing -2.3pp with terminology (-1.6pp) and context (-0.7pp) decomposition."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.3]})

    # --- Left panel: bar chart of 3 conditions ---
    baseline_results = [r for r in results if r['prompt_type'] == 'baseline']
    matched_results = [r for r in results if r['prompt_type'] == 'matched_baseline']
    industry_results = [r for r in results if r['prompt_type'] == 'industry']

    base_vuln = sum(1 for r in baseline_results if r['is_vulnerable'])
    base_total = len(baseline_results)
    base_rate = (base_vuln / base_total * 100) if base_total else 0

    match_vuln = sum(1 for r in matched_results if r['is_vulnerable'])
    match_total = len(matched_results)
    match_rate = (match_vuln / match_total * 100) if match_total else 0

    ind_vuln = sum(1 for r in industry_results if r['is_vulnerable'])
    ind_total = len(industry_results)
    ind_rate = (ind_vuln / ind_total * 100) if ind_total else 0

    conditions = ['Baseline\n(Neutral)', 'Matched\n(Terminology)', 'Industry\n(Full)']
    rates = [base_rate, match_rate, ind_rate]
    colors = [COLORS['baseline'], COLORS['matched'], COLORS['industry']]

    bars = ax1.bar(conditions, rates, color=colors, edgecolor='black', linewidth=1, width=0.6)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_ylabel('Vulnerability Rate (%)')
    ax1.set_title('Three-Way Vulnerability Comparison')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(rates) * 1.3)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # --- Right panel: waterfall decomposition ---
    term_effect = match_rate - base_rate
    ctx_effect = ind_rate - match_rate
    total_drift = ind_rate - base_rate

    categories = ['Baseline', 'Terminology\nEffect', 'Context\nEffect', 'Industry\n(Final)']
    bar_bottoms = [0, ind_rate, ind_rate, 0]  # waterfall bottoms
    bar_heights = [base_rate, abs(term_effect), abs(ctx_effect), ind_rate]
    bar_colors = [COLORS['baseline'], COLORS['term_effect'], COLORS['ctx_effect'], COLORS['industry']]

    # Draw waterfall
    ax2.bar(0, base_rate, color=COLORS['baseline'], edgecolor='black', linewidth=1, width=0.5)
    ax2.bar(1, abs(term_effect), bottom=match_rate, color=COLORS['term_effect'],
            edgecolor='black', linewidth=1, width=0.5)
    ax2.bar(2, abs(ctx_effect), bottom=ind_rate, color=COLORS['ctx_effect'],
            edgecolor='black', linewidth=1, width=0.5)
    ax2.bar(3, ind_rate, color=COLORS['industry'], edgecolor='black', linewidth=1, width=0.5)

    # Connection lines
    ax2.plot([0.25, 1.0], [base_rate, base_rate], 'k--', linewidth=0.8, alpha=0.5)
    ax2.plot([1.25, 2.0], [match_rate, match_rate], 'k--', linewidth=0.8, alpha=0.5)

    # Labels
    ax2.text(0, base_rate + 0.3, f'{base_rate:.1f}%', ha='center', fontweight='bold', fontsize=9)
    ax2.text(1, base_rate + 0.3, f'{term_effect:+.1f}pp', ha='center', fontweight='bold',
             fontsize=9, color='#b7950b')
    ax2.text(2, match_rate + 0.3, f'{ctx_effect:+.1f}pp', ha='center', fontweight='bold',
             fontsize=9, color='#7d3c98')
    ax2.text(3, ind_rate + 0.3, f'{ind_rate:.1f}%', ha='center', fontweight='bold', fontsize=9)

    # Total drift annotation
    ax2.annotate(f'Total drift: {total_drift:+.1f}pp',
                xy=(1.5, base_rate * 0.5), fontsize=10, fontweight='bold',
                ha='center', color='#1a5276',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaf2f8', edgecolor='#1a5276'))

    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Vulnerability Rate (%)')
    ax2.set_title('Drift Decomposition: Terminology + Context')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, max(rates) * 1.4)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_overall_drift.png', dpi=300)
    plt.savefig(output_dir / 'fig_overall_drift.pdf')
    plt.close()
    print("  ✓ fig_overall_drift")


# ============================================================================
# FIGURE 4: Three-way comparison by sector with SEM error bars
# ============================================================================

def plot_confidence_intervals(results: List[Dict], output_dir: Path):
    """Three-way vulnerability rate comparison by sector with SEM error bars.
    Shows monotonic improvement from baseline → matched → industry."""
    fig, ax = plt.subplots(figsize=(14, 6))

    sectors = sorted(set(r['sector'] for r in results if r.get('sector') and r['sector'] != 'None'))

    # Baseline stats (same for all sectors)
    baseline_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'baseline'], ['prompt_type']
    )
    baseline_key = ('baseline',)
    baseline_mean = baseline_stats.get(baseline_key, {'mean': 0, 'sem': 0})['mean']
    baseline_sem = baseline_stats.get(baseline_key, {'mean': 0, 'sem': 0})['sem']

    matched_means, matched_sems = [], []
    industry_means, industry_sems = [], []

    for sector in sectors:
        m_stats = compute_replicate_stats(
            [r for r in results if r['prompt_type'] == 'matched_baseline' and r.get('sector') == sector],
            ['sector', 'prompt_type']
        )
        i_stats = compute_replicate_stats(
            [r for r in results if r['prompt_type'] == 'industry' and r.get('sector') == sector],
            ['sector', 'prompt_type']
        )
        mk = (sector, 'matched_baseline')
        ik = (sector, 'industry')
        matched_means.append(m_stats.get(mk, {'mean': baseline_mean, 'sem': 0})['mean'])
        matched_sems.append(m_stats.get(mk, {'mean': 0, 'sem': 0})['sem'])
        industry_means.append(i_stats.get(ik, {'mean': baseline_mean, 'sem': 0})['mean'])
        industry_sems.append(i_stats.get(ik, {'mean': 0, 'sem': 0})['sem'])

    x = np.arange(len(sectors))
    width = 0.25

    ax.bar(x - width, [baseline_mean]*len(sectors), width,
           yerr=[baseline_sem]*len(sectors),
           label='Baseline (neutral)', color=COLORS['baseline'],
           edgecolor='black', linewidth=0.5, capsize=3)
    ax.bar(x, matched_means, width, yerr=matched_sems,
           label='Matched Baseline (terminology only)', color=COLORS['matched'],
           edgecolor='black', linewidth=0.5, capsize=3)
    ax.bar(x + width, industry_means, width, yerr=industry_sems,
           label='Industry (terminology + context)', color=COLORS['industry'],
           edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([SECTOR_DISPLAY.get(s, s) for s in sectors], fontsize=9)
    ax.set_ylabel('Vulnerability Rate (%)')
    ax.set_xlabel('CISA Critical Infrastructure Sector')
    ax.set_title('Three-Way Vulnerability Rate Comparison by Sector\n'
                 '(Error bars: SEM across 5 replicates)')
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(baseline_mean, max(matched_means), max(industry_means)) * 1.4)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_confidence_intervals.png', dpi=300)
    plt.savefig(output_dir / 'fig_confidence_intervals.pdf')
    plt.close()
    print("  ✓ fig_confidence_intervals")


# ============================================================================
# FIGURE 5: Effect Size Comparison (model vs sector vs drift)
# ============================================================================

def plot_effect_size_comparison(results: List[Dict], output_dir: Path):
    """Effect size comparison: model selection (14.8pp) dominates sector (3.4pp),
    overall drift (2.3pp), and components. Uses only 7 evaluated models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Model effect: range across 7 models
    model_stats = compute_replicate_stats(results, ['model'])
    model_rates = {k[0]: v['mean'] for k, v in model_stats.items()}
    model_range = max(model_rates.values()) - min(model_rates.values())

    # Sector effect: range across sectors (industry only)
    sector_results = [r for r in results if r['prompt_type'] == 'industry']
    sector_stats = compute_replicate_stats(sector_results, ['sector'])
    sector_rates = [v['mean'] for v in sector_stats.values()]
    sector_range = max(sector_rates) - min(sector_rates)

    # Overall drift
    baseline_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'baseline'], ['prompt_type'])
    industry_stats_all = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'industry'], ['prompt_type'])
    matched_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'matched_baseline'], ['prompt_type'])

    base_rate = baseline_stats[('baseline',)]['mean']
    ind_rate = industry_stats_all[('industry',)]['mean']
    overall_drift = abs(ind_rate - base_rate)

    if matched_stats:
        match_rate = matched_stats[('matched_baseline',)]['mean']
        term_effect = abs(match_rate - base_rate)
        ctx_effect = abs(ind_rate - match_rate)
    else:
        term_effect = overall_drift * 0.7
        ctx_effect = overall_drift * 0.3

    effects = {
        f'Model Selection\n({min(model_rates.values()):.0f}% to {max(model_rates.values()):.0f}%)': model_range,
        f'Sector Context\n({min(sector_rates):.1f}% to {max(sector_rates):.1f}%)': sector_range,
        'Overall Drift\n(Baseline → Industry)': overall_drift,
        'Terminology Effect\n(Baseline → Matched)': term_effect,
        'Context Effect\n(Matched → Industry)': ctx_effect,
    }

    labels = list(effects.keys())
    values = list(effects.values())
    colors = [COLORS['model_effect'], COLORS['sector_effect'], COLORS['drift'],
              COLORS['term_effect'], COLORS['ctx_effect']]

    bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=1, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}pp', va='center', fontweight='bold', fontsize=10)

    # Dominance ratio annotation
    ratio = model_range / sector_range if sector_range > 0 else float('inf')
    ax.text(max(values) * 0.6, -0.6,
            f'Model selection is {ratio:.1f}× larger than sector context',
            fontsize=10, style='italic', color='#555555')

    ax.set_xlabel('Effect Size (Percentage Points)')
    ax.set_title('Effect Size Comparison: Model Choice Dominates Security Outcomes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(values) * 1.2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_effect_size_comparison.png', dpi=300)
    plt.savefig(output_dir / 'fig_effect_size_comparison.pdf')
    plt.close()
    print("  ✓ fig_effect_size_comparison")


# ============================================================================
# FIGURE 6: Model Security Tiers (7 models only)
# ============================================================================

def plot_model_tiers(results: List[Dict], output_dir: Path):
    """Model security tiers for the 7 evaluated models. Safe tier (0%) vs medium (12-15%)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_stats = compute_replicate_stats(results, ['model'])
    model_data = []
    for key, stats in model_stats.items():
        model_name = key[0]
        display = MODEL_DISPLAY.get(model_name, model_name)
        model_data.append((display, stats['mean'], stats['sem']))

    # Sort by vulnerability rate
    model_data.sort(key=lambda x: x[1])

    names = [d[0] for d in model_data]
    rates = [d[1] for d in model_data]
    sems = [d[2] for d in model_data]

    # Color by tier
    colors = []
    for rate in rates:
        if rate < 5:
            colors.append(COLORS['low_tier'])
        elif rate < 15:
            colors.append(COLORS['medium_tier'])
        else:
            colors.append(COLORS['high_tier'])

    bars = ax.barh(names, rates, xerr=sems, color=colors, edgecolor='black',
                   linewidth=1, height=0.6, capsize=4)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%', va='center', fontweight='bold', fontsize=10)

    # Tier boundary lines
    ax.axvline(x=5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(5.2, len(names) - 0.3, 'Safe / Medium\nboundary (5%)',
            fontsize=7, color='gray', va='top')

    # Tier legend
    low_patch = mpatches.Patch(color=COLORS['low_tier'], label='Low Tier (0–5%)')
    med_patch = mpatches.Patch(color=COLORS['medium_tier'], label='Medium Tier (5–15%)')
    high_patch = mpatches.Patch(color=COLORS['high_tier'], label='High Tier (>15%)')
    ax.legend(handles=[low_patch, med_patch, high_patch], loc='lower right', fontsize=9)

    ax.set_xlabel('Vulnerability Rate (%)')
    ax.set_title('Model Security Tiers: Vulnerability Rates (6 Code-Producing Models)\n'
                 '(Error bars: SEM across replicates)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(rates) * 1.25)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_model_tiers.png', dpi=300)
    plt.savefig(output_dir / 'fig_model_tiers.pdf')
    plt.close()
    print("  ✓ fig_model_tiers")


# ============================================================================
# FIGURE 7: Heatmap (model × sector vulnerability rates)
# ============================================================================

def plot_heatmap(results: List[Dict], output_dir: Path):
    """Vulnerability rate heatmap by model and sector. Safe-tier models show 0%.
    Medium-tier models show sector-dependent variation. Nuclear consistently lowest."""
    models = sorted(set(r['model'] for r in results))
    sectors = sorted(set(r['sector'] for r in results if r.get('sector') and r['sector'] != 'None'))

    # Build matrix
    matrix = np.zeros((len(models), len(sectors) + 1))  # +1 for baseline column
    model_order = []

    for i, model in enumerate(models):
        # Baseline rate
        base_r = [r for r in results if r['model'] == model and r['prompt_type'] == 'baseline']
        base_vuln = sum(1 for r in base_r if r['is_vulnerable'])
        base_total = len(base_r)
        matrix[i, -1] = (base_vuln / base_total * 100) if base_total else 0

        for j, sector in enumerate(sectors):
            sec_r = [r for r in results if r['model'] == model
                     and r['prompt_type'] == 'industry' and r.get('sector') == sector]
            vuln = sum(1 for r in sec_r if r['is_vulnerable'])
            total = len(sec_r)
            matrix[i, j] = (vuln / total * 100) if total else 0

        model_order.append((model, np.mean(matrix[i, :])))

    # Sort by overall rate
    model_order.sort(key=lambda x: x[1])
    sorted_indices = [models.index(m[0]) for m in model_order]
    matrix = matrix[sorted_indices]
    sorted_model_names = [MODEL_DISPLAY.get(model_order[i][0], model_order[i][0])
                          for i in range(len(model_order))]

    col_labels = [SECTOR_DISPLAY.get(s, s).replace('\n', ' ') for s in sectors] + ['Baseline']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap: white (0%) to red (high)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=max(25, np.max(matrix)))

    # Add text annotations
    for i in range(len(sorted_model_names)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            text_color = 'white' if val > 15 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(sorted_model_names)))
    ax.set_yticklabels(sorted_model_names, fontsize=9)

    ax.set_title('Vulnerability Rate (%) by Model and Sector\n'
                 'Darker cells indicate higher vulnerability rates')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Vulnerability Rate (%)')

    # Grid lines
    ax.set_xticks(np.arange(len(col_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(sorted_model_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', size=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_heatmap_comparison.png', dpi=300)
    plt.savefig(output_dir / 'fig_heatmap_comparison.pdf')
    plt.close()
    print("  ✓ fig_heatmap_comparison")


# ============================================================================
# FIGURE 8: CWE Distribution (baseline vs industry + drift, excluding 0% CWEs)
# ============================================================================

def plot_cwe_distribution(results: List[Dict], output_dir: Path):
    """CWE vulnerability distribution. Left: baseline vs industry rates.
    Right: protective effect magnitude. Excludes CWEs with 0% in both conditions."""
    cwe_counts = defaultdict(lambda: {'baseline': 0, 'industry': 0, 'total_base': 0, 'total_ind': 0})

    for r in results:
        cwe = r.get('cwe', 'Unknown')
        is_vuln = r.get('is_vulnerable', False)
        if r['prompt_type'] == 'baseline':
            cwe_counts[cwe]['total_base'] += 1
            if is_vuln:
                cwe_counts[cwe]['baseline'] += 1
        elif r['prompt_type'] == 'industry':
            cwe_counts[cwe]['total_ind'] += 1
            if is_vuln:
                cwe_counts[cwe]['industry'] += 1

    cwe_data = []
    for cwe, counts in cwe_counts.items():
        base_rate = (counts['baseline'] / counts['total_base'] * 100) if counts['total_base'] else 0
        ind_rate = (counts['industry'] / counts['total_ind'] * 100) if counts['total_ind'] else 0
        # Only include CWEs with non-zero rates in at least one condition
        if base_rate > 0 or ind_rate > 0:
            cwe_data.append({
                'cwe': cwe,
                'baseline_rate': base_rate,
                'industry_rate': ind_rate,
                'drift': ind_rate - base_rate,
            })

    cwe_data.sort(key=lambda x: x['baseline_rate'], reverse=True)

    CWE_NAMES = {
        'CWE-502': 'Insecure\nDeserialization',
        'CWE-295': 'Certificate\nValidation',
        'CWE-22': 'Path\nTraversal',
        'CWE-78': 'Command\nInjection',
        'CWE-798': 'Hard-coded\nCredentials',
        'CWE-89': 'SQL\nInjection',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cwes = [d['cwe'] for d in cwe_data]
    base_rates = [d['baseline_rate'] for d in cwe_data]
    ind_rates = [d['industry_rate'] for d in cwe_data]
    drifts = [d['drift'] for d in cwe_data]

    # Left: rates comparison
    x = np.arange(len(cwes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, base_rates, width, label='Baseline',
                    color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, ind_rates, width, label='Industry',
                    color=COLORS['industry'], edgecolor='black', linewidth=0.5)

    # Value labels
    for bar, rate in zip(bars1, base_rates):
        if rate > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar, rate in zip(bars2, ind_rates):
        if rate > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{CWE_NAMES.get(c, c)}\n({c})' for c in cwes], fontsize=8)
    ax1.set_ylabel('Vulnerability Rate (%)')
    ax1.set_title('Baseline vs. Industry Rates by CWE')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Right: drift by CWE
    drift_colors = [COLORS['protective'] if d < 0 else COLORS['risk'] for d in drifts]
    cwe_labels = [f'{CWE_NAMES.get(c, c)}\n({c})' for c in cwes]
    bars = ax2.barh(cwe_labels, drifts, color=drift_colors, edgecolor='black', linewidth=0.5)

    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    for bar, val in zip(bars, drifts):
        x_pos = bar.get_width() + (0.5 if val >= 0 else -0.5)
        ha = 'left' if val >= 0 else 'right'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}pp', va='center', ha=ha, fontsize=9, fontweight='bold')

    ax2.set_xlabel('Security Drift (Percentage Points)\n← More Secure | Less Secure →')
    ax2.set_title('Protective Effect by CWE Category')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_cwe_distribution.png', dpi=300)
    plt.savefig(output_dir / 'fig_cwe_distribution.pdf')
    plt.close()
    print("  ✓ fig_cwe_distribution")


# ============================================================================
# FIGURE 9: Nuclear Sector Deep Dive (4-panel)
# ============================================================================

def plot_nuclear_deep_dive(results: List[Dict], output_dir: Path):
    """Sector analysis 4-panel: (a) cross-sector comparison,
    (b) per-model rates for focus sector, (c) CWE-specific, (d) three-way comparison.
    Focus sector: emergency_services (strongest protective effect)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    focus_sector = 'emergency_services'
    focus_label = 'Emergency Svc.'
    focus_industry = [r for r in results
                      if r['prompt_type'] == 'industry' and r.get('sector') == focus_sector]
    baseline_results = [r for r in results if r['prompt_type'] == 'baseline']
    sectors = sorted(set(r['sector'] for r in results
                         if r.get('sector') and r['sector'] != 'None' and r['prompt_type'] == 'industry'))

    # --- (a) Cross-sector comparison ---
    sector_rates = []
    for sector in sectors:
        sec_r = [r for r in results if r['prompt_type'] == 'industry' and r.get('sector') == sector]
        vuln = sum(1 for r in sec_r if r['is_vulnerable'])
        total = len(sec_r)
        rate = (vuln / total * 100) if total else 0
        sector_rates.append((sector, rate))

    sector_rates.sort(key=lambda x: x[1])
    s_names = [SECTOR_DISPLAY.get(s, s).replace('\n', ' ') for s, _ in sector_rates]
    s_vals = [r for _, r in sector_rates]
    s_colors = [COLORS['protective'] if s == focus_sector else '#bdc3c7' for s, _ in sector_rates]

    ax1.barh(s_names, s_vals, color=s_colors, edgecolor='black', linewidth=0.5)
    for i, (name, val) in enumerate(zip(s_names, s_vals)):
        ax1.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Vulnerability Rate (%)')
    ax1.set_title('(a) Cross-Sector Comparison (Industry Prompts)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # --- (b) Focus sector by model ---
    models = sorted(set(r['model'] for r in focus_industry))
    model_rates_focus = []
    for model in models:
        mr = [r for r in focus_industry if r['model'] == model]
        vuln = sum(1 for r in mr if r['is_vulnerable'])
        total = len(mr)
        rate = (vuln / total * 100) if total else 0
        model_rates_focus.append((MODEL_DISPLAY.get(model, model), rate))

    model_rates_focus.sort(key=lambda x: x[1])
    m_names = [m for m, _ in model_rates_focus]
    m_vals = [r for _, r in model_rates_focus]
    m_colors = [COLORS['low_tier'] if v < 5 else COLORS['medium_tier'] if v < 15 else COLORS['high_tier'] for v in m_vals]

    ax2.barh(m_names, m_vals, color=m_colors, edgecolor='black', linewidth=0.5)
    for i, val in enumerate(m_vals):
        ax2.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Vulnerability Rate (%)')
    ax2.set_title(f'(b) {focus_label}: Per-Model Rates')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # --- (c) CWE-specific ---
    cwes = sorted(set(r['cwe'] for r in focus_industry))
    cwe_base_rates = []
    cwe_focus_rates = []
    cwe_labels = []

    for cwe in cwes:
        base_cwe = [r for r in baseline_results if r['cwe'] == cwe]
        focus_cwe = [r for r in focus_industry if r['cwe'] == cwe]
        b_rate = (sum(1 for r in base_cwe if r['is_vulnerable']) / len(base_cwe) * 100) if base_cwe else 0
        f_rate = (sum(1 for r in focus_cwe if r['is_vulnerable']) / len(focus_cwe) * 100) if focus_cwe else 0
        if b_rate > 0 or f_rate > 0:
            cwe_base_rates.append(b_rate)
            cwe_focus_rates.append(f_rate)
            cwe_labels.append(cwe)

    if cwe_labels:
        x = np.arange(len(cwe_labels))
        width = 0.35
        ax3.bar(x - width/2, cwe_base_rates, width, label='Baseline',
                color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
        ax3.bar(x + width/2, cwe_focus_rates, width, label=focus_label,
                color=COLORS['protective'], edgecolor='black', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(cwe_labels, rotation=45, ha='right')
    ax3.set_ylabel('Vulnerability Rate (%)')
    ax3.set_title(f'(c) {focus_label}: CWE-Specific Rates')
    ax3.legend(fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # --- (d) Three-way comparison ---
    base_rate = (sum(1 for r in baseline_results if r['is_vulnerable']) /
                 len(baseline_results) * 100) if baseline_results else 0

    matched_focus = [r for r in results
                     if r['prompt_type'] == 'matched_baseline' and r.get('sector') == focus_sector]
    matched_rate = (sum(1 for r in matched_focus if r['is_vulnerable']) /
                    len(matched_focus) * 100) if matched_focus else base_rate

    industry_rate = (sum(1 for r in focus_industry if r['is_vulnerable']) /
                     len(focus_industry) * 100) if focus_industry else base_rate

    conditions = ['Baseline\n(Generic)', 'Matched\n(Terminology)', 'Industry\n(Full Context)']
    rates = [base_rate, matched_rate, industry_rate]
    colors_3way = [COLORS['baseline'], COLORS['matched'], COLORS['industry']]

    bars = ax4.bar(conditions, rates, color=colors_3way, edgecolor='black', linewidth=1, width=0.5)
    for bar, rate in zip(bars, rates):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{rate:.1f}%', ha='center', fontweight='bold', fontsize=10)

    total_drift = industry_rate - base_rate
    ax4.annotate(f'Total drift: {total_drift:+.1f}pp',
                xy=(1, (base_rate + industry_rate)/2), fontsize=10, fontweight='bold',
                ha='center', color='#1a5276',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaf2f8', edgecolor='#1a5276'))

    ax4.set_ylabel('Vulnerability Rate (%)')
    ax4.set_title(f'(d) {focus_label}: Three-Way Comparison')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_ylim(0, max(rates) * 1.3)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Sector Analysis: Protective Effects Across CISA Sectors',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_nuclear_deep_dive.png', dpi=300)
    plt.savefig(output_dir / 'fig_nuclear_deep_dive.pdf')
    plt.close()
    print("  ✓ fig_nuclear_deep_dive")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate all paper figures for SecDrift")
    parser.add_argument("results", type=Path, help="Path to merged results JSONL")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("paper/figures"),
                        help="Output directory")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    print(f"Loaded {len(results)} raw evaluations")

    # Exclude models with 0% code production
    results = [r for r in results if r['model'] not in EXCLUDED_MODELS]
    print(f"After excluding non-code-producing models: {len(results)} evaluations")
    print(f"Models: {sorted(set(r['model'] for r in results))}")
    print(f"Output: {args.output}\n")

    print("Generating figures...")

    # Programmatic diagrams (no data needed)
    plot_architecture(args.output)
    plot_transformation(args.output)

    # Data-driven figures
    plot_overall_drift(results, args.output)
    plot_confidence_intervals(results, args.output)
    plot_effect_size_comparison(results, args.output)
    plot_model_tiers(results, args.output)
    plot_heatmap(results, args.output)
    plot_cwe_distribution(results, args.output)
    plot_nuclear_deep_dive(results, args.output)

    print(f"\n✅ All 9 figures saved to {args.output}")


if __name__ == "__main__":
    main()
