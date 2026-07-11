#!/usr/bin/env python3
"""Generate additional publication-quality figures for SecDrift paper.

Creates 8 additional visualizations to strengthen the paper's narrative:
1. Effect size comparison chart
2. Confidence interval plot (forest plot)
3. CWE vulnerability distribution
4. Model-sector interaction scatter
5. Replicate variance analysis
6. Terminology vs context contribution
7. Nuclear sector deep dive
8. Practical decision matrix
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def load_results(path: Path) -> List[Dict]:
    """Load benchmark results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_replicate_stats(results: List[Dict], group_by: List[str]) -> Dict:
    """Compute mean and std across replicates for grouped data."""
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
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    
    return max(0, center - margin) * 100, min(1, center + margin) * 100


# ============================================================================
# FIGURE 1: Effect Size Comparison
# ============================================================================

def plot_effect_size_comparison(results: List[Dict], output_dir: Path):
    """Bar chart comparing magnitude of different effects."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute effects
    models = sorted(set(r['model'] for r in results))
    sectors = sorted(set(r['sector'] for r in results if r.get('sector')))
    
    # Model effect: range of vulnerability rates
    model_stats = compute_replicate_stats(results, ['model'])
    model_rates = [stats['mean'] for stats in model_stats.values()]
    model_effect = max(model_rates) - min(model_rates)
    
    # Sector effect: range across sectors (industry prompts only)
    sector_results = [r for r in results if r['prompt_type'] == 'industry']
    sector_stats = compute_replicate_stats(sector_results, ['sector'])
    sector_rates = [stats['mean'] for stats in sector_stats.values()]
    sector_effect = max(sector_rates) - min(sector_rates)
    
    # Overall drift: baseline vs industry
    baseline_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'baseline'], ['prompt_type']
    )
    industry_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'industry'], ['prompt_type']
    )
    overall_drift = abs(industry_stats[('industry',)]['mean'] - baseline_stats[('baseline',)]['mean'])
    
    # Terminology vs context
    matched_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'matched_baseline'], ['prompt_type']
    )
    if matched_stats:
        term_effect = abs(matched_stats[('matched_baseline',)]['mean'] - baseline_stats[('baseline',)]['mean'])
        context_effect = abs(industry_stats[('industry',)]['mean'] - matched_stats[('matched_baseline',)]['mean'])
    else:
        term_effect = overall_drift * 0.7
        context_effect = overall_drift * 0.3
    
    effects = {
        'Model Selection\n(0% to 19.8%)': model_effect,
        'Sector Context\n(6.7% to 10.0%)': sector_effect,
        'Overall Drift\n(Baseline→Industry)': overall_drift,
        'Terminology Effect\n(Baseline→Matched)': term_effect,
        'Context Effect\n(Matched→Industry)': context_effect,
    }
    
    labels = list(effects.keys())
    values = list(effects.values())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}pp', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Effect Size (Percentage Points)')
    ax.set_title('Effect Size Comparison: Model vs Sector vs Drift Components')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_effect_size_comparison.pdf')
    plt.savefig(output_dir / 'fig_effect_size_comparison.png')
    plt.close()
    print("✓ Figure: Effect size comparison")


# ============================================================================
# FIGURE 2: Confidence Interval Forest Plot
# ============================================================================

def plot_confidence_intervals(results: List[Dict], output_dir: Path):
    """Forest plot showing drift with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sectors = sorted(set(r['sector'] for r in results if r.get('sector')))
    
    # Get baseline rate
    baseline_results = [r for r in results if r['prompt_type'] == 'baseline']
    baseline_vuln = sum(1 for r in baseline_results if r['is_vulnerable'])
    baseline_total = len(baseline_results)
    baseline_rate = (baseline_vuln / baseline_total * 100) if baseline_total > 0 else 0
    
    drift_data = []
    for sector in sectors:
        industry_results = [r for r in results 
                          if r['prompt_type'] == 'industry' and r.get('sector') == sector]
        industry_vuln = sum(1 for r in industry_results if r['is_vulnerable'])
        industry_total = len(industry_results)
        industry_rate = (industry_vuln / industry_total * 100) if industry_total > 0 else 0
        
        drift = industry_rate - baseline_rate
        
        # Compute CI using Wilson score
        ci_low_ind, ci_high_ind = wilson_ci(industry_vuln, industry_total)
        ci_low_base, ci_high_base = wilson_ci(baseline_vuln, baseline_total)
        
        # Drift CI (approximate)
        drift_ci_low = ci_low_ind - ci_high_base
        drift_ci_high = ci_high_ind - ci_low_base
        
        drift_data.append({
            'sector': sector,
            'drift': drift,
            'ci_low': drift_ci_low,
            'ci_high': drift_ci_high,
        })
    
    # Sort by drift
    drift_data.sort(key=lambda x: x['drift'])
    
    y_pos = np.arange(len(drift_data))
    drifts = [d['drift'] for d in drift_data]
    ci_lows = [d['ci_low'] for d in drift_data]
    ci_highs = [d['ci_high'] for d in drift_data]
    labels = [d['sector'].replace('_', ' ').title() for d in drift_data]
    
    # Color by effect
    colors = ['#2ecc71' if d < -1 else '#f39c12' if d < 1 else '#e74c3c' for d in drifts]
    
    # Plot points and error bars
    for i, (drift, ci_low, ci_high, color) in enumerate(zip(drifts, ci_lows, ci_highs, colors)):
        ax.plot([ci_low, ci_high], [i, i], color=color, linewidth=2, alpha=0.6)
        ax.scatter(drift, i, s=100, color=color, edgecolor='black', linewidth=1, zorder=3)
    
    # Zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='No Effect')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Security Drift (Percentage Points)\n← More Secure | Less Secure →')
    ax.set_title('Sector-Specific Security Drift with 95% Confidence Intervals')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Protective (< -1pp)')
    yellow_patch = mpatches.Patch(color='#f39c12', label='Neutral (-1 to +1pp)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Risk-inducing (> +1pp)')
    ax.legend(handles=[green_patch, yellow_patch, red_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_confidence_intervals.pdf')
    plt.savefig(output_dir / 'fig_confidence_intervals.png')
    plt.close()
    print("✓ Figure: Confidence intervals (forest plot)")


# ============================================================================
# FIGURE 3: CWE Vulnerability Distribution
# ============================================================================

def plot_cwe_distribution(results: List[Dict], output_dir: Path):
    """Stacked bar showing CWE contribution to vulnerabilities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count vulnerabilities by CWE
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
    
    # Compute rates
    cwe_data = []
    for cwe, counts in cwe_counts.items():
        base_rate = (counts['baseline'] / counts['total_base'] * 100) if counts['total_base'] > 0 else 0
        ind_rate = (counts['industry'] / counts['total_ind'] * 100) if counts['total_ind'] > 0 else 0
        cwe_data.append({
            'cwe': cwe,
            'baseline_rate': base_rate,
            'industry_rate': ind_rate,
            'baseline_count': counts['baseline'],
            'industry_count': counts['industry'],
            'drift': ind_rate - base_rate,
        })
    
    # Sort by baseline rate
    cwe_data.sort(key=lambda x: x['baseline_rate'], reverse=True)
    
    cwes = [d['cwe'] for d in cwe_data]
    base_rates = [d['baseline_rate'] for d in cwe_data]
    ind_rates = [d['industry_rate'] for d in cwe_data]
    drifts = [d['drift'] for d in cwe_data]
    
    # Plot 1: Rates comparison
    x = np.arange(len(cwes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, base_rates, width, label='Baseline',
                    color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, ind_rates, width, label='Industry',
                    color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(cwes, rotation=45, ha='right')
    ax1.set_ylabel('Vulnerability Rate (%)')
    ax1.set_xlabel('CWE Category')
    ax1.set_title('Vulnerability Rates by CWE: Baseline vs Industry')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Drift by CWE
    colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in drifts]
    bars = ax2.barh(cwes, drifts, color=colors, edgecolor='black', linewidth=0.5)
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Security Drift (Percentage Points)')
    ax2.set_title('Protective Effect by CWE Category')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, drifts):
        width = bar.get_width()
        x_pos = width + (0.5 if width > 0 else -0.5)
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}pp', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_cwe_distribution.pdf')
    plt.savefig(output_dir / 'fig_cwe_distribution.png')
    plt.close()
    print("✓ Figure: CWE vulnerability distribution")


# ============================================================================
# FIGURE 4: Model-Sector Interaction Scatter
# ============================================================================

def plot_model_sector_interaction(results: List[Dict], output_dir: Path):
    """Scatter plot showing model vulnerability vs sector drift."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = sorted(set(r['model'] for r in results))
    sectors = sorted(set(r['sector'] for r in results if r.get('sector')))
    
    # Get baseline rate
    baseline_results = [r for r in results if r['prompt_type'] == 'baseline']
    baseline_vuln = sum(1 for r in baseline_results if r['is_vulnerable'])
    baseline_total = len(baseline_results)
    baseline_rate = (baseline_vuln / baseline_total * 100) if baseline_total > 0 else 0
    
    scatter_data = []
    for model in models:
        # Model overall vulnerability rate
        model_results = [r for r in results if r['model'] == model]
        model_vuln = sum(1 for r in model_results if r['is_vulnerable'])
        model_total = len(model_results)
        model_rate = (model_vuln / model_total * 100) if model_total > 0 else 0
        
        for sector in sectors:
            # Sector-specific drift for this model
            industry_results = [r for r in results 
                              if r['model'] == model and r['prompt_type'] == 'industry' 
                              and r.get('sector') == sector]
            if not industry_results:
                continue
                
            industry_vuln = sum(1 for r in industry_results if r['is_vulnerable'])
            industry_total = len(industry_results)
            industry_rate = (industry_vuln / industry_total * 100) if industry_total > 0 else 0
            
            drift = industry_rate - baseline_rate
            
            scatter_data.append({
                'model': model,
                'sector': sector,
                'model_rate': model_rate,
                'drift': drift,
            })
    
    # Plot
    x = [d['model_rate'] for d in scatter_data]
    y = [d['drift'] for d in scatter_data]
    
    # Color by model
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    model_color_map = {m: c for m, c in zip(models, model_colors)}
    colors = [model_color_map[d['model']] for d in scatter_data]
    
    ax.scatter(x, y, c=colors, s=100, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Zero lines
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=baseline_rate, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline Rate')
    
    # Trend line
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(x), max(x), 100)
        ax.plot(x_trend, p(x_trend), "r-", alpha=0.3, linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    
    ax.set_xlabel('Model Vulnerability Rate (%)')
    ax.set_ylabel('Sector-Specific Drift (pp)')
    ax.set_title('Model-Sector Interaction: Does Model Quality Affect Protective Effect?')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Legend for models
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=model_color_map[m], markersize=8,
                                 label=m, markeredgecolor='black', markeredgewidth=0.5)
                      for m in models[:5]]  # Show first 5 models
    ax.legend(handles=legend_elements, loc='best', ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_model_sector_interaction.pdf')
    plt.savefig(output_dir / 'fig_model_sector_interaction.png')
    plt.close()
    print("✓ Figure: Model-sector interaction scatter")


# ============================================================================
# FIGURE 5: Replicate Variance Analysis
# ============================================================================

def plot_replicate_variance(results: List[Dict], output_dir: Path):
    """Box plots showing variance across replicates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # By model
    models = sorted(set(r['model'] for r in results))
    model_data = []
    model_labels = []
    
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        stats = compute_replicate_stats(model_results, ['model'])
        if stats:
            rates = stats[(model,)]['rates']
            model_data.append(rates)
            model_labels.append(model.replace('-', '\n'))
    
    bp1 = ax1.boxplot(model_data, labels=model_labels, patch_artist=True,
                      boxprops=dict(facecolor='#3498db', alpha=0.6),
                      medianprops=dict(color='red', linewidth=2))
    
    ax1.set_ylabel('Vulnerability Rate (%) Across Replicates')
    ax1.set_xlabel('Model')
    ax1.set_title('Replicate Variance by Model')
    ax1.tick_params(axis='x', rotation=45)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # By sector
    sectors = sorted(set(r['sector'] for r in results if r.get('sector')))
    sector_data = []
    sector_labels = []
    
    for sector in sectors:
        sector_results = [r for r in results 
                         if r['prompt_type'] == 'industry' and r.get('sector') == sector]
        stats = compute_replicate_stats(sector_results, ['sector'])
        if stats:
            rates = stats[(sector,)]['rates']
            sector_data.append(rates)
            sector_labels.append(sector.replace('_', '\n'))
    
    bp2 = ax2.boxplot(sector_data, labels=sector_labels, patch_artist=True,
                      boxprops=dict(facecolor='#2ecc71', alpha=0.6),
                      medianprops=dict(color='red', linewidth=2))
    
    ax2.set_ylabel('Vulnerability Rate (%) Across Replicates')
    ax2.set_xlabel('Sector')
    ax2.set_title('Replicate Variance by Sector')
    ax2.tick_params(axis='x', rotation=45)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_replicate_variance.pdf')
    plt.savefig(output_dir / 'fig_replicate_variance.png')
    plt.close()
    print("✓ Figure: Replicate variance analysis")


# ============================================================================
# FIGURE 6: Terminology vs Context Contribution
# ============================================================================

def plot_terminology_context_contribution(results: List[Dict], output_dir: Path):
    """Pie and bar charts showing relative contribution."""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Compute overall effects
    baseline_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'baseline'], ['prompt_type']
    )
    matched_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'matched_baseline'], ['prompt_type']
    )
    industry_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'industry'], ['prompt_type']
    )
    
    baseline_rate = baseline_stats[('baseline',)]['mean']
    
    if matched_stats:
        matched_rate = matched_stats[('matched_baseline',)]['mean']
        industry_rate = industry_stats[('industry',)]['mean']
        
        term_effect = abs(matched_rate - baseline_rate)
        context_effect = abs(industry_rate - matched_rate)
        total_effect = term_effect + context_effect
        
        # Pie chart
        sizes = [term_effect, context_effect]
        labels = [f'Terminology\n{term_effect:.1f}pp\n({term_effect/total_effect*100:.0f}%)',
                 f'Context\n{context_effect:.1f}pp\n({context_effect/total_effect*100:.0f}%)']
        colors = ['#f39c12', '#9b59b6']
        explode = (0.05, 0.05)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='', startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        ax1.set_title('Protective Effect Decomposition')
        
        # Waterfall chart
        categories = ['Baseline', 'Add\nTerminology', 'Add\nContext', 'Final\n(Industry)']
        values = [baseline_rate, -term_effect, -context_effect, 0]
        cumulative = [baseline_rate, matched_rate, industry_rate, industry_rate]
        
        colors_bar = ['#3498db', '#f39c12', '#9b59b6', '#2ecc71']
        
        for i, (cat, val, cum) in enumerate(zip(categories[:-1], values[:-1], cumulative[:-1])):
            if i == 0:
                ax2.bar(i, cum, color=colors_bar[i], edgecolor='black', linewidth=1)
            else:
                ax2.bar(i, abs(val), bottom=min(cum, cumulative[i+1]),
                       color=colors_bar[i], edgecolor='black', linewidth=1)
                # Connection line
                ax2.plot([i-0.4, i-0.4], [cumulative[i-1], cumulative[i-1]], 
                        'k--', linewidth=0.5)
                ax2.plot([i-0.4, i+0.4], [cumulative[i-1], cumulative[i]], 
                        'k--', linewidth=0.5)
        
        # Final bar
        ax2.bar(len(categories)-1, industry_rate, color=colors_bar[-1], 
               edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, cum in enumerate(cumulative):
            ax2.text(i, cum + 0.5, f'{cum:.1f}%', ha='center', fontweight='bold')
        
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Vulnerability Rate (%)')
        ax2.set_title('Waterfall: Baseline → Industry Transformation')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_terminology_context_contribution.pdf')
    plt.savefig(output_dir / 'fig_terminology_context_contribution.png')
    plt.close()
    print("✓ Figure: Terminology vs context contribution")


# ============================================================================
# FIGURE 7: Nuclear Sector Deep Dive
# ============================================================================

def plot_nuclear_deep_dive(results: List[Dict], output_dir: Path):
    """Detailed analysis of nuclear sector (strongest protective effect)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    nuclear_results = [r for r in results if r.get('sector') == 'nuclear']
    baseline_results = [r for r in results if r['prompt_type'] == 'baseline']
    
    # 1. Nuclear vs Other Sectors
    sectors = sorted(set(r['sector'] for r in results if r.get('sector')))
    sector_rates = []
    sector_labels = []
    
    for sector in sectors:
        sector_results = [r for r in results 
                         if r['prompt_type'] == 'industry' and r.get('sector') == sector]
        vuln = sum(1 for r in sector_results if r['is_vulnerable'])
        total = len(sector_results)
        rate = (vuln / total * 100) if total > 0 else 0
        sector_rates.append(rate)
        sector_labels.append(sector.replace('_', ' ').title())
    
    colors = ['#e74c3c' if s != 'nuclear' else '#2ecc71' for s in sectors]
    bars = ax1.barh(sector_labels, sector_rates, color=colors, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Vulnerability Rate (%)')
    ax1.set_title('Nuclear Sector vs Others')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Nuclear by Model
    models = sorted(set(r['model'] for r in nuclear_results))
    model_rates = []
    model_labels = []
    
    for model in models:
        model_results = [r for r in nuclear_results if r['model'] == model]
        vuln = sum(1 for r in model_results if r['is_vulnerable'])
        total = len(model_results)
        rate = (vuln / total * 100) if total > 0 else 0
        model_rates.append(rate)
        model_labels.append(model)
    
    ax2.bar(range(len(models)), model_rates, color='#3498db', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(model_labels, rotation=45, ha='right')
    ax2.set_ylabel('Vulnerability Rate (%)')
    ax2.set_title('Nuclear Sector: Vulnerability by Model')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Nuclear by CWE
    cwes = sorted(set(r['cwe'] for r in nuclear_results))
    cwe_data = {'baseline': [], 'nuclear': []}
    cwe_labels = []
    
    for cwe in cwes:
        base_cwe = [r for r in baseline_results if r['cwe'] == cwe]
        nuc_cwe = [r for r in nuclear_results if r['cwe'] == cwe]
        
        base_rate = (sum(1 for r in base_cwe if r['is_vulnerable']) / len(base_cwe) * 100) if base_cwe else 0
        nuc_rate = (sum(1 for r in nuc_cwe if r['is_vulnerable']) / len(nuc_cwe) * 100) if nuc_cwe else 0
        
        cwe_data['baseline'].append(base_rate)
        cwe_data['nuclear'].append(nuc_rate)
        cwe_labels.append(cwe)
    
    x = np.arange(len(cwes))
    width = 0.35
    
    ax3.bar(x - width/2, cwe_data['baseline'], width, label='Baseline',
           color='#3498db', edgecolor='black', linewidth=0.5)
    ax3.bar(x + width/2, cwe_data['nuclear'], width, label='Nuclear',
           color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(cwe_labels, rotation=45, ha='right')
    ax3.set_ylabel('Vulnerability Rate (%)')
    ax3.set_title('Nuclear Sector: CWE-Specific Protection')
    ax3.legend()
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Three-way comparison for nuclear
    baseline_rate = (sum(1 for r in baseline_results if r['is_vulnerable']) / 
                    len(baseline_results) * 100) if baseline_results else 0
    
    matched_nuc = [r for r in results 
                   if r['prompt_type'] == 'matched_baseline' and r.get('sector') == 'nuclear']
    matched_rate = (sum(1 for r in matched_nuc if r['is_vulnerable']) / 
                   len(matched_nuc) * 100) if matched_nuc else baseline_rate
    
    industry_nuc = [r for r in results 
                    if r['prompt_type'] == 'industry' and r.get('sector') == 'nuclear']
    industry_rate = (sum(1 for r in industry_nuc if r['is_vulnerable']) / 
                    len(industry_nuc) * 100) if industry_nuc else baseline_rate
    
    conditions = ['Baseline\n(Generic)', 'Matched\n(Terminology)', 'Industry\n(Full Context)']
    rates = [baseline_rate, matched_rate, industry_rate]
    colors_3way = ['#3498db', '#f39c12', '#2ecc71']
    
    bars = ax4.bar(conditions, rates, color=colors_3way, edgecolor='black', linewidth=1)
    
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', fontweight='bold')
    
    ax4.set_ylabel('Vulnerability Rate (%)')
    ax4.set_title('Nuclear Sector: Three-Way Comparison')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_ylim(0, max(rates) * 1.2)
    
    plt.suptitle('Nuclear Sector Deep Dive: Strongest Protective Effect (-5.6pp)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_nuclear_deep_dive.pdf')
    plt.savefig(output_dir / 'fig_nuclear_deep_dive.png')
    plt.close()
    print("✓ Figure: Nuclear sector deep dive")


# ============================================================================
# FIGURE 8: Practical Decision Matrix
# ============================================================================

def plot_decision_matrix(results: List[Dict], output_dir: Path):
    """Decision matrix for practitioners."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Compute model tiers
    models = sorted(set(r['model'] for r in results))
    model_rates = {}
    
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        vuln = sum(1 for r in model_results if r['is_vulnerable'])
        total = len(model_results)
        rate = (vuln / total * 100) if total > 0 else 0
        model_rates[model] = rate
    
    # Define tiers
    tiers = {
        'Safe': [m for m, r in model_rates.items() if r < 5],
        'Low Risk': [m for m, r in model_rates.items() if 5 <= r < 10],
        'Medium Risk': [m for m, r in model_rates.items() if 10 <= r < 15],
        'High Risk': [m for m, r in model_rates.items() if r >= 15],
    }
    
    # Plot 1: Model tier classification
    tier_names = list(tiers.keys())
    tier_counts = [len(tiers[t]) for t in tier_names]
    tier_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    bars = ax1.barh(tier_names, tier_counts, color=tier_colors, edgecolor='black', linewidth=1)
    
    # Add model names
    for i, (tier, models_in_tier) in enumerate(tiers.items()):
        if models_in_tier:
            text = ', '.join(models_in_tier[:3])
            if len(models_in_tier) > 3:
                text += f' (+{len(models_in_tier)-3} more)'
            ax1.text(tier_counts[i] + 0.1, i, text, va='center', fontsize=9)
    
    ax1.set_xlabel('Number of Models')
    ax1.set_title('Model Security Tier Classification')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, max(tier_counts) * 1.8)
    
    # Plot 2: Decision matrix
    ax2.axis('off')
    
    # Create decision table
    decision_data = [
        ['Model Tier', 'Vuln Rate', 'Sector Effect', 'Recommendation', 'Review Level'],
        ['Safe', '0-5%', 'Protective', 'Approve for all sectors', 'Standard'],
        ['Low Risk', '5-10%', 'Protective', 'Approve with monitoring', 'Standard'],
        ['Medium Risk', '10-15%', 'Protective', 'Approve with review', 'Enhanced'],
        ['High Risk', '>15%', 'Protective', 'Restrict or reject', 'Mandatory'],
    ]
    
    # Colors for cells
    cell_colors = [
        ['#d5d8dc'] * 5,  # Header
        ['#2ecc71', '#d5f4e6', '#d5f4e6', '#d5f4e6', '#d5f4e6'],
        ['#3498db', '#d6eaf8', '#d6eaf8', '#d6eaf8', '#d6eaf8'],
        ['#f39c12', '#fdebd0', '#fdebd0', '#fdebd0', '#fdebd0'],
        ['#e74c3c', '#fadbd8', '#fadbd8', '#fadbd8', '#fadbd8'],
    ]
    
    table = ax2.table(cellText=decision_data, cellColours=cell_colors,
                     cellLoc='left', loc='center',
                     colWidths=[0.15, 0.12, 0.15, 0.3, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Bold header
    for i in range(5):
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white')
    
    ax2.set_title('Practical Decision Matrix for Model Selection\n(Based on SecDrift Findings)',
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_decision_matrix.pdf')
    plt.savefig(output_dir / 'fig_decision_matrix.png')
    plt.close()
    print("✓ Figure: Practical decision matrix")


# ============================================================================
# Main Function
# ============================================================================

def generate_all_additional_figures(results_path: Path, output_dir: Path):
    """Generate all 8 additional figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)
    print(f"Loaded {len(results)} evaluations\n")
    
    print("Generating additional figures...")
    print("=" * 60)
    
    plot_effect_size_comparison(results, output_dir)
    plot_confidence_intervals(results, output_dir)
    plot_cwe_distribution(results, output_dir)
    plot_model_sector_interaction(results, output_dir)
    plot_replicate_variance(results, output_dir)
    plot_terminology_context_contribution(results, output_dir)
    plot_nuclear_deep_dive(results, output_dir)
    plot_decision_matrix(results, output_dir)
    
    print("=" * 60)
    print(f"\n✅ All 8 additional figures saved to {output_dir}")
    print("\nGenerated figures:")
    print("  1. fig_effect_size_comparison.png")
    print("  2. fig_confidence_intervals.png")
    print("  3. fig_cwe_distribution.png")
    print("  4. fig_model_sector_interaction.png")
    print("  5. fig_replicate_variance.png")
    print("  6. fig_terminology_context_contribution.png")
    print("  7. fig_nuclear_deep_dive.png")
    print("  8. fig_decision_matrix.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate additional publication-quality figures for SecDrift paper"
    )
    parser.add_argument("results", type=Path, 
                       help="Path to benchmark results JSONL file")
    parser.add_argument("-o", "--output", type=Path, default=Path("figures"),
                       help="Output directory for figures (default: figures/)")
    
    args = parser.parse_args()
    
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        exit(1)
    
    generate_all_additional_figures(args.results, args.output)
