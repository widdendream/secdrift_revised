"""Consolidated visualization module for SecDrift benchmark results.

Generates publication-quality figures for research papers.
Supports basic (2-way), enhanced (3-way with replicates), and CWE-specific visualizations.
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
        # Compute vulnerability rate per replicate
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


# ============================================================================
# ENHANCED VISUALIZATIONS (3-way comparison with replicates)
# ============================================================================

def plot_three_way_sector_drift(results: List[Dict], output_dir: Path):
    """Three-way comparison by sector with error bars."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sectors = sorted(set(r['sector'] for r in results if r.get('sector')))
    
    # Compute stats for each prompt type per sector
    baseline_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'baseline'],
        ['prompt_type']
    )
    
    sector_data = {}
    for sector in sectors:
        matched_stats = compute_replicate_stats(
            [r for r in results if r['prompt_type'] == 'matched_baseline' and r.get('sector') == sector],
            ['sector', 'prompt_type']
        )
        industry_stats = compute_replicate_stats(
            [r for r in results if r['prompt_type'] == 'industry' and r.get('sector') == sector],
            ['sector', 'prompt_type']
        )
        
        baseline_key = ('baseline',)
        matched_key = (sector, 'matched_baseline')
        industry_key = (sector, 'industry')
        
        sector_data[sector] = {
            'baseline': baseline_stats.get(baseline_key, {'mean': 0, 'sem': 0}),
            'matched': matched_stats.get(matched_key, {'mean': 0, 'sem': 0}),
            'industry': industry_stats.get(industry_key, {'mean': 0, 'sem': 0}),
        }
    
    # Plot
    x = np.arange(len(sectors))
    width = 0.25
    
    baseline_means = [sector_data[s]['baseline']['mean'] for s in sectors]
    baseline_sems = [sector_data[s]['baseline']['sem'] for s in sectors]
    
    matched_means = [sector_data[s]['matched']['mean'] for s in sectors]
    matched_sems = [sector_data[s]['matched']['sem'] for s in sectors]
    
    industry_means = [sector_data[s]['industry']['mean'] for s in sectors]
    industry_sems = [sector_data[s]['industry']['sem'] for s in sectors]
    
    bars1 = ax.bar(x - width, baseline_means, width, yerr=baseline_sems,
                   label='Baseline (generic)', color='#3498db', 
                   edgecolor='black', linewidth=0.5, capsize=3)
    bars2 = ax.bar(x, matched_means, width, yerr=matched_sems,
                   label='Matched Baseline (terminology)', color='#f39c12',
                   edgecolor='black', linewidth=0.5, capsize=3)
    bars3 = ax.bar(x + width, industry_means, width, yerr=industry_sems,
                   label='Industry (terminology + context)', color='#2ecc71',
                   edgecolor='black', linewidth=0.5, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in sectors], fontsize=9)
    ax.set_ylabel('Vulnerability Rate (%)')
    ax.set_xlabel('CISA Critical Infrastructure Sector')
    ax.set_title('Three-Way Comparison: Vulnerability Rates by Sector\n(Error bars show SEM across 5 replicates)')
    ax.legend(loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(max(baseline_means), max(matched_means), max(industry_means)) * 1.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_threeway_sector.pdf')
    plt.savefig(output_dir / 'fig1_threeway_sector.png')
    plt.close()
    print("✓ Figure 1: Three-way sector comparison")


def plot_drift_decomposition(results: List[Dict], output_dir: Path):
    """Drift decomposition showing terminology vs context effects."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sectors = sorted(set(r['sector'] for r in results if r.get('sector')))
    
    # Get baseline rate (same for all sectors)
    baseline_stats = compute_replicate_stats(
        [r for r in results if r['prompt_type'] == 'baseline'],
        ['prompt_type']
    )
    baseline_mean = baseline_stats[('baseline',)]['mean']
    
    terminology_drifts = []
    context_drifts = []
    total_drifts = []
    
    for sector in sectors:
        matched_stats = compute_replicate_stats(
            [r for r in results if r['prompt_type'] == 'matched_baseline' and r.get('sector') == sector],
            ['sector', 'prompt_type']
        )
        industry_stats = compute_replicate_stats(
            [r for r in results if r['prompt_type'] == 'industry' and r.get('sector') == sector],
            ['sector', 'prompt_type']
        )
        
        matched_mean = matched_stats.get((sector, 'matched_baseline'), {'mean': baseline_mean})['mean']
        industry_mean = industry_stats.get((sector, 'industry'), {'mean': baseline_mean})['mean']
        
        term_drift = matched_mean - baseline_mean
        context_drift = industry_mean - matched_mean
        total_drift = industry_mean - baseline_mean
        
        terminology_drifts.append(term_drift)
        context_drifts.append(context_drift)
        total_drifts.append(total_drift)
    
    x = np.arange(len(sectors))
    width = 0.6
    
    # Stacked bar chart
    bars1 = ax.bar(x, terminology_drifts, width, label='Terminology Effect',
                   color='#f39c12', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, context_drifts, width, bottom=terminology_drifts,
                   label='Context Effect', color='#9b59b6', edgecolor='black', linewidth=0.5)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Add total drift labels
    for i, (term, context, total) in enumerate(zip(terminology_drifts, context_drifts, total_drifts)):
        y_pos = term + context
        ax.text(i, y_pos + (0.5 if y_pos > 0 else -0.5), f'{total:+.1f}pp',
               ha='center', va='bottom' if y_pos > 0 else 'top',
               fontsize=8, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in sectors], fontsize=9)
    ax.set_ylabel('Drift (Percentage Points)')
    ax.set_xlabel('CISA Critical Infrastructure Sector')
    ax.set_title('Drift Decomposition: Terminology vs Context Effects\n(Negative = More Secure)')
    ax.legend(loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_drift_decomposition.pdf')
    plt.savefig(output_dir / 'fig2_drift_decomposition.png')
    plt.close()
    print("✓ Figure 2: Drift decomposition")


def plot_cwe_comparison(results: List[Dict], output_dir: Path):
    """CWE-specific three-way comparison."""
    # Compute CWE stats
    cwe_data = defaultdict(lambda: defaultdict(lambda: {'vuln': 0, 'total': 0}))
    
    for r in results:
        cwe = r.get('cwe', 'Unknown')
        prompt_type = r.get('prompt_type', 'unknown')
        is_vuln = 1 if r.get('is_vulnerable') else 0
        
        cwe_data[cwe][prompt_type]['vuln'] += is_vuln
        cwe_data[cwe][prompt_type]['total'] += 1
    
    # Compute rates
    stats = {}
    for cwe, prompt_types in cwe_data.items():
        stats[cwe] = {}
        for prompt_type, counts in prompt_types.items():
            rate = (counts['vuln'] / counts['total'] * 100) if counts['total'] > 0 else 0
            stats[cwe][prompt_type] = rate
    
    # Sort CWEs by baseline rate
    cwes = sorted(stats.keys(), key=lambda c: stats[c].get('baseline', 0), reverse=True)
    
    # CWE display names
    cwe_names = {
        'CWE-89': 'SQL Injection\n(CWE-89)',
        'CWE-78': 'Command Injection\n(CWE-78)',
        'CWE-502': 'Insecure\nDeserialization\n(CWE-502)',
        'CWE-22': 'Path Traversal\n(CWE-22)',
        'CWE-79': 'Cross-Site\nScripting\n(CWE-79)',
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(cwes))
    width = 0.25
    
    baseline_rates = [stats[c].get('baseline', 0) for c in cwes]
    matched_rates = [stats[c].get('matched_baseline', 0) for c in cwes]
    industry_rates = [stats[c].get('industry', 0) for c in cwes]
    
    bars1 = ax.bar(x - width, baseline_rates, width, 
                   label='Baseline (generic)', color='#3498db',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, matched_rates, width,
                   label='Matched Baseline (terminology)', color='#f39c12',
                   edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, industry_rates, width,
                   label='Industry (terminology + context)', color='#2ecc71',
                   edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    def add_labels(bars, rates):
        for bar, rate in zip(bars, rates):
            if rate > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1, baseline_rates)
    add_labels(bars2, matched_rates)
    add_labels(bars3, industry_rates)
    
    ax.set_xticks(x)
    ax.set_xticklabels([cwe_names.get(c, c) for c in cwes], fontsize=9)
    ax.set_ylabel('Vulnerability Rate (%)')
    ax.set_xlabel('CWE Category')
    ax.set_title('Vulnerability Rates by CWE Category: Three-Way Comparison')
    ax.legend(loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(max(baseline_rates), max(matched_rates), max(industry_rates)) * 1.2)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_cwe_comparison.pdf')
    plt.savefig(output_dir / 'fig6_cwe_comparison.png')
    plt.close()
    print("✓ Figure 6: CWE comparison")


def generate_enhanced_figures(results_path: Path, output_dir: Path):
    """Generate all enhanced figures with replicates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)
    print(f"Loaded {len(results)} evaluations")
    
    # Check data
    prompt_types = set(r['prompt_type'] for r in results)
    print(f"Prompt types: {prompt_types}")
    
    if 'matched_baseline' not in prompt_types:
        print("WARNING: No matched_baseline data found. Some figures will be incomplete.")
    
    print("\nGenerating enhanced figures...")
    plot_three_way_sector_drift(results, output_dir)
    plot_drift_decomposition(results, output_dir)
    plot_cwe_comparison(results, output_dir)
    
    print(f"\n✅ Enhanced figures saved to {output_dir}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SecDrift visualizations")
    parser.add_argument("results", type=Path, help="Path to benchmark results JSONL")
    parser.add_argument("-o", "--output", type=Path, default=Path("figures"),
                       help="Output directory for figures")
    parser.add_argument("--mode", choices=['basic', 'enhanced', 'cwe', 'all'],
                       default='enhanced',
                       help="Visualization mode (default: enhanced)")
    
    args = parser.parse_args()
    
    if args.mode in ['enhanced', 'all']:
        generate_enhanced_figures(args.results, args.output)
