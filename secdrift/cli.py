"""SecDrift Command Line Interface.

Provides unified CLI for all SecDrift operations.

Usage:
    secdrift run --group paper --replicates 5
    secdrift analyze results/benchmark.jsonl
    secdrift list-models
    secdrift validate-sector config/my_sector.yaml
"""

import argparse
import sys
from pathlib import Path


def cmd_run(args):
    """Run benchmark."""
    from secdrift.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        models=args.models,
        group=args.group,
        parallel_workers=args.workers,
        replicates=args.replicates,
    )

    results = runner.run_benchmark(
        output_file=args.output,
        scenarios=args.scenarios,
        sectors=args.sectors,
    )

    print(f"\nBenchmark complete!")
    print(f"  Total evaluations: {len(results)}")
    print(f"  Vulnerable: {sum(1 for r in results if r.is_vulnerable)}")
    print(f"\nNext: secdrift analyze {args.output or 'results/<timestamp>.jsonl'}")


def cmd_analyze(args):
    """Analyze results."""
    from secdrift.analysis import (
        analyze_benchmark_results,
        format_analysis_report,
        export_results_json,
    )

    results = analyze_benchmark_results(Path(args.results_file))
    report = format_analysis_report(results)
    print(report)
    if args.output:
        out = Path(args.output)
        if out.suffix == ".json":
            export_results_json(results, out)
        else:
            out.write_text(report)
        print(f"\nWrote {out}")


def cmd_list_models(args):
    """List available models."""
    from secdrift.models import list_models
    from secdrift.models.registry import get_config

    config = get_config()

    print("Available Models:")
    print("-" * 60)
    for name in list_models():
        model_config = config.get_model_config(name)
        enabled = "✓" if model_config.get("enabled", False) else " "
        vendor = model_config.get("vendor", "Unknown")
        display = model_config.get("display_name", name)
        print(f"  [{enabled}] {name:<25} {vendor:<12} {display}")

    print("\nModel Groups:")
    print("-" * 60)
    for group_name, models in config.groups.items():
        print(f"  {group_name}: {', '.join(models)}")


def cmd_list_sectors(args):
    """List available sectors."""
    from secdrift.transformer import list_sectors, get_sector_config

    print("Available Sectors:")
    print("-" * 60)
    for sector_id in list_sectors():
        config = get_sector_config(sector_id)
        scenarios = len(config.context_templates)
        terms = len(config.terminology)
        print(f"  {sector_id:<25} {config.name:<30} ({scenarios} scenarios, {terms} terms)")


def cmd_validate_sector(args):
    """Validate a sector configuration file."""
    from secdrift.transformer import load_sector_from_yaml, validate_transformation

    try:
        config = load_sector_from_yaml(Path(args.sector_file))
        print(f"✓ Loaded sector: {config.name} ({config.id})")
        print(f"  CISA Sector: {config.cisa_sector}")
        print(f"  Terminology mappings: {len(config.terminology)}")
        print(f"  Stakeholder mappings: {len(config.stakeholders)}")
        print(f"  Context templates: {len(config.context_templates)}")
        print(f"  Use cases: {len(config.use_cases)}")

        # Validate no pressure signals
        print("\nValidating for pressure signals...")
        issues = []
        for scenario, template in config.context_templates.items():
            result = validate_transformation(template, template)
            for signal in result["pressure_signals_detected"]:
                issues.append(f"{scenario}: pressure signal '{signal}'")

        if issues:
            print(f"✗ Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ No pressure signals detected")

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        sys.exit(1)


def cmd_transform(args):
    """Transform a baseline prompt to industry prompt."""
    from secdrift.transformer import PromptTransformer

    transformer = PromptTransformer(args.sector)

    if args.prompt:
        prompt = args.prompt
    elif args.file:
        prompt = Path(args.file).read_text()
    else:
        print("Provide --prompt or --file")
        sys.exit(1)

    industry_prompt = transformer.transform(prompt, args.scenario)
    print(industry_prompt)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="secdrift",
        description="SecDrift: Measuring Sector-Conditioned Security Drift",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark")
    run_parser.add_argument("-m", "--model", dest="models", action="append", help="Model(s) to use")
    run_parser.add_argument("-g", "--group", help="Model group to use")
    run_parser.add_argument("-o", "--output", help="Output file path")
    run_parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers")
    run_parser.add_argument("-r", "--replicates", type=int, default=1, help="Replicates per condition")
    run_parser.add_argument("--scenarios", nargs="+", help="Specific scenarios")
    run_parser.add_argument("--sectors", nargs="+", help="Specific sectors")
    run_parser.set_defaults(func=cmd_run)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("results_file", help="Results JSONL file")
    analyze_parser.add_argument("-o", "--output", help="Output report path")
    analyze_parser.set_defaults(func=cmd_analyze)

    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List available models")
    list_models_parser.set_defaults(func=cmd_list_models)

    # List sectors command
    list_sectors_parser = subparsers.add_parser("list-sectors", help="List available sectors")
    list_sectors_parser.set_defaults(func=cmd_list_sectors)

    # Validate sector command
    validate_parser = subparsers.add_parser("validate-sector", help="Validate sector config")
    validate_parser.add_argument("sector_file", help="Sector YAML file")
    validate_parser.set_defaults(func=cmd_validate_sector)

    # Transform command
    transform_parser = subparsers.add_parser("transform", help="Transform prompt")
    transform_parser.add_argument("--sector", required=True, help="Target sector")
    transform_parser.add_argument("--scenario", required=True, help="Scenario type")
    transform_parser.add_argument("--prompt", help="Prompt text")
    transform_parser.add_argument("--file", help="Prompt file")
    transform_parser.set_defaults(func=cmd_transform)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
