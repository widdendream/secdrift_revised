"""
SecDrift Benchmark Runner

Executes paired-prompt evaluations:
1. Load baseline prompts from established benchmarks
2. Load industry-transformed prompts
3. Generate code with LLMs (any provider via config)
4. Detect vulnerabilities with Bandit + Semgrep
5. Record results for drift analysis

Usage:
    python -m secdrift.runner --model claude-3-5-sonnet
    python -m secdrift.runner --group paper
    python -m secdrift.runner --model llama-3-1-70b --replicates 5
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import yaml
from tqdm import tqdm

from secdrift.models import get_model, get_models, list_models
from secdrift.models.base import ModelAdapter
from secdrift.analyzers.bandit_analyzer import BanditAnalyzer
from secdrift.analyzers.semgrep_analyzer import SemgrepAnalyzer
from secdrift.transformer import PromptTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    evaluation_id: str
    timestamp: str
    scenario_id: str
    cwe: str
    prompt_type: str  # 'baseline', 'matched_baseline', or 'industry'
    sector: Optional[str]  # None for baseline, sector_id for matched_baseline and industry
    model: str
    provider: str
    prompt_text: str
    generated_code: str
    vulnerabilities: List[str]  # List of CWE IDs
    is_vulnerable: bool
    detection_tool: str
    confidence: str
    latency_ms: float
    replicate: int = 0
    error: Optional[str] = None


class BenchmarkRunner:
    """Executes SecDrift benchmark evaluations.

    Workflow:
    1. Load configuration and prompts
    2. For each (scenario, prompt_type, sector, model, replicate):
       a. Generate code with LLM
       b. Detect vulnerabilities
       c. Record result
    3. Save results to JSONL
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        group: Optional[str] = None,
        output_dir: str = "results",
        parallel_workers: Optional[int] = None,
        replicates: int = 1,
        config_path: Optional[Path] = None,
    ):
        """Initialize the benchmark runner.

        Args:
            models: List of model names to use (from config)
            group: Model group name (ignored if models provided)
            output_dir: Directory for results
            parallel_workers: Number of parallel workers (default from config)
            replicates: Number of replicates per condition
            config_path: Path to model config (default: config/models.yaml)
        """
        self.output_dir = PROJECT_ROOT / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replicates = replicates

        # Load config for execution settings
        config_path = config_path or CONFIG_DIR / "models.yaml"
        with open(config_path) as f:
            full_config = yaml.safe_load(f)

        # Get parallel workers from config or argument
        exec_config = full_config.get("execution", {})
        self.parallel_workers = parallel_workers or exec_config.get("parallel_workers", 15)

        # Initialize models from config
        self.model_adapters = {}

        if models:
            for name in models:
                self.model_adapters[name] = get_model(name, config_path)
        else:
            for adapter in get_models(group=group, config_path=config_path):
                self.model_adapters[adapter.model_id] = adapter

        # Initialize analyzers
        self.bandit = BanditAnalyzer()
        self.semgrep = SemgrepAnalyzer()

        logger.info(f"Initialized BenchmarkRunner with {len(self.model_adapters)} models")
        logger.info(f"Models: {list(self.model_adapters.keys())}")

    def load_prompts(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load baseline and industry-specific prompts.

        Returns:
            Dict with structure:
            {
                'baseline': {scenario_id: {'prompt': text, 'cwe': cwe_id}},
                'emergency_services': {scenario_id: {'prompt': text, 'cwe': cwe_id}},
                'government': {scenario_id: {'prompt': text, 'cwe': cwe_id}}
            }
        """
        prompts = {'baseline': {}}

        # Load baseline prompts
        baseline_path = SCENARIOS_DIR / "baseline_prompts.yaml"
        with open(baseline_path) as f:
            baseline_data = yaml.safe_load(f)
            for scenario_id, scenario in baseline_data['scenarios'].items():
                prompts['baseline'][scenario_id] = {
                    'prompt': scenario['baseline_prompt'],
                    'cwe': scenario.get('cwe', 'unknown'),
                }

        # Load industry prompts
        for sector_file in SCENARIOS_DIR.glob("industry_*.yaml"):
            sector_id = sector_file.stem.replace("industry_", "")
            prompts[sector_id] = {}

            with open(sector_file) as f:
                industry_data = yaml.safe_load(f)
                for scenario_id, scenario in industry_data['scenarios'].items():
                    prompts[sector_id][scenario_id] = {
                        'prompt': scenario['industry_prompt'],
                        'cwe': scenario.get('cwe', 'unknown'),
                    }

        sectors = [k for k in prompts.keys() if k != 'baseline']
        logger.info(f"Loaded prompts: baseline + {len(sectors)} sectors ({', '.join(sectors)})")
        logger.info(f"Scenarios per sector: {len(prompts['baseline'])}")

        return prompts

    def generate_code(
        self,
        adapter: ModelAdapter,
        prompt: str,
    ) -> Tuple[str, float, Optional[str]]:
        """Generate code using model adapter.

        Args:
            adapter: Model adapter instance
            prompt: Prompt text

        Returns:
            (generated_code, latency_ms, error)
        """
        try:
            response = adapter.generate(prompt)

            if response.success:
                code = adapter.extract_code(response.text)
                return code, response.latency_ms, None
            else:
                return f"# Error: {response.error}", response.latency_ms, response.error

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return f"# Error: {str(e)}", 0.0, str(e)

    def detect_vulnerabilities(self, code: str) -> Tuple[List[str], bool, str, str]:
        """Detect vulnerabilities in generated code.

        Args:
            code: Generated Python code

        Returns:
            (vulnerabilities, is_vulnerable, detection_tool, confidence)
        """
        # Run Bandit
        bandit_results = self.bandit.analyze(code)

        # Run Semgrep
        semgrep_results = self.semgrep.analyze(code)

        # Combine results
        all_cwes = set()
        all_cwes.update(bandit_results.get('cwes', []))
        all_cwes.update(semgrep_results.get('cwes', []))

        vulnerabilities = sorted(list(all_cwes))
        is_vulnerable = len(vulnerabilities) > 0

        # Determine detection tool and confidence
        if bandit_results.get('cwes') and semgrep_results.get('cwes'):
            detection_tool = "Bandit + Semgrep"
            confidence = "HIGH"
        elif bandit_results.get('cwes') or semgrep_results.get('cwes'):
            detection_tool = "Bandit" if bandit_results.get('cwes') else "Semgrep"
            confidence = "MEDIUM"
        else:
            detection_tool = "None"
            confidence = "N/A"

        return vulnerabilities, is_vulnerable, detection_tool, confidence

    def run_evaluation(
        self,
        scenario_id: str,
        cwe: str,
        prompt_type: str,
        sector: Optional[str],
        model_name: str,
        prompt_text: str,
        replicate: int,
    ) -> EvaluationResult:
        """Run single evaluation.

        Args:
            scenario_id: Scenario identifier (e.g., 'sql_injection')
            cwe: CWE identifier (e.g., 'CWE-89')
            prompt_type: 'baseline' or 'industry'
            sector: Sector identifier (None for baseline)
            model_name: Model identifier
            prompt_text: Prompt to send to LLM
            replicate: Replicate number

        Returns:
            EvaluationResult
        """
        eval_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        logger.debug(f"Running: {scenario_id}/{prompt_type}/{sector or 'baseline'}/{model_name}/r{replicate}")

        try:
            adapter = self.model_adapters[model_name]

            # Generate code
            generated_code, latency_ms, error = self.generate_code(adapter, prompt_text)

            if error:
                return EvaluationResult(
                    evaluation_id=eval_id,
                    timestamp=timestamp,
                    scenario_id=scenario_id,
                    cwe=cwe,
                    prompt_type=prompt_type,
                    sector=sector,
                    model=model_name,
                    provider=adapter.provider,
                    prompt_text=prompt_text,
                    generated_code=generated_code,
                    vulnerabilities=[],
                    is_vulnerable=False,
                    detection_tool="Error",
                    confidence="N/A",
                    latency_ms=latency_ms,
                    replicate=replicate,
                    error=error,
                )

            # Detect vulnerabilities
            vulns, is_vuln, tool, conf = self.detect_vulnerabilities(generated_code)

            return EvaluationResult(
                evaluation_id=eval_id,
                timestamp=timestamp,
                scenario_id=scenario_id,
                cwe=cwe,
                prompt_type=prompt_type,
                sector=sector,
                model=model_name,
                provider=adapter.provider,
                prompt_text=prompt_text,
                generated_code=generated_code,
                vulnerabilities=vulns,
                is_vulnerable=is_vuln,
                detection_tool=tool,
                confidence=conf,
                latency_ms=latency_ms,
                replicate=replicate,
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {scenario_id}/{model_name} - {e}")
            return EvaluationResult(
                evaluation_id=eval_id,
                timestamp=timestamp,
                scenario_id=scenario_id,
                cwe=cwe,
                prompt_type=prompt_type,
                sector=sector,
                model=model_name,
                provider="unknown",
                prompt_text=prompt_text,
                generated_code="",
                vulnerabilities=[],
                is_vulnerable=False,
                detection_tool="Error",
                confidence="N/A",
                latency_ms=0.0,
                replicate=replicate,
                error=str(e),
            )

    def run_benchmark(
        self,
        output_file: Optional[str] = None,
        scenarios: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Run full benchmark.

        Args:
            output_file: Output file path (default: results/benchmark_results.jsonl)
            scenarios: Optional list of scenarios to run (default: all)
            sectors: Optional list of sectors to run (default: all)

        Returns:
            List of EvaluationResult
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"benchmark_{timestamp}.jsonl"
        else:
            output_file = Path(output_file)

        # Load prompts
        prompts = self.load_prompts()

        # Filter scenarios if specified
        baseline_scenarios = prompts['baseline']
        if scenarios:
            baseline_scenarios = {k: v for k, v in baseline_scenarios.items() if k in scenarios}

        # Filter sectors if specified
        available_sectors = [k for k in prompts.keys() if k != 'baseline']
        if sectors:
            available_sectors = [s for s in available_sectors if s in sectors]

        # Build evaluation tasks
        tasks = []

        # Baseline evaluations (generic terminology, no context)
        for scenario_id, scenario_data in baseline_scenarios.items():
            for model_name in self.model_adapters.keys():
                for rep in range(self.replicates):
                    tasks.append({
                        'scenario_id': scenario_id,
                        'cwe': scenario_data['cwe'],
                        'prompt_type': 'baseline',
                        'sector': None,
                        'model_name': model_name,
                        'prompt_text': scenario_data['prompt'],
                        'replicate': rep,
                    })

        # Matched baseline evaluations (sector terminology, no context)
        for sector in available_sectors:
            transformer = PromptTransformer(sector)
            for scenario_id, scenario_data in baseline_scenarios.items():
                # Generate matched baseline by applying only terminology transformation
                matched_prompt = transformer.transform_terminology_only(scenario_data['prompt'])
                
                for model_name in self.model_adapters.keys():
                    for rep in range(self.replicates):
                        tasks.append({
                            'scenario_id': scenario_id,
                            'cwe': scenario_data['cwe'],
                            'prompt_type': 'matched_baseline',
                            'sector': sector,
                            'model_name': model_name,
                            'prompt_text': matched_prompt,
                            'replicate': rep,
                        })

        # Industry evaluations (sector terminology + full context)
        for sector in available_sectors:
            for scenario_id, scenario_data in prompts[sector].items():
                if scenarios and scenario_id not in scenarios:
                    continue
                for model_name in self.model_adapters.keys():
                    for rep in range(self.replicates):
                        tasks.append({
                            'scenario_id': scenario_id,
                            'cwe': scenario_data['cwe'],
                            'prompt_type': 'industry',
                            'sector': sector,
                            'model_name': model_name,
                            'prompt_text': scenario_data['prompt'],
                            'replicate': rep,
                        })

        n_models = len(self.model_adapters)
        n_scenarios = len(baseline_scenarios)
        n_sectors = len(available_sectors)

        logger.info(f"Benchmark configuration:")
        logger.info(f"  Models: {n_models}")
        logger.info(f"  Scenarios: {n_scenarios}")
        logger.info(f"  Sectors: {n_sectors}")
        logger.info(f"  Prompt types: 3 (baseline, matched_baseline, industry)")
        logger.info(f"  Replicates: {self.replicates}")
        logger.info(f"  Total evaluations: {len(tasks)}")

        # Clear output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("")

        # Run evaluations in parallel
        results = []
        errors = 0

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self.run_evaluation, **task): task
                for task in tasks
            }

            with tqdm(total=len(tasks), desc="Running evaluations") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    if result.error:
                        errors += 1

                    # Save incrementally
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(asdict(result)) + '\n')

                    pbar.update(1)

        # Summary
        vulnerable = sum(1 for r in results if r.is_vulnerable)
        logger.info(f"Benchmark complete!")
        logger.info(f"  Results: {output_file}")
        logger.info(f"  Total: {len(results)}, Vulnerable: {vulnerable}, Errors: {errors}")

        return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SecDrift benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default models (from config)
  python -m secdrift.runner

  # Run with specific model
  python -m secdrift.runner --model claude-3-5-sonnet

  # Run with model group
  python -m secdrift.runner --group paper

  # Run with multiple replicates
  python -m secdrift.runner --replicates 5

  # List available models
  python -m secdrift.runner --list-models
        """
    )

    parser.add_argument(
        '--model', '-m',
        action='append',
        dest='models',
        help='Model name(s) to use (can specify multiple)'
    )
    parser.add_argument(
        '--group', '-g',
        help='Model group to use (from config)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--replicates', '-r',
        type=int,
        default=1,
        help='Number of replicates per condition (default: 1)'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        help='Specific scenarios to run'
    )
    parser.add_argument(
        '--sectors',
        nargs='+',
        help='Specific sectors to run'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in list_models():
            print(f"  - {model}")
        return

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
    print(f"\nNext steps:")
    print(f"  python -m secdrift.analysis <results_file>")


if __name__ == '__main__':
    main()
