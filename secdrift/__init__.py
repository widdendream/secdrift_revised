"""
SecDrift: Measuring Sector-Conditioned Security Drift in AI-Generated Code

A reproducible benchmark framework for evaluating how industry context
affects the security posture of LLM-generated code.

Research Question: Does industry context systematically influence the
security posture of generated code, even when the functional task is unchanged?

Key Components:
- models: Provider-agnostic model adapters (Bedrock, OpenAI, DeepSeek, etc.)
- transformer: Systematic baseline-to-industry prompt transformation
- runner: Benchmark execution engine
- analysis: Statistical analysis with proper corrections
- analyzers: Vulnerability detection (Bandit + Semgrep)

Usage:
    # Run benchmark with specific model
    python -m secdrift.runner --model claude-3-5-sonnet

    # Run with model group
    python -m secdrift.runner --group paper --replicates 5

    # Analyze results
    python -m secdrift.analysis results/benchmark_results.jsonl

    # List available models
    python -m secdrift.runner --list-models
"""

__version__ = "1.0.0"
__author__ = "SecDrift Research Team"

from secdrift.runner import BenchmarkRunner
from secdrift.models import get_model, get_models, list_models
from secdrift.transformer import (
    PromptTransformer,
    SectorConfig,
    list_sectors,
    get_sector_config,
    generate_sector_prompts,
)

__all__ = [
    # Runner
    "BenchmarkRunner",
    # Models
    "get_model",
    "get_models",
    "list_models",
    # Transformer
    "PromptTransformer",
    "SectorConfig",
    "list_sectors",
    "get_sector_config",
    "generate_sector_prompts",
]
