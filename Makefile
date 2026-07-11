.PHONY: help install install-dev test lint format clean reproduce run analyze figures dist

help:
	@echo "SecDrift - Measuring Sector-Conditioned Security Drift in AI-Generated Code"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install package and static-analysis backends"
	@echo "  make install-dev - Install with dev dependencies"
	@echo "  make test        - Run unit tests with coverage"
	@echo "  make lint        - Run linters (ruff, mypy)"
	@echo "  make format      - Format code with black + ruff --fix"
	@echo "  make reproduce   - Regenerate the paper's statistics from shipped data (no API keys)"
	@echo "  make run         - Re-run the full benchmark (needs model access)"
	@echo "  make analyze     - Analyze the shipped evaluation data"
	@echo "  make figures     - Regenerate paper figures into paper/figures/"
	@echo "  make clean       - Remove build/test artifacts"

install:
	pip install -e .
	pip install bandit semgrep

install-dev:
	pip install -e ".[dev]"
	pip install bandit semgrep

test:
	pytest

test-cov:
	pytest --cov=secdrift --cov-report=term --cov-report=html

lint:
	ruff check secdrift/
	mypy secdrift/

format:
	black secdrift/
	ruff check --fix secdrift/

# Reproduce the paper's numbers from the shipped dataset (no credentials needed).
reproduce:
	python analysis/cwe_exclusion.py
	python analysis/leave_one_cwe_out.py
	python analysis/cramers_v.py
	python analysis/completeness.py
	python analysis/functional_correctness.py
	python analysis/contract_drift.py
	python analysis/analyze_control_pilot.py

# Re-run the benchmark end-to-end (requires AWS Bedrock or configured provider).
run:
	python -m secdrift.runner --group paper --replicates 5 \
		--output results/benchmark_results.jsonl --workers 20

analyze:
	python -m secdrift.analysis results/latest_run/merged_results_5rep.jsonl

figures:
	python scripts/generate_paper_figures.py \
		results/latest_run/merged_results_5rep.jsonl -o paper/figures

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

dist:
	python -m build
	twine check dist/*
