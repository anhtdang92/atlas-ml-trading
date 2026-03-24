.PHONY: help install install-dev lint format test test-cov test-ml clean run train

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install dev dependencies (lint, test, type-check)
	pip install -r requirements.txt
	pip install mypy pre-commit
	pre-commit install

lint:  ## Run linters (flake8 + black check + mypy)
	black --check --line-length 120 ml/ data/ ui/ tests/
	flake8 ml/ data/ --max-line-length 120 --extend-ignore E203,W503
	mypy ml/ data/ --ignore-missing-imports --no-error-summary || true

format:  ## Auto-format code with black
	black --line-length 120 ml/ data/ ui/ tests/

test:  ## Run unit tests
	pytest tests/unit/ -v --tb=short -x

test-cov:  ## Run tests with coverage report
	pytest tests/unit/ -v --tb=short --cov=ml --cov=data --cov-report=term-missing --cov-fail-under=60

test-ml:  ## Run ML pipeline tests only
	pytest tests/unit/test_ml_pipeline.py -v --tb=short

test-all:  ## Run all tests including integration
	pytest tests/ -v --tb=short -m "not gcp"

validate:  ## Validate stock universe and feature engineering
	python -c "from data.stock_api import get_all_symbols; symbols = get_all_symbols(); assert len(symbols) >= 30, f'Expected 30+ symbols, got {len(symbols)}'; print(f'Stock universe OK: {len(symbols)} symbols')"
	python -c "from ml.feature_engineering import FeatureEngineer; fe = FeatureEngineer(); assert fe.EXPECTED_FEATURE_COUNT == 29; print(f'Feature count OK: {fe.EXPECTED_FEATURE_COUNT}')"

run:  ## Start the Streamlit dashboard
	streamlit run app.py

train:  ## Launch ML training (interactive)
	./bin/train_now.sh

clean:  ## Remove build artifacts, caches, pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
