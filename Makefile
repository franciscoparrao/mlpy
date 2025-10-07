# Makefile for MLPY development

.PHONY: help clean test coverage install dev-install lint format docs

help:
	@echo "MLPY Development Commands"
	@echo "========================="
	@echo "make install      - Install MLPY in production mode"
	@echo "make dev-install  - Install MLPY in development mode with all dependencies"
	@echo "make test         - Run all tests"
	@echo "make test-unit    - Run unit tests only"
	@echo "make test-int     - Run integration tests only"
	@echo "make coverage     - Run tests with coverage report"
	@echo "make lint         - Run code linters"
	@echo "make format       - Format code with black and isort"
	@echo "make docs         - Build documentation"
	@echo "make clean        - Clean build artifacts"
	@echo "make benchmark    - Run performance benchmarks"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pip install pytest pytest-cov pytest-xdist pytest-timeout
	pip install black isort flake8 mypy
	pip install sphinx sphinx-rtd-theme
	@echo "Installing optional dependencies..."
	-pip install scikit-learn xgboost lightgbm
	@echo "Development environment ready!"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit -v

test-int:
	pytest tests/integration -v

test-smoke:
	pytest tests/ -v -m smoke

coverage:
	pytest tests/ -v --cov=mlpy --cov-report=term-missing --cov-report=html
	@echo "Coverage report available at htmlcov/index.html"

coverage-report:
	coverage report
	coverage html
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html

lint:
	flake8 mlpy tests --max-line-length=100 --ignore=E203,W503
	mypy mlpy --ignore-missing-imports
	black --check mlpy tests
	isort --check-only mlpy tests

format:
	black mlpy tests
	isort mlpy tests
	@echo "Code formatted!"

docs:
	cd docs/sphinx && make clean && make html
	@echo "Documentation built at docs/sphinx/_build/html/index.html"

docs-serve:
	cd docs/sphinx/_build/html && python -m http.server 8000

benchmark:
	pytest tests/benchmarks -v --benchmark-only

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaned all build artifacts!"

# Development workflow shortcuts
check: lint test
	@echo "All checks passed!"

ci: lint test coverage
	@echo "CI checks completed!"

# Docker commands
docker-build:
	docker build -t mlpy:latest .

docker-test:
	docker run --rm mlpy:latest pytest tests/

docker-shell:
	docker run --rm -it -v $(PWD):/app mlpy:latest bash

# Release commands
release-test:
	python -m build
	twine check dist/*

release-pypi:
	python -m build
	twine upload dist/*

# Git hooks
install-hooks:
	pre-commit install
	@echo "Git hooks installed!"

# Quick test for specific file
test-file:
	@read -p "Enter test file path: " filepath; \
	pytest $$filepath -v

# Watch mode for development
watch:
	ptw -- -v