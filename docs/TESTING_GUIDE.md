# 🧪 MLPY Testing Guide

## 📋 Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Coverage Reports](#coverage-reports)
- [CI/CD Pipeline](#cicd-pipeline)
- [Best Practices](#best-practices)

---

## 🎯 Overview

MLPY uses a comprehensive testing strategy to ensure code quality and reliability:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Benchmarks**: Track performance metrics
- **Coverage Target**: >90% code coverage

### Testing Stack
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **pytest-benchmark**: Performance testing
- **GitHub Actions**: CI/CD automation

---

## 🚀 Quick Start

### Installation
```bash
# Install development dependencies
pip install -e ".[dev]"

# Or use make
make dev-install
```

### Run All Tests
```bash
# Using pytest directly
pytest tests/ -v

# Using make
make test

# Using custom runner
python run_tests.py --all
```

### Quick Coverage Check
```bash
# Generate coverage report
make coverage

# View HTML report
open htmlcov/index.html
```

---

## 📁 Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Unit tests
│   ├── test_core_models.py
│   ├── test_learners.py
│   ├── test_tasks.py
│   └── test_validation.py
├── integration/          # Integration tests
│   ├── test_end_to_end.py
│   ├── test_pipelines.py
│   └── test_model_registry.py
├── benchmarks/           # Performance tests
│   └── test_performance.py
└── fixtures/            # Test data and resources
    └── sample_data.csv
```

---

## 🏃 Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_core_models.py

# Run specific test function
pytest tests/unit/test_core_models.py::TestLearnerBase::test_learner_initialization

# Run tests by marker
pytest -m unit
pytest -m "not slow"
pytest -m "requires_sklearn"
```

### Using the Test Runner Script

```bash
# Run unit tests only
python run_tests.py --unit

# Run integration tests only
python run_tests.py --integration

# Run with coverage and HTML report
python run_tests.py --coverage --html

# Run in parallel (4 workers)
python run_tests.py --parallel 4

# Stop on first failure
python run_tests.py --failfast
```

### Using Make Commands

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-int

# Run smoke tests
make test-smoke

# Generate coverage report
make coverage

# Run benchmarks
make benchmark
```

---

## ✍️ Writing Tests

### Test Naming Convention
```python
# File naming
test_<module_name>.py

# Class naming
class Test<ComponentName>:
    
# Method naming
def test_<specific_behavior>():
```

### Basic Test Structure
```python
import pytest
import numpy as np
from mlpy.learners.base import Learner

class TestLearnerBase:
    """Test base learner functionality."""
    
    @pytest.mark.unit
    def test_learner_initialization(self):
        """Test learner can be initialized."""
        learner = Learner(id="test", label="Test")
        assert learner.id == "test"
        assert learner.label == "Test"
        assert not learner.is_trained
    
    @pytest.mark.unit
    @pytest.mark.requires_sklearn
    def test_learner_with_sklearn(self, sample_data):
        """Test learner with scikit-learn dependency."""
        # Test implementation
        pass
```

### Using Fixtures

```python
# In conftest.py
@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    df = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'target': np.random.choice(['A', 'B', 'C'], 100)
    })
    return df

# In test file
def test_classification(sample_classification_data):
    """Test with fixture data."""
    assert len(sample_classification_data) == 100
```

### Test Markers

```python
# Mark test types
@pytest.mark.unit          # Fast, isolated unit test
@pytest.mark.integration   # Integration test
@pytest.mark.slow          # Slow test (>1s)
@pytest.mark.smoke         # Quick smoke test

# Mark dependencies
@pytest.mark.requires_sklearn      # Requires scikit-learn
@pytest.mark.requires_torch        # Requires PyTorch
@pytest.mark.requires_transformers # Requires transformers

# Skip conditions
@pytest.mark.skipif(not has_sklearn, reason="scikit-learn not installed")
@pytest.mark.xfail(reason="Known issue, fix pending")
```

### Parameterized Tests

```python
@pytest.mark.parametrize("n_estimators,max_depth,expected_score", [
    (10, 5, 0.8),
    (100, 10, 0.9),
    (200, None, 0.95),
])
def test_random_forest_params(n_estimators, max_depth, expected_score):
    """Test RandomForest with different parameters."""
    model = RandomForest(n_estimators=n_estimators, max_depth=max_depth)
    score = model.train_and_score(data)
    assert score >= expected_score
```

### Testing Exceptions

```python
def test_invalid_input():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="Invalid parameter"):
        learner = Learner(invalid_param="bad")
    
    with pytest.raises(RuntimeError, match="Model must be trained"):
        untrained_learner.predict(data)
```

---

## 📊 Coverage Reports

### Generate Coverage
```bash
# Terminal report
pytest --cov=mlpy --cov-report=term-missing

# HTML report
pytest --cov=mlpy --cov-report=html

# XML report (for CI)
pytest --cov=mlpy --cov-report=xml

# Multiple reports
pytest --cov=mlpy --cov-report=term-missing --cov-report=html --cov-report=xml
```

### Coverage Configuration
```ini
# .coveragerc
[run]
source = mlpy
omit = 
    */tests/*
    */test_*.py

[report]
precision = 2
show_missing = True
fail_under = 90

exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
```

### Viewing Reports
```bash
# Open HTML report
open htmlcov/index.html

# Or serve it
python -m http.server 8000 --directory htmlcov
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Our CI/CD pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Daily at 2 AM UTC (scheduled)

### Pipeline Stages

1. **Linting**: Code quality checks
   - Black (formatting)
   - isort (import sorting)
   - Flake8 (style guide)
   - mypy (type checking)

2. **Testing**: Multi-platform tests
   - OS: Ubuntu, Windows, macOS
   - Python: 3.8, 3.9, 3.10, 3.11
   - Unit and integration tests

3. **Coverage**: Code coverage reporting
   - Upload to Codecov
   - Fail if <90% coverage

4. **Documentation**: Build docs
   - Sphinx documentation
   - API reference generation

5. **Security**: Security scanning
   - Bandit security linter
   - Safety dependency check

### Running CI Locally
```bash
# Simulate CI checks locally
make ci

# Or run individual checks
make lint
make test
make coverage
```

---

## 💡 Best Practices

### 1. Test Organization
- Keep tests close to the code they test
- Use descriptive test names
- Group related tests in classes
- One assertion per test when possible

### 2. Fixtures
- Use fixtures for reusable test data
- Keep fixtures in `conftest.py`
- Use descriptive fixture names
- Clean up resources in fixtures

### 3. Mocking
```python
from unittest.mock import Mock, patch

@patch('mlpy.external.api_call')
def test_with_mock(mock_api):
    mock_api.return_value = {'status': 'success'}
    result = function_using_api()
    assert result == 'success'
```

### 4. Performance Testing
```python
@pytest.mark.benchmark
def test_model_performance(benchmark):
    """Benchmark model training time."""
    model = RandomForest()
    result = benchmark(model.train, data)
    assert result.training_time < 1.0  # Less than 1 second
```

### 5. Test Data
- Use small, representative datasets
- Generate synthetic data when possible
- Store large test files in `fixtures/`
- Use deterministic random seeds

### 6. Continuous Testing
```bash
# Watch mode - rerun tests on file changes
pip install pytest-watch
ptw -- -v

# Or use make
make watch
```

---

## 🐛 Debugging Tests

### Verbose Output
```bash
# Show print statements
pytest -s

# Show detailed assertion failures
pytest -vv

# Show local variables in tracebacks
pytest -l
```

### Interactive Debugging
```python
def test_complex_logic():
    """Test with debugging."""
    import pdb; pdb.set_trace()  # Breakpoint
    result = complex_function()
    assert result == expected
```

### Run Specific Tests
```bash
# Run failed tests only
pytest --lf

# Run failed tests first, then others
pytest --ff

# Run tests that match keyword
pytest -k "classification"
```

---

## 📈 Testing Metrics

### Current Status
- **Total Tests**: 50+
- **Coverage**: Target >90%
- **Test Execution Time**: <5 minutes
- **Platforms**: Linux, Windows, macOS
- **Python Versions**: 3.8-3.11

### Test Categories
- Unit Tests: 70%
- Integration Tests: 20%
- End-to-End Tests: 10%

---

## 🆘 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure MLPY is installed in development mode
pip install -e .
```

**2. Missing Dependencies**
```bash
# Install all test dependencies
pip install -r requirements-dev.txt
```

**3. Coverage Not Working**
```bash
# Reinstall pytest-cov
pip install --upgrade pytest-cov
```

**4. Tests Too Slow**
```bash
# Run in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

---

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)

---

## 🎉 Contributing

When contributing to MLPY:

1. Write tests for new features
2. Ensure all tests pass locally
3. Maintain >90% coverage
4. Update test documentation
5. Follow naming conventions

```bash
# Before committing
make check  # Runs lint and tests
```

---

*Happy Testing! 🧪*