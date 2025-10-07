# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLPY is a modern, extensible machine learning framework for Python inspired by mlr3. It provides a unified, object-oriented interface for machine learning tasks with emphasis on composability and extensibility.

**Package Name:** `mlpy-geo` (PyPI)
**Version:** 0.1.0-dev
**Python Support:** 3.8+

## Core Architecture

MLPY follows a modular, mlr3-inspired design with these core components:

### 1. Tasks (`mlpy/tasks/`)
Encapsulate ML problems with data and metadata:
- `TaskClassif` / `TaskRegr` - Supervised learning
- `TaskCluster` - Clustering
- `TaskTimeSeries`, `TaskForecasting` - Time series analysis
- `TaskClassifSpatial`, `TaskRegrSpatial` - Geospatial ML
- `ValidatedTask` - Tasks with automatic data validation

### 2. Learners (`mlpy/learners/`)
Unified interface for ML algorithms:
- Base classes: `Learner`, `LearnerClassif`, `LearnerRegr`
- Sklearn wrappers in `sklearn/`
- Native implementations in `native/`
- Optional integrations: XGBoost, LightGBM, CatBoost, TGPY, H2O
- Time series learners: ARIMA, Prophet, Exponential Smoothing
- Ensemble learners: Voting, Stacking, Blending

### 3. Measures (`mlpy/measures/`)
Performance evaluation metrics:
- Classification: Accuracy, AUC, F1, Precision, Recall, MCC, LogLoss
- Regression: MSE, RMSE, MAE, MAPE, R2, Bias
- Spatial-aware measures for geospatial tasks

### 4. Resamplings (`mlpy/resamplings/`)
Cross-validation and resampling strategies:
- `ResamplingCV`, `ResamplingHoldout`, `ResamplingBootstrap`
- Spatial resampling: `SpatialKFold`, `SpatialBlockCV`, `SpatialBufferCV`

### 5. Pipelines (`mlpy/pipelines/`)
Composable data processing graphs:
- `PipeOp` - Base pipeline operation
- `GraphLearner` - Learner with preprocessing pipeline
- `linear_pipeline()` - Helper for sequential pipelines
- Advanced operators in `advanced_operators.py`
- Lazy evaluation support in `lazy_ops.py`

### 6. AutoML (`mlpy/automl/`)
Automated machine learning:
- Hyperparameter tuning in `tuning.py`
- Feature engineering in `feature_engineering.py`
- Meta-learning in `meta_learning.py`
- Simple AutoML in `simple_automl.py`

### 7. Optional Modules
- `backends/` - Big data support (Dask, Vaex)
- `callbacks/` - Training callbacks (History, EarlyStopping, Checkpoint)
- `interpretability/` - SHAP, LIME integration
- `persistence/` - Model serialization and registry
- `validation/` - Data validation system
- `visualization/` - Plotting utilities
- `cli/` - Command-line interface

## Common Development Commands

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=mlpy tests/

# Specific test file
pytest tests/unit/test_tasks.py -v

# Single test function
pytest tests/unit/test_tasks.py::test_task_creation -v

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Quick validation (runs fast smoke tests)
python test_quick_validation.py
```

### Code Quality

```bash
# Format code with black
black mlpy/ tests/

# Sort imports
isort mlpy/ tests/

# Lint with flake8
flake8 mlpy/ tests/

# Type checking
mypy mlpy/
```

### CLI Commands

```bash
# Access CLI
python -m mlpy --help
mlpy --help

# (See mlpy/cli/ for available commands)
```

### Building and Installation

```bash
# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e .[all]

# Install minimal dependencies only
pip install -e .

# Install specific extras
pip install -e .[dev,visualization,parallel]
```

## Important Design Patterns

### Optional Dependencies Pattern
Many modules use try/except imports to handle optional dependencies gracefully:

```python
try:
    from .optional_module import feature
    _HAS_FEATURE = True
except ImportError:
    _HAS_FEATURE = False
```

This is used throughout for: sklearn, visualization, interpretability, persistence, big data backends, etc.

### Task-Learner-Measure Workflow
The standard MLPY workflow:

```python
# 1. Create task
task = TaskClassif(data=df, target="species")

# 2. Create learner
learner = LearnerRandomForestClassifier()

# 3. Train
learner.train(task)

# 4. Predict
predictions = learner.predict(task)

# 5. Evaluate
measure = MeasureClassifAccuracy()
score = measure.score(predictions.truth, predictions.response)
```

### Resampling Pattern
For cross-validation:

```python
from mlpy import resample

result = resample(
    task=task,
    learner=learner,
    resampling=ResamplingCV(folds=5),
    measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
)
```

### Benchmarking Pattern
Compare multiple learners:

```python
from mlpy import benchmark

result = benchmark(
    tasks=[task1, task2],
    learners=[learner1, learner2, learner3],
    resamplings=ResamplingCV(folds=5),
    measures=[measure1, measure2]
)
```

## Key Files to Understand

- `mlpy/__init__.py` - Main exports and feature flags
- `mlpy/base.py` - Core base classes
- `mlpy/resample.py` - High-level resampling function
- `mlpy/benchmark.py` - Benchmarking functionality
- `pyproject.toml` - Project configuration, dependencies, and tools

## Data Validation System

MLPY includes a comprehensive validation system in `mlpy/validation/`:
- Automatic data quality checks
- Detailed error reporting with `ErrorContext`
- `ValidatedTask` wrapper for tasks
- Use `validate_task_data()` for explicit validation

## Spatial ML Support

Geospatial machine learning features:
- Spatial tasks preserve coordinate information
- Spatial-aware cross-validation prevents spatial autocorrelation
- Spatial measures account for geographic structure
- Located in `mlpy/tasks/spatial.py` and `mlpy/resamplings/spatial.py`

## Testing Strategy

Current status:
- 85% of tests passing (29/34 unit tests)
- 16% code coverage
- Tests organized in `tests/unit/` and `tests/integration/`
- Use `pytest` markers: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `ci.yml` - Main CI pipeline (tests on Ubuntu, Windows, macOS with Python 3.8-3.11)
- `quality.yml` - Code quality checks
- `docs.yml` - Documentation building
- `benchmarks.yml` - Performance benchmarks
- `release.yml` - Release automation

## Model Registry

MLPY includes an experimental model registry system:
- Track model versions and metadata
- Store models with tags and descriptions
- Located in `mlpy/model_registry/`

## Extension Points

To add new components:

1. **New Learner**: Inherit from `Learner`, `LearnerClassif`, or `LearnerRegr`
2. **New Measure**: Inherit from `Measure`, `MeasureClassif`, or `MeasureRegr`
3. **New Resampling**: Inherit from `Resampling` base class
4. **New PipeOp**: Inherit from `PipeOp` base class
5. **New Backend**: Inherit from backend base classes in `mlpy/backends/base.py`
