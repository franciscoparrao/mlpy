# Changelog

All notable changes to MLPY will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [0.1.0] - 2025-01-01

### Added
- Initial release of MLPY framework
- Core components: Task, Learner, Measure, Resampling
- Integration with scikit-learn models
- Pipeline system for composable ML workflows
- Benchmark functionality for model comparison
- AutoML capabilities: hyperparameter tuning and feature engineering
- Parallel execution backends (threading, multiprocessing, joblib)
- Callback system for monitoring experiments
- Visualization tools for results analysis
- Comprehensive documentation and examples

### Core Features
- **Tasks**: Unified interface for classification and regression problems
- **Learners**: Abstract interface with sklearn integration
- **Measures**: 15+ evaluation metrics
- **Resampling**: CV, holdout, bootstrap, and more
- **Pipelines**: Composable preprocessing and modeling
- **Benchmarking**: Systematic model comparison
- **AutoML**: Grid/random search, feature engineering
- **Parallelization**: Multiple backend options
- **Callbacks**: Progress, logging, early stopping
- **Visualization**: Result plots and analysis

### Documentation
- Sphinx-based documentation
- 2 interactive Jupyter notebooks
- 2 complete example scripts
- API reference (in progress)

[Unreleased]: https://github.com/mlpy-project/mlpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mlpy-project/mlpy/releases/tag/v0.1.0