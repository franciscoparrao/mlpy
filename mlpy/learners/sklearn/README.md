# Scikit-learn Integration for MLPY

This module provides seamless integration between MLPY and scikit-learn, allowing you to use any scikit-learn model within the MLPY framework.

## Features

- **Pre-built Wrappers**: Ready-to-use wrappers for common scikit-learn models
- **Auto-Wrapper**: Automatically wrap any scikit-learn compatible estimator
- **Full MLPY Integration**: Use sklearn models with all MLPY features (pipelines, tuning, benchmarking, etc.)
- **Consistent API**: Same interface across all learners
- **Type Safety**: Proper type hints and parameter validation

## Installation

```bash
pip install scikit-learn
```

## Structure

```
mlpy/learners/sklearn/
├── __init__.py           # Module exports
├── base.py              # Base wrapper classes
├── classification.py    # Classification model wrappers
├── regression.py        # Regression model wrappers
├── auto_wrap.py        # Automatic wrapper functionality
└── README.md           # This file
```

## Basic Usage

```python
from mlpy.learners import LearnerRandomForest
from mlpy.tasks import TaskClassif
from mlpy import resample

# Create task
task = TaskClassif(data=df, target='target')

# Create learner
learner = LearnerRandomForest(n_estimators=100, max_depth=10)

# Evaluate
result = resample(task, learner)
print(f"Accuracy: {result.aggregate()['classif.acc']['mean']:.3f}")
```

## Available Wrappers

### Classification
- `LearnerLogisticRegression` - Logistic Regression
- `LearnerDecisionTree` - Decision Tree Classifier  
- `LearnerRandomForest` - Random Forest Classifier
- `LearnerGradientBoosting` - Gradient Boosting Classifier
- `LearnerSVM` - Support Vector Machine
- `LearnerKNN` - K-Nearest Neighbors
- `LearnerNaiveBayes` - Gaussian Naive Bayes
- `LearnerMLPClassifier` - Multi-layer Perceptron

### Regression
- `LearnerLinearRegression` - Linear Regression
- `LearnerRidge` - Ridge Regression
- `LearnerLasso` - Lasso Regression
- `LearnerElasticNet` - Elastic Net
- `LearnerDecisionTreeRegressor` - Decision Tree Regressor
- `LearnerRandomForestRegressor` - Random Forest Regressor
- `LearnerGradientBoostingRegressor` - Gradient Boosting Regressor
- `LearnerSVR` - Support Vector Regression
- `LearnerKNNRegressor` - K-Nearest Neighbors Regressor
- `LearnerMLPRegressor` - Multi-layer Perceptron Regressor

## Auto-Wrapper

Wrap any scikit-learn estimator automatically:

```python
from mlpy.learners import auto_sklearn
from sklearn.ensemble import ExtraTreesClassifier

# Wrap a class
learner = auto_sklearn(ExtraTreesClassifier, n_estimators=100)

# Wrap an instance
clf = ExtraTreesClassifier(n_estimators=100)
learner = auto_sklearn(clf)

# Specify task type if needed
learner = auto_sklearn(MyEstimator, task_type='regr')
```

## Advanced Features

### Pipeline Integration
```python
from mlpy.pipelines import linear_pipeline, PipeOpScale, PipeOpLearner

pipeline = linear_pipeline([
    PipeOpScale(method='standard'),
    PipeOpLearner(LearnerSVM(C=1.0, kernel='rbf'))
])
```

### Hyperparameter Tuning
```python
from mlpy.automl import ParamSet, ParamInt, ParamFloat, TunerGrid

params = ParamSet([
    ParamInt("n_estimators", 50, 200),
    ParamFloat("learning_rate", 0.01, 0.3, log=True)
])

tuner = TunerGrid(resolution=5)
result = tuner.tune(learner, task, params)
```

### Parallel Processing
```python
# Use all CPU cores
learner = LearnerRandomForest(n_jobs=-1)

# Or set globally
from mlpy.parallel import set_parallel_backend, BackendJoblib
set_parallel_backend(BackendJoblib(n_jobs=4))
```

## Implementation Details

### Base Classes

- `LearnerSKLearn`: Generic base class for any sklearn estimator
- `LearnerClassifSKLearn`: Base for classification models
- `LearnerRegrSKLearn`: Base for regression models

### Key Methods

- `train(task)`: Train the model
- `predict(task)`: Make predictions
- `clone()`: Create a deep copy
- `reset()`: Reset to untrained state
- `get_params()`: Get hyperparameters
- `set_params(**params)`: Set hyperparameters

### Features

1. **Automatic Type Detection**: Automatically detects if model is classifier or regressor
2. **Probability Support**: Handles probability predictions for classifiers
3. **Parameter Management**: Full parameter getting/setting support
4. **Model Access**: Access underlying sklearn model via `.model` attribute
5. **Graceful Degradation**: Falls back gracefully if model doesn't support certain features

## Examples

See the `examples/` directory for complete examples:
- `sklearn_integration.py` - Basic usage examples
- `automl_example.py` - Using sklearn models with AutoML
- `automl_advanced.py` - Advanced techniques

## Contributing

When adding new sklearn wrappers:

1. Add the wrapper class to `classification.py` or `regression.py`
2. Follow the existing pattern with proper docstrings
3. Export in `__init__.py`
4. Add tests in `tests/test_learners_sklearn.py`
5. Update documentation

## Testing

Run tests with:
```bash
pytest tests/test_learners_sklearn.py
```

## License

Same as MLPY - MIT License