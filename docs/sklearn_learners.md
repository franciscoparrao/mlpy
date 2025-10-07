# Scikit-learn Learners in MLPY

This guide covers how to use scikit-learn models within the MLPY framework. MLPY provides seamless integration with scikit-learn through dedicated wrappers and an auto-wrapper functionality.

## Table of Contents
- [Installation](#installation)
- [Available Learners](#available-learners)
- [Basic Usage](#basic-usage)
- [Classification Learners](#classification-learners)
- [Regression Learners](#regression-learners)
- [Auto-Wrapper](#auto-wrapper)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

## Installation

To use scikit-learn learners in MLPY, you need to have scikit-learn installed:

```bash
pip install scikit-learn
```

MLPY will automatically detect if scikit-learn is available and enable the sklearn learners.

## Available Learners

### Classification Learners
- `LearnerLogisticRegression` - Logistic Regression
- `LearnerDecisionTree` - Decision Tree Classifier
- `LearnerRandomForest` - Random Forest Classifier
- `LearnerGradientBoosting` - Gradient Boosting Classifier
- `LearnerSVM` - Support Vector Machine Classifier
- `LearnerKNN` - K-Nearest Neighbors Classifier
- `LearnerNaiveBayes` - Gaussian Naive Bayes
- `LearnerMLPClassifier` - Multi-layer Perceptron Classifier

### Regression Learners
- `LearnerLinearRegression` - Linear Regression
- `LearnerRidge` - Ridge Regression
- `LearnerLasso` - Lasso Regression
- `LearnerElasticNet` - Elastic Net Regression
- `LearnerDecisionTreeRegressor` - Decision Tree Regressor
- `LearnerRandomForestRegressor` - Random Forest Regressor
- `LearnerGradientBoostingRegressor` - Gradient Boosting Regressor
- `LearnerSVR` - Support Vector Regression
- `LearnerKNNRegressor` - K-Nearest Neighbors Regressor
- `LearnerMLPRegressor` - Multi-layer Perceptron Regressor

## Basic Usage

### Quick Start

```python
from mlpy.tasks import TaskClassif
from mlpy.learners import LearnerRandomForest
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resamplings import ResamplingCV
from mlpy import resample

# Create a task
task = TaskClassif(data=your_data, target='target_column')

# Create a learner
learner = LearnerRandomForest(n_estimators=100, max_depth=10)

# Evaluate using cross-validation
result = resample(
    task=task,
    learner=learner,
    resampling=ResamplingCV(folds=5),
    measures=MeasureClassifAccuracy()
)

print(f"Accuracy: {result.aggregate()['classif.acc']['mean']:.3f}")
```

### Training and Prediction

```python
# Train a model
learner = LearnerLogisticRegression(C=1.0)
learner.train(task)

# Make predictions
predictions = learner.predict(new_task)
print(predictions.response)  # Class predictions
print(predictions.prob)      # Probability predictions (if available)
```

## Classification Learners

### Logistic Regression

```python
from mlpy.learners import LearnerLogisticRegression

# Basic usage
learner = LearnerLogisticRegression()

# With parameters
learner = LearnerLogisticRegression(
    penalty='l2',           # Regularization type: 'l1', 'l2', 'elasticnet', 'none'
    C=1.0,                 # Inverse regularization strength
    solver='lbfgs',        # Optimization algorithm
    max_iter=100,          # Maximum iterations
    class_weight='balanced', # Handle imbalanced classes
    random_state=42
)

# For probability predictions
learner = LearnerLogisticRegression(predict_type='prob')
```

### Random Forest

```python
from mlpy.learners import LearnerRandomForest

learner = LearnerRandomForest(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Maximum tree depth
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples in leaf
    max_features='sqrt',   # Features to consider: 'sqrt', 'log2', float
    criterion='gini',      # Split criterion: 'gini', 'entropy'
    class_weight=None,     # Class weights
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)
```

### Support Vector Machine

```python
from mlpy.learners import LearnerSVM

# For classification with probability estimates
learner = LearnerSVM(
    C=1.0,                # Regularization parameter
    kernel='rbf',         # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    degree=3,             # Degree for polynomial kernel
    gamma='scale',        # Kernel coefficient: 'scale', 'auto', float
    probability=True,     # Enable probability estimates
    random_state=42
)

# Note: Set predict_type='prob' to get probability predictions
learner = LearnerSVM(predict_type='prob', probability=True)
```

### Gradient Boosting

```python
from mlpy.learners import LearnerGradientBoosting

learner = LearnerGradientBoosting(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Shrinks contribution of each tree
    max_depth=3,           # Maximum depth of trees
    subsample=1.0,         # Fraction of samples for fitting
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

### K-Nearest Neighbors

```python
from mlpy.learners import LearnerKNN

learner = LearnerKNN(
    n_neighbors=5,         # Number of neighbors
    weights='uniform',     # Weight function: 'uniform', 'distance'
    algorithm='auto',      # Algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',    # Distance metric
    p=2                   # Parameter for Minkowski metric
)
```

### Neural Network

```python
from mlpy.learners import LearnerMLPClassifier

learner = LearnerMLPClassifier(
    hidden_layer_sizes=(100, 50),  # Architecture: 2 hidden layers
    activation='relu',              # Activation: 'identity', 'logistic', 'tanh', 'relu'
    solver='adam',                  # Optimizer: 'lbfgs', 'sgd', 'adam'
    alpha=0.0001,                   # L2 penalty parameter
    learning_rate='constant',       # Learning rate schedule
    max_iter=200,
    random_state=42
)
```

## Regression Learners

### Linear Regression

```python
from mlpy.learners import LearnerLinearRegression

# Basic linear regression
learner = LearnerLinearRegression()

# With parameters
learner = LearnerLinearRegression(
    fit_intercept=True,    # Whether to calculate intercept
    normalize=False,       # Whether to normalize features
    copy_X=True,          # Whether to copy X
    n_jobs=None           # Number of jobs for parallelism
)
```

### Ridge Regression

```python
from mlpy.learners import LearnerRidge

learner = LearnerRidge(
    alpha=1.0,            # Regularization strength
    fit_intercept=True,
    normalize=False,
    solver='auto',        # Solver: 'auto', 'svd', 'cholesky', 'lsqr', etc.
    random_state=42
)
```

### Lasso Regression

```python
from mlpy.learners import LearnerLasso

learner = LearnerLasso(
    alpha=1.0,            # Regularization strength
    fit_intercept=True,
    normalize=False,
    max_iter=1000,        # Maximum iterations
    selection='cyclic',   # Feature selection: 'cyclic', 'random'
    random_state=42
)
```

### Elastic Net

```python
from mlpy.learners import LearnerElasticNet

learner = LearnerElasticNet(
    alpha=1.0,            # Regularization strength
    l1_ratio=0.5,         # Mix between L1 and L2 (0=Ridge, 1=Lasso)
    fit_intercept=True,
    max_iter=1000,
    selection='cyclic',
    random_state=42
)
```

### Random Forest Regressor

```python
from mlpy.learners import LearnerRandomForestRegressor

learner = LearnerRandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,     # For regression: int, float, 'auto', 'sqrt', 'log2'
    criterion='squared_error',  # 'squared_error', 'absolute_error', 'poisson'
    random_state=42,
    n_jobs=-1
)
```

## Auto-Wrapper

The auto-wrapper allows you to wrap any scikit-learn compatible estimator:

### Basic Auto-Wrapping

```python
from mlpy.learners import auto_sklearn
from sklearn.ensemble import ExtraTreesClassifier

# Wrap a scikit-learn class
learner = auto_sklearn(ExtraTreesClassifier, n_estimators=100)

# Wrap an instance
clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
learner = auto_sklearn(clf)

# Specify task type explicitly
from sklearn.isotonic import IsotonicRegression
learner = auto_sklearn(IsotonicRegression, task_type='regr')
```

### List Available Models

```python
from mlpy.learners.sklearn.auto_wrap import list_available_sklearn_models

models = list_available_sklearn_models()
print(f"Available classifiers: {models['classifiers']}")
print(f"Available regressors: {models['regressors']}")
```

## Advanced Usage

### Parameter Tuning

```python
from mlpy.automl import ParamSet, ParamInt, ParamFloat, TunerGrid

# Define parameter space
param_set = ParamSet([
    ParamInt("n_estimators", lower=50, upper=200),
    ParamInt("max_depth", lower=3, upper=20),
    ParamFloat("min_samples_split", lower=0.01, upper=0.1)
])

# Tune hyperparameters
tuner = TunerGrid(resolution=5)
result = tuner.tune(
    learner=LearnerRandomForest(),
    task=task,
    resampling=ResamplingCV(folds=3),
    measure=MeasureClassifAccuracy(),
    param_set=param_set
)

print(f"Best parameters: {result.best_params}")
```

### Using in Pipelines

```python
from mlpy.pipelines import PipeOpScale, PipeOpImpute, PipeOpLearner, linear_pipeline

# Create preprocessing pipeline with sklearn learner
pipeline = linear_pipeline([
    PipeOpImpute(method='median'),
    PipeOpScale(method='standard'),
    PipeOpLearner(LearnerRandomForest(n_estimators=100))
])

# Use like any MLPY learner
result = resample(task, pipeline, ResamplingCV(folds=5), measures)
```

### Model Persistence

```python
# Train a model
learner = LearnerRandomForest()
learner.train(task)

# Access the underlying sklearn model
sklearn_model = learner.model

# Save using joblib (recommended for sklearn)
import joblib
joblib.dump(sklearn_model, 'model.pkl')

# Load and use
loaded_model = joblib.load('model.pkl')
learner_loaded = LearnerRandomForest()
learner_loaded.model = loaded_model
learner_loaded.is_trained = True
```

### Custom Preprocessing

```python
# Sklearn models expect numpy arrays, but MLPY handles conversion
# You can still do custom preprocessing:

from sklearn.preprocessing import StandardScaler

# Get data from task
X = task.X.values  # Convert to numpy if needed
y = task.y.values

# Apply sklearn preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create new task with preprocessed data
import pandas as pd
data_scaled = pd.DataFrame(X_scaled, columns=task.feature_names)
data_scaled[task.target] = y
task_scaled = TaskClassif(data=data_scaled, target=task.target)
```

## Best Practices

### 1. Choose the Right Learner

- **Linear models** (Logistic/Linear Regression): Good baseline, interpretable, fast
- **Tree-based** (RF, GB): Handle non-linearity, feature interactions, robust
- **SVM**: Good for high-dimensional data, various kernels for non-linearity
- **Neural Networks**: Complex patterns, large datasets, requires tuning

### 2. Handle Class Imbalance

```python
# Use class weights
learner = LearnerRandomForest(class_weight='balanced')

# Or specify custom weights
learner = LearnerLogisticRegression(class_weight={0: 1, 1: 10})
```

### 3. Feature Scaling

Some models require feature scaling:

```python
# Models that need scaling: SVM, Neural Networks, KNN
pipeline = linear_pipeline([
    PipeOpScale(method='standard'),
    PipeOpLearner(LearnerSVM())
])

# Models that don't need scaling: Tree-based models
learner = LearnerRandomForest()  # No scaling needed
```

### 4. Probability Calibration

```python
# Get calibrated probabilities
learner = LearnerSVM(predict_type='prob', probability=True)

# For better probability estimates with SVM
from sklearn.calibration import CalibratedClassifierCV
base_svm = learner.model  # After training
calibrated = CalibratedClassifierCV(base_svm, cv=3)
```

### 5. Memory and Performance

```python
# Use parallel processing
learner = LearnerRandomForest(n_jobs=-1)  # Use all cores

# Reduce memory usage
learner = LearnerRandomForest(
    max_depth=10,  # Limit tree depth
    max_samples=0.8  # Subsample data
)
```

### 6. Debugging

```python
# Enable verbose output
learner = LearnerGradientBoosting(verbose=1)

# Check feature importances (tree-based models)
learner.train(task)
importances = learner.model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': task.feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
```

## Common Issues and Solutions

### Issue: "ImportError: scikit-learn is required"
**Solution**: Install scikit-learn: `pip install scikit-learn`

### Issue: "ValueError: Unknown label type"
**Solution**: Ensure your target variable is properly encoded for classification

### Issue: Poor probability estimates
**Solution**: Use `predict_type='prob'` and ensure the model supports probabilities

### Issue: Memory errors with large datasets
**Solution**: Use `n_jobs=1` to reduce memory usage, or subsample your data

### Issue: Slow training
**Solution**: 
- Use fewer iterations/estimators
- Enable parallel processing (`n_jobs=-1`)
- Consider simpler models for baseline

## Examples

Complete examples using sklearn learners can be found in:
- `examples/sklearn_integration.py` - Basic sklearn usage
- `examples/automl_example.py` - AutoML with sklearn models
- `examples/automl_advanced.py` - Advanced techniques

## Summary

MLPY's scikit-learn integration provides:
- Pre-configured wrappers for common sklearn models
- Consistent API across all learners
- Automatic parameter validation
- Integration with MLPY's ecosystem (pipelines, tuning, etc.)
- Flexible auto-wrapper for any sklearn estimator

This allows you to leverage the power of scikit-learn within MLPY's unified framework while maintaining full compatibility with all MLPY features.