# MLPY Scikit-learn Quick Reference

## Classification Learners

| Learner | Import | Key Parameters | When to Use |
|---------|---------|----------------|-------------|
| **Logistic Regression** | `LearnerLogisticRegression` | `C`, `penalty`, `solver` | Linear relationships, baseline model, interpretability needed |
| **Decision Tree** | `LearnerDecisionTree` | `max_depth`, `min_samples_split` | Non-linear patterns, interpretable rules, feature importance |
| **Random Forest** | `LearnerRandomForest` | `n_estimators`, `max_depth`, `max_features` | General purpose, robust, handles mixed features well |
| **Gradient Boosting** | `LearnerGradientBoosting` | `n_estimators`, `learning_rate`, `max_depth` | High accuracy, competitions, careful tuning needed |
| **SVM** | `LearnerSVM` | `C`, `kernel`, `gamma` | High-dimensional data, non-linear boundaries with kernels |
| **K-NN** | `LearnerKNN` | `n_neighbors`, `weights` | Local patterns, non-parametric, simple baseline |
| **Naive Bayes** | `LearnerNaiveBayes` | `var_smoothing` | Text classification, fast training, probabilistic |
| **Neural Network** | `LearnerMLPClassifier` | `hidden_layer_sizes`, `activation`, `solver` | Complex patterns, large datasets, requires scaling |

## Regression Learners

| Learner | Import | Key Parameters | When to Use |
|---------|---------|----------------|-------------|
| **Linear Regression** | `LearnerLinearRegression` | `fit_intercept` | Linear relationships, baseline, interpretability |
| **Ridge** | `LearnerRidge` | `alpha` | Linear with L2 regularization, multicollinearity |
| **Lasso** | `LearnerLasso` | `alpha` | Linear with L1 regularization, feature selection |
| **ElasticNet** | `LearnerElasticNet` | `alpha`, `l1_ratio` | Balance between Ridge and Lasso |
| **Decision Tree** | `LearnerDecisionTreeRegressor` | `max_depth`, `min_samples_split` | Non-linear patterns, interactions |
| **Random Forest** | `LearnerRandomForestRegressor` | `n_estimators`, `max_depth` | General purpose, robust to outliers |
| **Gradient Boosting** | `LearnerGradientBoostingRegressor` | `n_estimators`, `learning_rate` | High accuracy, careful tuning |
| **SVR** | `LearnerSVR` | `C`, `kernel`, `epsilon` | Non-linear patterns, robust to outliers |
| **K-NN** | `LearnerKNNRegressor` | `n_neighbors`, `weights` | Local patterns, non-parametric |
| **Neural Network** | `LearnerMLPRegressor` | `hidden_layer_sizes`, `activation` | Complex non-linear patterns |

## Common Usage Patterns

### Basic Classification
```python
from mlpy.learners import LearnerRandomForest
from mlpy.tasks import TaskClassif
from mlpy import resample

# Quick start
learner = LearnerRandomForest(n_estimators=100)
result = resample(task, learner)
```

### Basic Regression
```python
from mlpy.learners import LearnerRidge

learner = LearnerRidge(alpha=1.0)
learner.train(task)
predictions = learner.predict(test_task)
```

### With Preprocessing Pipeline
```python
from mlpy.pipelines import linear_pipeline, PipeOpScale, PipeOpLearner

pipeline = linear_pipeline([
    PipeOpScale(method='standard'),
    PipeOpLearner(LearnerSVM(C=1.0))
])
```

### Hyperparameter Tuning
```python
from mlpy.automl import ParamSet, ParamInt, ParamFloat, TunerRandom

params = ParamSet([
    ParamInt("n_estimators", 50, 200),
    ParamInt("max_depth", 3, 20)
])

tuner = TunerRandom(n_evals=20)
best = tuner.tune(learner, task, params)
```

### Auto-wrap Any Sklearn Model
```python
from mlpy.learners import auto_sklearn
from sklearn.ensemble import ExtraTreesClassifier

learner = auto_sklearn(ExtraTreesClassifier, n_estimators=100)
```

## Parameter Guidelines

### Regularization Parameters
- **C** (SVM, Logistic): Higher = less regularization (0.01 to 100)
- **alpha** (Ridge, Lasso): Higher = more regularization (0.001 to 10)

### Tree Parameters
- **n_estimators**: More = better but slower (50-500)
- **max_depth**: Deeper = more complex (3-20, None for no limit)
- **min_samples_split**: Higher = simpler trees (2-20)

### Neural Network Parameters
- **hidden_layer_sizes**: (100,) for one layer, (100, 50) for two
- **activation**: 'relu' (default), 'tanh', 'logistic'
- **solver**: 'adam' (default), 'lbfgs' for small data

## Quick Decision Guide

**Need interpretability?**
→ Logistic/Linear Regression, Decision Tree

**Have lots of features?**
→ Lasso (with selection), SVM, Random Forest

**Non-linear patterns?**
→ Random Forest, SVM with RBF kernel, Neural Networks

**Need probability estimates?**
→ Logistic Regression, Random Forest, Neural Networks

**Limited training time?**
→ Logistic/Linear Regression, Naive Bayes

**Want best accuracy?**
→ Gradient Boosting, Random Forest, Neural Networks (with tuning)

**Imbalanced classes?**
→ Use `class_weight='balanced'` parameter

**Large dataset?**
→ Use `n_jobs=-1` for parallel processing