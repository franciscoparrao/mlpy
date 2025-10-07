"""Example: Using scikit-learn models with MLPY.

This example demonstrates how to use scikit-learn models within the MLPY framework,
including direct wrappers and the auto-wrapper functionality.
"""

import numpy as np
import pandas as pd
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrRMSE
from mlpy.resamplings import ResamplingCV
from mlpy import resample, benchmark

# Try importing sklearn wrappers
try:
    from mlpy.learners import (
        LearnerLogisticRegression,
        LearnerRandomForest,
        LearnerSVM,
        LearnerLinearRegression,
        LearnerRidge,
        auto_sklearn
    )
    HAS_SKLEARN = True
except ImportError:
    print("Scikit-learn wrappers not available. Install scikit-learn.")
    HAS_SKLEARN = False


def example_classification():
    """Example using sklearn classifiers with MLPY."""
    print("=== Classification Example ===\n")
    
    # Create a classification dataset
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Create a simple linear decision boundary
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(int)
    y = ['Class_' + str(i) for i in y]
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    # Create task
    task = TaskClassif(data=data, target='target', id='binary_classification')
    print(f"Task: {task.id}")
    print(f"Features: {task.n_features}")
    print(f"Observations: {task.n_obs}")
    print(f"Classes: {task.class_names}\n")
    
    # Create learners
    learners = [
        LearnerLogisticRegression(id='logreg', C=1.0),
        LearnerRandomForest(id='rf', n_estimators=50, max_depth=5, random_state=42),
        LearnerSVM(id='svm', kernel='rbf', C=1.0, probability=True)
    ]
    
    # Evaluate each learner
    for learner in learners:
        print(f"\nEvaluating {learner.id}...")
        
        # Resample with cross-validation
        result = resample(
            task=task,
            learner=learner,
            resampling=ResamplingCV(folds=5),
            measures=MeasureClassifAccuracy()
        )
        
        print(f"Accuracy: {result.aggregate()['classif.acc']['mean']:.3f} "
              f"(+/- {result.aggregate()['classif.acc']['sd']:.3f})")
        
    # Benchmark comparison
    print("\n=== Benchmark Comparison ===")
    benchmark_result = benchmark(
        tasks=[task],
        learners=learners,
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy()
    )
    
    print("\n" + str(benchmark_result))


def example_regression():
    """Example using sklearn regressors with MLPY."""
    print("\n\n=== Regression Example ===\n")
    
    # Create a regression dataset
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # True function: y = 2*x1 - 3*x2 + 0.5*x3 + noise
    y = 2 * X[:, 0] - 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.5
    
    # Create DataFrame
    feature_names = [f'x{i}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['y'] = y
    
    # Create task
    task = TaskRegr(data=data, target='y', id='regression_task')
    print(f"Task: {task.id}")
    print(f"Features: {task.n_features}")
    print(f"Observations: {task.n_obs}\n")
    
    # Create learners
    learners = [
        LearnerLinearRegression(id='linear'),
        LearnerRidge(id='ridge', alpha=1.0),
        LearnerRidge(id='ridge_0.1', alpha=0.1)
    ]
    
    # Evaluate
    for learner in learners:
        print(f"\nEvaluating {learner.id}...")
        
        result = resample(
            task=task,
            learner=learner,
            resampling=ResamplingCV(folds=5),
            measures=MeasureRegrRMSE()
        )
        
        print(f"RMSE: {result.aggregate()['regr.rmse']['mean']:.3f} "
              f"(+/- {result.aggregate()['regr.rmse']['sd']:.3f})")


def example_auto_wrapper():
    """Example using the auto-wrapper functionality."""
    print("\n\n=== Auto-Wrapper Example ===\n")
    
    # Import additional sklearn models
    try:
        from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesRegressor
        from sklearn.linear_model import ElasticNet
    except ImportError:
        print("Scikit-learn not installed. Skipping auto-wrapper example.")
        return
        
    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(str)
    
    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
    data['y'] = y
    task = TaskClassif(data=data, target='y')
    
    # Auto-wrap a scikit-learn model not explicitly wrapped
    print("Auto-wrapping GradientBoostingClassifier...")
    learner = auto_sklearn(
        GradientBoostingClassifier,
        id='gb_auto',
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Use it like any MLPY learner
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=3),
        measures=MeasureClassifAccuracy()
    )
    
    print(f"Accuracy: {result.aggregate()['classif.acc']['mean']:.3f}")
    
    # Auto-wrap regression model
    print("\nAuto-wrapping ElasticNet...")
    X_reg = np.random.randn(100, 5)
    y_reg = X_reg[:, 0] + 0.5 * X_reg[:, 1] + np.random.randn(100) * 0.1
    
    data_reg = pd.DataFrame(X_reg, columns=[f'x{i}' for i in range(5)])
    data_reg['y'] = y_reg
    task_reg = TaskRegr(data=data_reg, target='y')
    
    # Auto-wrap with instance
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    learner_reg = auto_sklearn(elastic_net, id='elastic_auto')
    
    result_reg = resample(
        task=task_reg,
        learner=learner_reg,
        resampling=ResamplingCV(folds=3),
        measures=MeasureRegrRMSE()
    )
    
    print(f"RMSE: {result_reg.aggregate()['regr.rmse']['mean']:.3f}")
    
    # List available models
    from mlpy.learners.sklearn.auto_wrap import list_available_sklearn_models
    models = list_available_sklearn_models()
    print(f"\nAvailable sklearn models:")
    print(f"Classifiers: {len(models['classifiers'])}")
    print(f"Regressors: {len(models['regressors'])}")


def example_pipeline_with_sklearn():
    """Example using sklearn models in MLPY pipelines."""
    print("\n\n=== Pipeline with Sklearn Models ===\n")
    
    try:
        from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline
    except ImportError:
        print("Pipeline functionality not available.")
        return
        
    # Create dataset with different scales
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 0] *= 100  # Large scale
    X[:, 2] *= 0.01  # Small scale
    y = (X[:, 0] / 100 + X[:, 1] - X[:, 2] * 100 > 0).astype(str)
    
    data = pd.DataFrame(X, columns=['large_feature', 'normal_feature', 'small_feature'])
    data['y'] = y
    task = TaskClassif(data=data, target='y')
    
    # Create pipeline: Scale -> Logistic Regression
    print("Creating pipeline: Scaler -> Logistic Regression")
    
    scaler = PipeOpScale()
    learner = PipeOpLearner(LearnerLogisticRegression(C=1.0))
    
    pipeline = linear_pipeline([scaler, learner], id='scaled_logreg')
    
    # Evaluate pipeline
    result = resample(
        task=task,
        learner=pipeline,
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy()
    )
    
    print(f"Pipeline accuracy: {result.aggregate()['classif.acc']['mean']:.3f}")
    
    # Compare with unscaled
    print("\nComparing with unscaled model...")
    result_unscaled = resample(
        task=task,
        learner=LearnerLogisticRegression(C=1.0),
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy()
    )
    
    print(f"Unscaled accuracy: {result_unscaled.aggregate()['classif.acc']['mean']:.3f}")
    print("(Scaling helps when features have very different scales)")


if __name__ == "__main__":
    if not HAS_SKLEARN:
        print("This example requires scikit-learn to be installed.")
        print("Install with: pip install scikit-learn")
    else:
        example_classification()
        example_regression()
        example_auto_wrapper()
        example_pipeline_with_sklearn()