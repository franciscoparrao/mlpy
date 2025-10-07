"""
Simple test script for native learners in MLPY.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Import MLPY components
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrRMSE
from mlpy.benchmark import benchmark
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy.learners.native import (
    LearnerDecisionTree, LearnerDecisionTreeRegressor,
    LearnerLinearRegression, LearnerLogisticRegression,
    LearnerKNN, LearnerKNNRegressor,
    LearnerNaiveBayesGaussian
)


def test_native_learners():
    """Test all native learners."""
    print("=" * 80)
    print("TESTING NATIVE LEARNERS")
    print("=" * 80)
    
    # Create classification dataset
    print("\n1. Creating classification dataset...")
    X_classif, y_classif = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    data_classif = pd.DataFrame(X_classif, columns=[f"feat_{i}" for i in range(10)])
    data_classif['target'] = y_classif
    
    task_classif = TaskClassif(
        id="test_classif",
        data=data_classif,
        target="target"
    )
    
    # Create regression dataset
    print("2. Creating regression dataset...")
    X_regr, y_regr = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )
    
    data_regr = pd.DataFrame(X_regr, columns=[f"feat_{i}" for i in range(10)])
    data_regr['target'] = y_regr
    
    task_regr = TaskRegr(
        id="test_regr",
        data=data_regr,
        target="target"
    )
    
    # Test classification learners
    print("\n3. Testing Classification Learners...")
    learners_classif = [
        LearnerDecisionTree(id="decision_tree", max_depth=5),
        LearnerLogisticRegression(id="logistic_regression", n_iterations=200),
        LearnerKNN(id="knn", n_neighbors=5),
        LearnerNaiveBayesGaussian(id="naive_bayes")
    ]
    
    # Simple train/predict test
    train_ids = list(range(150))
    test_ids = list(range(150, 200))
    
    for learner in learners_classif:
        try:
            # Train
            learner.train(task_classif, row_ids=train_ids)
            
            # Predict
            pred = learner.predict(task_classif, row_ids=test_ids)
            
            # Calculate accuracy
            accuracy = MeasureClassifAccuracy()
            score = accuracy.score(pred)
            
            print(f"   {learner.id}: Accuracy = {score:.4f}")
            
        except Exception as e:
            print(f"   {learner.id}: FAILED - {str(e)}")
    
    # Test with benchmark
    print("\n4. Running Classification Benchmark (3-fold CV)...")
    bench_classif = benchmark(
        tasks=[task_classif],
        learners=learners_classif,
        resampling=ResamplingCV(folds=3, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )
    
    results = bench_classif.aggregate()
    print("\nClassification Results:")
    print(results[['learner_id', 'mean_test_Accuracy', 'std_test_Accuracy']])
    
    # Test regression learners
    print("\n5. Testing Regression Learners...")
    learners_regr = [
        LearnerDecisionTreeRegressor(id="decision_tree_regr", max_depth=5),
        LearnerLinearRegression(id="linear_regression", method='normal'),
        LearnerLinearRegression(id="linear_regression_gd", method='gradient_descent', n_iterations=500),
        LearnerKNNRegressor(id="knn_regr", n_neighbors=5)
    ]
    
    for learner in learners_regr:
        try:
            # Train
            learner.train(task_regr, row_ids=train_ids)
            
            # Predict
            pred = learner.predict(task_regr, row_ids=test_ids)
            
            # Calculate RMSE
            rmse = MeasureRegrRMSE()
            score = rmse.score(pred)
            
            print(f"   {learner.id}: RMSE = {score:.4f}")
            
        except Exception as e:
            print(f"   {learner.id}: FAILED - {str(e)}")
    
    # Test with benchmark
    print("\n6. Running Regression Benchmark (3-fold CV)...")
    bench_regr = benchmark(
        tasks=[task_regr],
        learners=learners_regr,
        resampling=ResamplingCV(folds=3),
        measures=[MeasureRegrRMSE()]
    )
    
    results = bench_regr.aggregate()
    print("\nRegression Results:")
    print(results[['learner_id', 'mean_test_RMSE', 'std_test_RMSE']])
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    test_native_learners()