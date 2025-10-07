"""
Test script for native learners in MLPY.

This script tests all native learners (implemented from scratch)
on both classification and regression tasks.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import MLPY components
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifAUC, MeasureRegrRMSE, MeasureRegrMAE
from mlpy.resamplings import ResamplingHoldout, ResamplingCV
from mlpy.learners.native import (
    LearnerDecisionTree, LearnerDecisionTreeRegressor,
    LearnerLinearRegression, LearnerLogisticRegression,
    LearnerKNN, LearnerKNNRegressor,
    LearnerNaiveBayesGaussian
)

def test_classification_learners():
    """Test all native classification learners."""
    print("=" * 80)
    print("TESTING NATIVE CLASSIFICATION LEARNERS")
    print("=" * 80)
    
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_cols)
    data['target'] = y
    
    # Create task
    task = TaskClassif(
        id="test_classification",
        data=data,
        target="target"
    )
    
    # Define learners to test
    learners = [
        LearnerDecisionTree(max_depth=5),
        LearnerLogisticRegression(n_iterations=500, learning_rate=0.1),
        LearnerKNN(n_neighbors=5),
        LearnerNaiveBayesGaussian()
    ]
    
    # Test each learner
    for learner in learners:
        print(f"\nTesting {learner.id}...")
        
        try:
            # Train-test split
            train_ids = list(range(400))
            test_ids = list(range(400, 500))
            
            # Train
            learner.train(task, row_ids=train_ids)
            
            # Predict (response)
            pred_response = learner.predict(task, row_ids=test_ids)
            
            # Calculate accuracy
            accuracy = MeasureClassifAccuracy()
            acc_score = accuracy.score(pred_response)
            print(f"  Accuracy: {acc_score:.4f}")
            
            # Predict (probabilities)
            learner.predict_type = "prob"
            pred_prob = learner.predict(task, row_ids=test_ids)
            
            # For multiclass, we can't use AUC directly
            # Just check that probabilities sum to 1
            prob_sums = np.sum(pred_prob.prob, axis=1)
            print(f"  Probability check (should be ~1.0): {np.mean(prob_sums):.4f}")
            
            # Test with cross-validation
            cv = ResamplingCV(folds=5)
            cv_results = cv.evaluate(learner, task, [accuracy])
            cv_mean = cv_results.aggregate().mean()
            print(f"  5-fold CV Accuracy: {cv_mean['mean_test_Accuracy']:.4f}")
            
            print(f"  [OK] {learner.id} passed all tests!")
            
        except Exception as e:
            print(f"  [FAIL] {learner.id} failed: {str(e)}")
            import traceback
            traceback.print_exc()


def test_regression_learners():
    """Test all native regression learners."""
    print("\n" + "=" * 80)
    print("TESTING NATIVE REGRESSION LEARNERS")
    print("=" * 80)
    
    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_cols)
    data['target'] = y
    
    # Create task
    task = TaskRegr(
        id="test_regression",
        data=data,
        target="target"
    )
    
    # Define learners to test
    learners = [
        LearnerDecisionTreeRegressor(max_depth=5),
        LearnerLinearRegression(method='normal'),
        LearnerLinearRegression(method='gradient_descent', n_iterations=1000, learning_rate=0.01),
        LearnerKNNRegressor(n_neighbors=5)
    ]
    
    # Test each learner
    for learner in learners:
        print(f"\nTesting {learner.id}...")
        
        try:
            # Train-test split
            train_ids = list(range(400))
            test_ids = list(range(400, 500))
            
            # Train
            learner.train(task, row_ids=train_ids)
            
            # Predict
            pred = learner.predict(task, row_ids=test_ids)
            
            # Calculate metrics
            rmse = MeasureRegrRMSE()
            mae = MeasureRegrMAE()
            
            rmse_score = rmse.score(pred)
            mae_score = mae.score(pred)
            
            print(f"  RMSE: {rmse_score:.4f}")
            print(f"  MAE: {mae_score:.4f}")
            
            # Test standard error prediction if supported
            if "se" in learner.predict_types:
                learner.predict_type = "se"
                pred_se = learner.predict(task, row_ids=test_ids)
                mean_se = np.mean(pred_se.se)
                print(f"  Mean Standard Error: {mean_se:.4f}")
            
            # Test with holdout resampling
            holdout = ResamplingHoldout(ratio=0.8)
            holdout_results = holdout.evaluate(learner, task, [rmse, mae])
            results_df = holdout_results.aggregate()
            print(f"  Holdout RMSE: {results_df['test_RMSE'].iloc[0]:.4f}")
            print(f"  Holdout MAE: {results_df['test_MAE'].iloc[0]:.4f}")
            
            print(f"  [OK] {learner.id} passed all tests!")
            
        except Exception as e:
            print(f"  [FAIL] {learner.id} failed: {str(e)}")
            import traceback
            traceback.print_exc()


def test_real_datasets():
    """Test native learners on real datasets."""
    print("\n" + "=" * 80)
    print("TESTING ON REAL DATASETS")
    print("=" * 80)
    
    # Test on Iris (classification)
    print("\nIris Dataset (Classification):")
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_data['target'] = iris.target
    
    task_iris = TaskClassif(
        id="iris",
        data=iris_data,
        target="target"
    )
    
    learners_classif = [
        LearnerDecisionTree(max_depth=3),
        LearnerLogisticRegression(n_iterations=200),
        LearnerKNN(n_neighbors=3),
        LearnerNaiveBayesGaussian()
    ]
    
    accuracy = MeasureClassifAccuracy()
    
    for learner in learners_classif:
        cv = ResamplingCV(folds=5)
        results = cv.evaluate(learner, task_iris, [accuracy])
        mean_acc = results.aggregate()['mean_test_Accuracy'].iloc[0]
        print(f"  {learner.id}: {mean_acc:.4f}")
    
    # Test on Diabetes (regression)
    print("\nDiabetes Dataset (Regression):")
    diabetes = load_diabetes()
    diabetes_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_data['target'] = diabetes.target
    
    task_diabetes = TaskRegr(
        id="diabetes",
        data=diabetes_data,
        target="target"
    )
    
    learners_regr = [
        LearnerDecisionTreeRegressor(max_depth=5),
        LearnerLinearRegression(),
        LearnerKNNRegressor(n_neighbors=10)
    ]
    
    rmse = MeasureRegrRMSE()
    
    for learner in learners_regr:
        cv = ResamplingCV(folds=5)
        results = cv.evaluate(learner, task_diabetes, [rmse])
        mean_rmse = results.aggregate()['mean_test_RMSE'].iloc[0]
        print(f"  {learner.id}: {mean_rmse:.4f}")


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)
    
    # Test with small dataset
    print("\nSmall dataset (10 samples):")
    X_small = np.random.randn(10, 3)
    y_small = np.random.randint(0, 2, 10)
    
    small_data = pd.DataFrame(X_small, columns=['f1', 'f2', 'f3'])
    small_data['target'] = y_small
    
    task_small = TaskClassif(
        id="small",
        data=small_data,
        target="target"
    )
    
    # KNN with more neighbors than samples
    knn = LearnerKNN(n_neighbors=15)  # More than 10 samples
    knn.train(task_small)
    pred = knn.predict(task_small)
    print(f"  KNN with k > n_samples: OK")
    
    # Test with perfect separation
    print("\nPerfectly separable data:")
    X_sep = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_sep = np.array([0, 0, 1, 1])
    
    sep_data = pd.DataFrame(X_sep, columns=['x1', 'x2'])
    sep_data['target'] = y_sep
    
    task_sep = TaskClassif(
        id="separable",
        data=sep_data,
        target="target"
    )
    
    # All learners should achieve perfect accuracy
    learners = [
        LearnerDecisionTree(max_depth=1),
        LearnerLogisticRegression(n_iterations=100),
        LearnerKNN(n_neighbors=1),
        LearnerNaiveBayesGaussian()
    ]
    
    accuracy = MeasureClassifAccuracy()
    
    for learner in learners:
        learner.train(task_sep)
        pred = learner.predict(task_sep)
        acc = accuracy.score(pred)
        print(f"  {learner.id}: {acc:.2f} (should be 1.0)")


if __name__ == "__main__":
    # Run all tests
    test_classification_learners()
    test_regression_learners()
    test_real_datasets()
    test_edge_cases()
    
    print("\n" + "=" * 80)
    print("ALL NATIVE LEARNER TESTS COMPLETED!")
    print("=" * 80)