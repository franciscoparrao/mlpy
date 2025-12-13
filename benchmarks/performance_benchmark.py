"""
Performance Benchmark: MLPY vs scikit-learn

Compares training time, prediction time, and memory usage
on standard UCI datasets.

Run with: python benchmarks/performance_benchmark.py
"""

import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import tracemalloc

# Import MLPY
import sys
sys.path.insert(0, '.')
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners.sklearn import (
    LearnerRandomForestClassifier,
    LearnerGradientBoostingClassifier,
    LearnerLogisticRegression,
    LearnerRandomForestRegressor,
    LearnerLinearRegression
)
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrRMSE
from mlpy.resamplings import ResamplingCV
from mlpy import resample


def benchmark_classification_sklearn(X, y, model, n_folds=5):
    """Benchmark sklearn classification"""
    # Training time
    tracemalloc.start()
    start = time.time()
    model.fit(X, y)
    train_time = time.time() - start
    train_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
    tracemalloc.stop()

    # Prediction time
    start = time.time()
    predictions = model.predict(X)
    predict_time = time.time() - start

    # Accuracy
    accuracy = accuracy_score(y, predictions)

    # Cross-validation time
    start = time.time()
    cv_scores = cross_val_score(model, X, y, cv=n_folds)
    cv_time = time.time() - start
    cv_accuracy = cv_scores.mean()

    return {
        'train_time': train_time,
        'predict_time': predict_time,
        'train_memory_mb': train_memory,
        'accuracy': accuracy,
        'cv_time': cv_time,
        'cv_accuracy': cv_accuracy
    }


def benchmark_classification_mlpy(X, y, learner, n_folds=5):
    """Benchmark MLPY classification"""
    df = pd.DataFrame(X)
    df['target'] = y
    task = TaskClassif(data=df, target='target')

    # Training time
    tracemalloc.start()
    start = time.time()
    learner.train(task)
    train_time = time.time() - start
    train_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
    tracemalloc.stop()

    # Prediction time
    start = time.time()
    predictions = learner.predict(task)
    predict_time = time.time() - start

    # Accuracy
    measure = MeasureClassifAccuracy()
    accuracy = measure.score(predictions.truth, predictions.response)

    # Cross-validation time
    start = time.time()
    result = resample(
        task=task,
        learner=learner.__class__(),  # Fresh instance
        resampling=ResamplingCV(folds=n_folds),
        measures=[measure]
    )
    cv_time = time.time() - start
    cv_accuracy = result.score('classif.acc', average='mean')

    return {
        'train_time': train_time,
        'predict_time': predict_time,
        'train_memory_mb': train_memory,
        'accuracy': accuracy,
        'cv_time': cv_time,
        'cv_accuracy': cv_accuracy
    }


def benchmark_regression_sklearn(X, y, model, n_folds=5):
    """Benchmark sklearn regression"""
    # Training time
    tracemalloc.start()
    start = time.time()
    model.fit(X, y)
    train_time = time.time() - start
    train_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
    tracemalloc.stop()

    # Prediction time
    start = time.time()
    predictions = model.predict(X)
    predict_time = time.time() - start

    # RMSE
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y, predictions))

    # Cross-validation time
    start = time.time()
    cv_scores = cross_val_score(model, X, y, cv=n_folds,
                                 scoring='neg_mean_squared_error')
    cv_time = time.time() - start
    cv_rmse = np.sqrt(-cv_scores.mean())

    return {
        'train_time': train_time,
        'predict_time': predict_time,
        'train_memory_mb': train_memory,
        'rmse': rmse,
        'cv_time': cv_time,
        'cv_rmse': cv_rmse
    }


def benchmark_regression_mlpy(X, y, learner, n_folds=5):
    """Benchmark MLPY regression"""
    df = pd.DataFrame(X)
    df['target'] = y
    task = TaskRegr(data=df, target='target')

    # Training time
    tracemalloc.start()
    start = time.time()
    learner.train(task)
    train_time = time.time() - start
    train_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
    tracemalloc.stop()

    # Prediction time
    start = time.time()
    predictions = learner.predict(task)
    predict_time = time.time() - start

    # RMSE
    measure = MeasureRegrRMSE()
    rmse = measure.score(predictions.truth, predictions.response)

    # Cross-validation time
    start = time.time()
    result = resample(
        task=task,
        learner=learner.__class__(),  # Fresh instance
        resampling=ResamplingCV(folds=n_folds),
        measures=[measure]
    )
    cv_time = time.time() - start
    cv_rmse = result.score('regr.rmse', average='mean')

    return {
        'train_time': train_time,
        'predict_time': predict_time,
        'train_memory_mb': train_memory,
        'rmse': rmse,
        'cv_time': cv_time,
        'cv_rmse': cv_rmse
    }


def run_benchmarks():
    """Run all benchmarks"""
    results = []

    print("=" * 80)
    print("MLPY vs scikit-learn Performance Benchmark")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. Iris Dataset (150 samples, 4 features, 3 classes)
    # ========================================================================
    print("1. Iris Dataset (150 samples, 4 features, 3 classes)")
    print("-" * 80)
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Random Forest
    print("  - Random Forest...")
    sklearn_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlpy_rf = LearnerRandomForestClassifier(n_estimators=100, random_state=42)

    sklearn_result = benchmark_classification_sklearn(X, y, sklearn_rf)
    mlpy_result = benchmark_classification_mlpy(X, y, mlpy_rf)

    results.append({
        'dataset': 'Iris',
        'n_samples': 150,
        'n_features': 4,
        'task': 'Classification',
        'algorithm': 'Random Forest',
        'framework': 'sklearn',
        **sklearn_result
    })
    results.append({
        'dataset': 'Iris',
        'n_samples': 150,
        'n_features': 4,
        'task': 'Classification',
        'algorithm': 'Random Forest',
        'framework': 'MLPY',
        **mlpy_result
    })

    print(f"    sklearn: train={sklearn_result['train_time']:.4f}s, "
          f"predict={sklearn_result['predict_time']:.4f}s, "
          f"cv_acc={sklearn_result['cv_accuracy']:.4f}")
    print(f"    MLPY:    train={mlpy_result['train_time']:.4f}s, "
          f"predict={mlpy_result['predict_time']:.4f}s, "
          f"cv_acc={mlpy_result['cv_accuracy']:.4f}")
    print()

    # Logistic Regression
    print("  - Logistic Regression...")
    sklearn_lr = LogisticRegression(max_iter=1000, random_state=42)
    mlpy_lr = LearnerLogisticRegression(max_iter=1000, random_state=42)

    sklearn_result = benchmark_classification_sklearn(X, y, sklearn_lr)
    mlpy_result = benchmark_classification_mlpy(X, y, mlpy_lr)

    results.append({
        'dataset': 'Iris',
        'n_samples': 150,
        'n_features': 4,
        'task': 'Classification',
        'algorithm': 'Logistic Regression',
        'framework': 'sklearn',
        **sklearn_result
    })
    results.append({
        'dataset': 'Iris',
        'n_samples': 150,
        'n_features': 4,
        'task': 'Classification',
        'algorithm': 'Logistic Regression',
        'framework': 'MLPY',
        **mlpy_result
    })

    print(f"    sklearn: train={sklearn_result['train_time']:.4f}s, "
          f"predict={sklearn_result['predict_time']:.4f}s, "
          f"cv_acc={sklearn_result['cv_accuracy']:.4f}")
    print(f"    MLPY:    train={mlpy_result['train_time']:.4f}s, "
          f"predict={mlpy_result['predict_time']:.4f}s, "
          f"cv_acc={mlpy_result['cv_accuracy']:.4f}")
    print()

    # ========================================================================
    # 2. Wine Dataset (178 samples, 13 features, 3 classes)
    # ========================================================================
    print("2. Wine Dataset (178 samples, 13 features, 3 classes)")
    print("-" * 80)
    wine = datasets.load_wine()
    X, y = wine.data, wine.target

    print("  - Random Forest...")
    sklearn_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlpy_rf = LearnerRandomForestClassifier(n_estimators=100, random_state=42)

    sklearn_result = benchmark_classification_sklearn(X, y, sklearn_rf)
    mlpy_result = benchmark_classification_mlpy(X, y, mlpy_rf)

    results.append({
        'dataset': 'Wine',
        'n_samples': 178,
        'n_features': 13,
        'task': 'Classification',
        'algorithm': 'Random Forest',
        'framework': 'sklearn',
        **sklearn_result
    })
    results.append({
        'dataset': 'Wine',
        'n_samples': 178,
        'n_features': 13,
        'task': 'Classification',
        'algorithm': 'Random Forest',
        'framework': 'MLPY',
        **mlpy_result
    })

    print(f"    sklearn: train={sklearn_result['train_time']:.4f}s, "
          f"cv_acc={sklearn_result['cv_accuracy']:.4f}")
    print(f"    MLPY:    train={mlpy_result['train_time']:.4f}s, "
          f"cv_acc={mlpy_result['cv_accuracy']:.4f}")
    print()

    # ========================================================================
    # 3. Breast Cancer Dataset (569 samples, 30 features, 2 classes)
    # ========================================================================
    print("3. Breast Cancer Dataset (569 samples, 30 features, 2 classes)")
    print("-" * 80)
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target

    print("  - Random Forest...")
    sklearn_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlpy_rf = LearnerRandomForestClassifier(n_estimators=100, random_state=42)

    sklearn_result = benchmark_classification_sklearn(X, y, sklearn_rf)
    mlpy_result = benchmark_classification_mlpy(X, y, mlpy_rf)

    results.append({
        'dataset': 'Breast Cancer',
        'n_samples': 569,
        'n_features': 30,
        'task': 'Classification',
        'algorithm': 'Random Forest',
        'framework': 'sklearn',
        **sklearn_result
    })
    results.append({
        'dataset': 'Breast Cancer',
        'n_samples': 569,
        'n_features': 30,
        'task': 'Classification',
        'algorithm': 'Random Forest',
        'framework': 'MLPY',
        **mlpy_result
    })

    print(f"    sklearn: train={sklearn_result['train_time']:.4f}s, "
          f"cv_acc={sklearn_result['cv_accuracy']:.4f}")
    print(f"    MLPY:    train={mlpy_result['train_time']:.4f}s, "
          f"cv_acc={mlpy_result['cv_accuracy']:.4f}")
    print()

    # ========================================================================
    # 4. Diabetes Dataset (442 samples, 10 features, regression)
    # ========================================================================
    print("4. Diabetes Dataset (442 samples, 10 features, regression)")
    print("-" * 80)
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target

    print("  - Random Forest Regressor...")
    from sklearn.ensemble import RandomForestRegressor
    sklearn_rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    mlpy_rfr = LearnerRandomForestRegressor(n_estimators=100, random_state=42)

    sklearn_result = benchmark_regression_sklearn(X, y, sklearn_rfr)
    mlpy_result = benchmark_regression_mlpy(X, y, mlpy_rfr)

    results.append({
        'dataset': 'Diabetes',
        'n_samples': 442,
        'n_features': 10,
        'task': 'Regression',
        'algorithm': 'Random Forest',
        'framework': 'sklearn',
        **sklearn_result
    })
    results.append({
        'dataset': 'Diabetes',
        'n_samples': 442,
        'n_features': 10,
        'task': 'Regression',
        'algorithm': 'Random Forest',
        'framework': 'MLPY',
        **mlpy_result
    })

    print(f"    sklearn: train={sklearn_result['train_time']:.4f}s, "
          f"cv_rmse={sklearn_result['cv_rmse']:.4f}")
    print(f"    MLPY:    train={mlpy_result['train_time']:.4f}s, "
          f"cv_rmse={mlpy_result['cv_rmse']:.4f}")
    print()

    # ========================================================================
    # Save results
    # ========================================================================
    df_results = pd.DataFrame(results)
    df_results.to_csv('benchmarks/results/performance_results.csv', index=False)

    print("=" * 80)
    print("Results saved to: benchmarks/results/performance_results.csv")
    print("=" * 80)
    print()

    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("SUMMARY")
    print("=" * 80)

    # Average speedup
    sklearn_times = df_results[df_results['framework'] == 'sklearn']['train_time']
    mlpy_times = df_results[df_results['framework'] == 'MLPY']['train_time']

    print(f"\nAverage Training Time:")
    print(f"  sklearn: {sklearn_times.mean():.4f}s")
    print(f"  MLPY:    {mlpy_times.mean():.4f}s")
    print(f"  Ratio:   {mlpy_times.mean() / sklearn_times.mean():.2f}x")

    sklearn_mem = df_results[df_results['framework'] == 'sklearn']['train_memory_mb']
    mlpy_mem = df_results[df_results['framework'] == 'MLPY']['train_memory_mb']

    print(f"\nAverage Memory Usage:")
    print(f"  sklearn: {sklearn_mem.mean():.2f} MB")
    print(f"  MLPY:    {mlpy_mem.mean():.2f} MB")
    print(f"  Ratio:   {mlpy_mem.mean() / sklearn_mem.mean():.2f}x")

    # Accuracy comparison (classification tasks only)
    classif_df = df_results[df_results['task'] == 'Classification']
    sklearn_acc = classif_df[classif_df['framework'] == 'sklearn']['cv_accuracy']
    mlpy_acc = classif_df[classif_df['framework'] == 'MLPY']['cv_accuracy']

    print(f"\nAverage CV Accuracy (Classification):")
    print(f"  sklearn: {sklearn_acc.mean():.4f}")
    print(f"  MLPY:    {mlpy_acc.mean():.4f}")
    print(f"  Diff:    {(mlpy_acc.mean() - sklearn_acc.mean()):.4f}")

    print()
    print("=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)

    return df_results


if __name__ == '__main__':
    results = run_benchmarks()
