#!/usr/bin/env python3
"""
Ensemble Learning Demonstration for MLPY
========================================

This script demonstrates the ensemble learning capabilities implemented in MLPY.
It shows voting, stacking, and blending ensemble methods working together.

Features demonstrated:
- LearnerVoting (hard and soft voting)
- LearnerStacking (with cross-validation)  
- LearnerBlending (with holdout validation)
- Performance comparison between methods

Run: python examples/ensemble_demo.py
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import (
    LearnerClassifSklearn, 
    LearnerRegrSklearn,
    LearnerVoting,
    LearnerStacking, 
    LearnerBlending,
    create_ensemble
)
from mlpy.resamplings import ResamplingHoldout
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE


def create_classification_demo():
    """Create and run classification ensemble demo."""
    print("=" * 60)
    print("ENSEMBLE LEARNING CLASSIFICATION DEMO")
    print("=" * 60)
    
    # Create classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=3,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    task = TaskClassif(df, target='target')
    
    print(f"Dataset: {task.nrow} samples, {task.ncol-1} features, {len(task.class_names)} classes")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Create train/test split
    holdout = ResamplingHoldout(ratio=0.3, stratify=True)
    split = holdout.instantiate(task)
    train_task = task.filter(split.train_set(0))
    test_task = task.filter(split.test_set(0))
    
    print(f"Train: {train_task.nrow} samples, Test: {test_task.nrow} samples")
    
    # Base learners
    base_learners = [
        LearnerClassifSklearn(
            estimator=LogisticRegression(random_state=42, max_iter=200),
            id="logistic"
        ),
        LearnerClassifSklearn(
            estimator=DecisionTreeClassifier(random_state=42, max_depth=10),
            id="tree"
        ),
        LearnerClassifSklearn(
            estimator=KNeighborsClassifier(n_neighbors=5),
            id="knn"
        )
    ]
    
    # Meta-learner for stacking/blending
    meta_learner = LearnerClassifSklearn(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        id="meta_rf"
    )
    
    # Test individual base learners
    print("\n" + "-" * 40)
    print("INDIVIDUAL LEARNER PERFORMANCE")
    print("-" * 40)
    
    individual_results = {}
    for learner in base_learners:
        trained = learner.train(train_task)
        pred = trained.predict(test_task)
        accuracy = accuracy_score(test_task.truth(), pred.response)
        individual_results[learner.id] = accuracy
        print(f"{learner.id:10}: {accuracy:.4f}")
    
    # Test ensemble methods
    print("\n" + "-" * 40)
    print("ENSEMBLE METHODS PERFORMANCE")
    print("-" * 40)
    
    ensemble_methods = [
        ("Hard Voting", LearnerVoting(base_learners, voting='hard', id="voting_hard")),
        ("Soft Voting", LearnerVoting(base_learners, voting='soft', id="voting_soft")),
        ("Stacking", LearnerStacking(base_learners, meta_learner, cv_folds=3, id="stacking")),
        ("Stacking+Proba", LearnerStacking(base_learners, meta_learner, use_proba=True, cv_folds=3, id="stacking_proba")),
        ("Blending", LearnerBlending(base_learners, meta_learner, blend_ratio=0.2, id="blending")),
        ("Blending+Proba", LearnerBlending(base_learners, meta_learner, use_proba=True, blend_ratio=0.2, id="blending_proba"))
    ]
    
    ensemble_results = {}
    for name, ensemble in ensemble_methods:
        try:
            print(f"Training {name}...")
            trained = ensemble.train(train_task)
            pred = trained.predict(test_task)
            accuracy = accuracy_score(test_task.truth(), pred.response)
            ensemble_results[name] = accuracy
            print(f"{name:15}: {accuracy:.4f}")
        except Exception as e:
            print(f"{name:15}: ERROR - {str(e)[:50]}...")
            ensemble_results[name] = None
    
    # Summary
    print("\n" + "=" * 40)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 40)
    
    print("Individual Learners:")
    for name, acc in individual_results.items():
        print(f"  {name:15}: {acc:.4f}")
    
    print("\nEnsemble Methods:")
    for name, acc in ensemble_results.items():
        if acc is not None:
            print(f"  {name:15}: {acc:.4f}")
        else:
            print(f"  {name:15}: FAILED")
    
    best_individual = max(individual_results.values())
    best_ensemble = max([acc for acc in ensemble_results.values() if acc is not None])
    improvement = best_ensemble - best_individual
    
    print(f"\nBest individual: {best_individual:.4f}")
    print(f"Best ensemble:   {best_ensemble:.4f}")
    print(f"Improvement:     {improvement:.4f} ({improvement/best_individual*100:.1f}%)")


def create_regression_demo():
    """Create and run regression ensemble demo."""
    print("\n\n" + "=" * 60)
    print("ENSEMBLE LEARNING REGRESSION DEMO")
    print("=" * 60)
    
    # Create regression dataset
    X, y = make_regression(
        n_samples=800,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
    df['target'] = y
    task = TaskRegr(df, target='target')
    
    print(f"Dataset: {task.nrow} samples, {task.ncol-1} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}], std: {y.std():.2f}")
    
    # Create train/test split
    holdout = ResamplingHoldout(ratio=0.3)
    split = holdout.instantiate(task)
    train_task = task.filter(split.train_set(0))
    test_task = task.filter(split.test_set(0))
    
    print(f"Train: {train_task.nrow} samples, Test: {test_task.nrow} samples")
    
    # Base learners
    base_learners = [
        LearnerRegrSklearn(
            estimator=LinearRegression(),
            id="linear"
        ),
        LearnerRegrSklearn(
            estimator=DecisionTreeRegressor(random_state=42, max_depth=10),
            id="tree"
        ),
        LearnerRegrSklearn(
            estimator=KNeighborsRegressor(n_neighbors=5),
            id="knn"
        )
    ]
    
    # Meta-learner
    meta_learner = LearnerRegrSklearn(
        estimator=LinearRegression(),
        id="meta_linear"
    )
    
    # Test individual base learners
    print("\n" + "-" * 40)
    print("INDIVIDUAL LEARNER PERFORMANCE")
    print("-" * 40)
    
    individual_results = {}
    for learner in base_learners:
        trained = learner.train(train_task)
        pred = trained.predict(test_task)
        mse = mean_squared_error(test_task.truth(), pred.response)
        rmse = np.sqrt(mse)
        individual_results[learner.id] = rmse
        print(f"{learner.id:10}: RMSE = {rmse:.4f}")
    
    # Test ensemble methods (voting = averaging for regression)
    print("\n" + "-" * 40)
    print("ENSEMBLE METHODS PERFORMANCE") 
    print("-" * 40)
    
    ensemble_methods = [
        ("Averaging", LearnerVoting(base_learners, id="averaging")),
        ("Stacking", LearnerStacking(base_learners, meta_learner, cv_folds=3, id="stacking")),
        ("Blending", LearnerBlending(base_learners, meta_learner, blend_ratio=0.2, id="blending"))
    ]
    
    ensemble_results = {}
    for name, ensemble in ensemble_methods:
        try:
            print(f"Training {name}...")
            trained = ensemble.train(train_task)
            pred = trained.predict(test_task)
            mse = mean_squared_error(test_task.truth(), pred.response)
            rmse = np.sqrt(mse)
            ensemble_results[name] = rmse
            print(f"{name:15}: RMSE = {rmse:.4f}")
        except Exception as e:
            print(f"{name:15}: ERROR - {str(e)[:50]}...")
            ensemble_results[name] = None
    
    # Summary
    print("\n" + "=" * 40)
    print("REGRESSION RESULTS SUMMARY")
    print("=" * 40)
    
    print("Individual Learners:")
    for name, rmse in individual_results.items():
        print(f"  {name:15}: {rmse:.4f}")
    
    print("\nEnsemble Methods:")
    for name, rmse in ensemble_results.items():
        if rmse is not None:
            print(f"  {name:15}: {rmse:.4f}")
        else:
            print(f"  {name:15}: FAILED")
    
    best_individual = min(individual_results.values())
    best_ensemble = min([rmse for rmse in ensemble_results.values() if rmse is not None])
    improvement = best_individual - best_ensemble
    
    print(f"\nBest individual: {best_individual:.4f}")
    print(f"Best ensemble:   {best_ensemble:.4f}")
    print(f"Improvement:     {improvement:.4f} ({improvement/best_individual*100:.1f}%)")


def demonstrate_convenience_functions():
    """Demonstrate the convenience functions."""
    print("\n\n" + "=" * 60)
    print("CONVENIENCE FUNCTIONS DEMO")
    print("=" * 60)
    
    # Quick dataset
    X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(10)])
    df['target'] = y
    task = TaskClassif(df, target='target')
    
    base_learners = [
        LearnerClassifSklearn(LogisticRegression(random_state=42, max_iter=200)),
        LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    ]
    
    meta_learner = LearnerClassifSklearn(LogisticRegression(random_state=42, max_iter=200))
    
    print("Creating ensembles using create_ensemble() function...")
    
    # Using convenience function
    voting_ensemble = create_ensemble('voting', base_learners, voting='soft')
    stacking_ensemble = create_ensemble('stacking', base_learners, meta_learner=meta_learner)
    blending_ensemble = create_ensemble('blending', base_learners, meta_learner=meta_learner)
    
    for name, ensemble in [('Voting', voting_ensemble), ('Stacking', stacking_ensemble), ('Blending', blending_ensemble)]:
        ensemble.train(task)
        pred = ensemble.predict(task)
        accuracy = accuracy_score(task.truth(), pred.response)
        print(f"{name} ensemble accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    print("MLPY Ensemble Learning Demonstration")
    print("====================================")
    print()
    print("This demo shows the ensemble learning capabilities:")
    print("- Voting (hard/soft)")
    print("- Stacking with meta-learner")
    print("- Blending with holdout")
    print("- Performance comparisons")
    print()
    
    try:
        create_classification_demo()
        create_regression_demo()
        demonstrate_convenience_functions()
        
        print("\n\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key takeaways:")
        print("- Ensemble methods often outperform individual learners")
        print("- Stacking and blending can capture complex patterns")
        print("- Different methods work better for different problems")
        print("- MLPY provides easy-to-use ensemble tools")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()