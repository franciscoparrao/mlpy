"""
MLPY AutoML Demo
================

Demonstrates the AutoML capabilities of MLPY.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    load_diabetes, make_classification,
    make_regression
)
from sklearn.model_selection import train_test_split
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from mlpy.automl.auto_learner import AutoMLearner, AutoMLConfig
from mlpy.automl.search_spaces import get_default_search_space
from mlpy.automl.optimizers import get_optimizer
from mlpy.automl.pipeline import AutoPipeline, PipelineOptimizer
from mlpy.automl.neural_architecture import NASSearcher, ArchitectureSpace, NetworkBuilder


def demo_basic_automl():
    """Demo basic AutoML functionality."""
    print("\n" + "="*60)
    print("1. BASIC AUTOML DEMO")
    print("="*60)
    
    # Load dataset
    print("\nLoading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Configure AutoML
    config = AutoMLConfig(
        task_type="classification",
        time_budget=60,  # 1 minute
        max_trials=20,
        optimization_metric="accuracy",
        optimizer="random",
        auto_feature_engineering=True,
        verbose=1
    )
    
    print("\nAutoML Configuration:")
    print(f"  Task: {config.task_type}")
    print(f"  Time budget: {config.time_budget}s")
    print(f"  Max trials: {config.max_trials}")
    print(f"  Optimizer: {config.optimizer}")
    
    # Run AutoML
    print("\nStarting AutoML search...")
    automl = AutoMLearner(config)
    
    start_time = time.time()
    automl.fit(X_train, y_train, X_test, y_test)
    elapsed = time.time() - start_time
    
    # Results
    print(f"\nAutoML completed in {elapsed:.2f} seconds")
    print(f"Best score: {automl.results.best_score:.4f}")
    print(f"Trials completed: {automl.results.n_trials_completed}")
    
    # Show top models
    print("\nTop 5 models:")
    summary = automl.results.summary()
    print(summary.head())
    
    # Make predictions
    predictions = automl.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    return automl


def demo_advanced_automl():
    """Demo advanced AutoML with custom configuration."""
    print("\n" + "="*60)
    print("2. ADVANCED AUTOML DEMO")
    print("="*60)
    
    # Create synthetic dataset
    print("\nCreating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Advanced configuration
    config = AutoMLConfig(
        task_type="classification",
        time_budget=120,  # 2 minutes
        max_trials=50,
        optimizer="bayesian",  # Bayesian optimization
        auto_feature_engineering=True,
        max_features_generated=30,
        feature_selection=True,
        ensemble_size=3,
        early_stopping=True,
        patience=10,
        verbose=1
    )
    
    # Custom model selection
    config.include_models = ["RandomForest", "GradientBoosting", "XGBoost", "LogisticRegression"]
    
    print("\nAdvanced AutoML Configuration:")
    print(f"  Optimizer: Bayesian")
    print(f"  Feature engineering: Enabled")
    print(f"  Ensemble size: {config.ensemble_size}")
    print(f"  Early stopping: Enabled")
    print(f"  Models: {config.include_models}")
    
    # Run AutoML
    print("\nStarting advanced AutoML search...")
    automl = AutoMLearner(config)
    automl.fit(X_train, y_train, X_test, y_test)
    
    # Detailed results
    print("\nDetailed Results:")
    print(f"Best model: {automl.results.best_model.__class__.__name__}")
    print(f"Best score: {automl.results.best_score:.4f}")
    
    if automl.results.generated_features:
        print(f"Generated features: {len(automl.results.generated_features)}")
    
    if automl.results.selected_features:
        print(f"Selected features: {len(automl.results.selected_features)}")
    
    # Feature importance
    if automl.results.feature_importance:
        print("\nTop 5 important features:")
        importance = sorted(
            automl.results.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for feat, imp in importance:
            print(f"  Feature {feat}: {imp:.4f}")
    
    return automl


def demo_pipeline_optimization():
    """Demo pipeline optimization."""
    print("\n" + "="*60)
    print("3. PIPELINE OPTIMIZATION DEMO")
    print("="*60)
    
    # Load dataset
    print("\nLoading Wine dataset...")
    wine = load_wine()
    X, y = wine.data, wine.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline optimizer
    print("\nOptimizing ML pipeline...")
    pipeline_opt = PipelineOptimizer(
        task_type="classification",
        time_budget=60,
        n_jobs=-1
    )
    
    # Optimize
    best_pipeline = pipeline_opt.optimize(X_train, y_train, n_trials=20)
    
    print(f"\nBest pipeline score: {pipeline_opt.best_score:.4f}")
    print("\nBest pipeline steps:")
    for step_name, step in best_pipeline.steps:
        print(f"  - {step_name}: {step.__class__.__name__}")
    
    # Evaluate on test set
    from sklearn.metrics import accuracy_score
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    return best_pipeline


def demo_hyperparameter_optimization():
    """Demo different hyperparameter optimization strategies."""
    print("\n" + "="*60)
    print("4. HYPERPARAMETER OPTIMIZATION STRATEGIES")
    print("="*60)
    
    # Load dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get search space
    search_space = get_default_search_space(
        task_type="classification",
        n_features=X.shape[1],
        n_samples=X.shape[0],
        include_models=["RandomForest", "GradientBoosting"]
    )
    
    print(f"\nSearch space: {list(search_space.keys())}")
    
    # Test different optimizers
    optimizers_to_test = ["random", "grid", "bayesian", "evolutionary"]
    results = {}
    
    for opt_name in optimizers_to_test:
        print(f"\nTesting {opt_name} optimizer...")
        
        # Create optimizer
        optimizer = get_optimizer(
            opt_name,
            search_space,
            n_trials=10,
            random_state=42
        )
        
        # Simple evaluation loop
        for i in range(10):
            config = optimizer.get_next_config()
            if config is None:
                break
            
            # Quick evaluation with cross-validation
            from sklearn.model_selection import cross_val_score
            model = config['model']
            
            try:
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                score = scores.mean()
                optimizer.update(config, score)
            except:
                score = 0.0
        
        best_config, best_score = optimizer.get_best()
        results[opt_name] = best_score
        print(f"  Best score: {best_score:.4f}")
    
    # Compare results
    print("\nOptimizer Comparison:")
    for opt_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {opt_name:15s}: {score:.4f}")
    
    return results


def demo_nas_simple():
    """Demo simple Neural Architecture Search."""
    print("\n" + "="*60)
    print("5. NEURAL ARCHITECTURE SEARCH (NAS) DEMO")
    print("="*60)
    
    # Check if deep learning libraries are available
    try:
        import tensorflow as tf
        backend = "tensorflow"
        print("\nUsing TensorFlow backend")
    except ImportError:
        try:
            import torch
            backend = "pytorch"
            print("\nUsing PyTorch backend")
        except ImportError:
            print("\nNo deep learning backend available (TensorFlow or PyTorch)")
            print("Install with: pip install tensorflow or pip install torch")
            return None
    
    # Create small dataset for quick demo
    print("\nCreating dataset for NAS...")
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define search space
    arch_space = ArchitectureSpace(
        input_shape=(10,),
        output_shape=2,
        task_type="classification",
        max_layers=5,
        max_units=128
    )
    
    print("\nArchitecture Search Space:")
    print(f"  Input shape: {arch_space.input_shape}")
    print(f"  Output shape: {arch_space.output_shape}")
    print(f"  Max layers: {arch_space.max_layers}")
    print(f"  Layer types: {arch_space.layer_types}")
    
    # Create NAS searcher
    nas = NASSearcher(
        search_space=arch_space,
        backend=backend,
        search_strategy="random",
        n_trials=5,  # Small number for demo
        time_budget=60
    )
    
    print("\nSearching for best architecture...")
    best_arch = nas.search(X_train, y_train, X_test, y_test)
    
    if best_arch:
        print(f"\nBest architecture found:")
        print(f"  Score: {nas.best_score:.4f}")
        print(f"  Layers: {len(best_arch['layers'])}")
        print(f"  Optimizer: {best_arch['optimizer']}")
        print(f"  Learning rate: {best_arch['learning_rate']}")
        
        print("\nArchitecture details:")
        for i, layer in enumerate(best_arch['layers']):
            print(f"  Layer {i+1}: {layer}")
    
    return nas


def demo_regression_automl():
    """Demo AutoML for regression tasks."""
    print("\n" + "="*60)
    print("6. REGRESSION AUTOML DEMO")
    print("="*60)
    
    # Create regression dataset
    print("\nCreating regression dataset...")
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configure AutoML for regression
    config = AutoMLConfig(
        task_type="regression",
        time_budget=60,
        max_trials=20,
        optimization_metric="neg_mean_squared_error",
        optimizer="bayesian",
        auto_feature_engineering=True,
        verbose=1
    )
    
    print("\nRegression AutoML Configuration:")
    print(f"  Task: {config.task_type}")
    print(f"  Metric: {config.optimization_metric}")
    
    # Run AutoML
    print("\nStarting regression AutoML...")
    automl = AutoMLearner(config)
    automl.fit(X_train, y_train, X_test, y_test)
    
    # Results
    print(f"\nBest score (negative MSE): {automl.results.best_score:.4f}")
    
    # Make predictions
    predictions = automl.predict(X_test)
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    return automl


def main():
    """Run all AutoML demos."""
    print("\n" + "="*60)
    print("MLPY AUTOML DEMONSTRATION")
    print("="*60)
    print("\nThis demo showcases the AutoML capabilities of MLPY")
    
    demos = [
        ("Basic AutoML", demo_basic_automl),
        ("Advanced AutoML", demo_advanced_automl),
        ("Pipeline Optimization", demo_pipeline_optimization),
        ("Hyperparameter Optimization", demo_hyperparameter_optimization),
        ("Neural Architecture Search", demo_nas_simple),
        ("Regression AutoML", demo_regression_automl)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\nRunning: {demo_name}")
            result = demo_func()
            results[demo_name] = "Success"
        except Exception as e:
            print(f"\nError in {demo_name}: {e}")
            results[demo_name] = f"Failed: {str(e)[:50]}"
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    for demo_name, status in results.items():
        print(f"{demo_name:30s}: {status}")
    
    print("\n" + "="*60)
    print("Key Features Demonstrated:")
    print("  - Automatic hyperparameter optimization")
    print("  - Multiple optimization strategies (Random, Grid, Bayesian, Evolutionary)")
    print("  - Automatic feature engineering")
    print("  - Pipeline optimization")
    print("  - Neural Architecture Search (NAS)")
    print("  - Support for classification and regression")
    print("  - Ensemble methods")
    print("  - Early stopping")
    print("="*60)


if __name__ == "__main__":
    main()