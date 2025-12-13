"""
Quick AutoML Tuning Example - Simplified for faster execution
"""

import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import json
from pathlib import Path

# MLPY imports
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resamplings import ResamplingCV
from mlpy.automl.tuning import TunerGrid, ParamSet, ParamInt, ParamCategorical


def load_iris_task():
    """Load Iris dataset as MLPY task."""
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    task = TaskClassif(data=df, target='species')
    return task, iris.data, iris.target


def main():
    print("\n" * 2)
    print("=" * 80)
    print("AUTOML HYPERPARAMETER TUNING - QUICK EXAMPLE")
    print("=" * 80)
    print()

    # Load data
    print("Loading Iris dataset...")
    task, X, y = load_iris_task()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, 3 classes")
    print()

    # MLPY tuning
    print("=" * 80)
    print("MLPY HYPERPARAMETER TUNING")
    print("=" * 80)
    print()

    learner = LearnerRandomForestClassifier(random_state=42)

    # Smaller parameter space for quick results
    param_set = ParamSet([
        ParamInt('n_estimators', lower=50, upper=150),
        ParamInt('max_depth', lower=5, upper=15),
        ParamCategorical('min_samples_split', values=[2, 5])
    ])

    print("Parameter space:")
    print("  - n_estimators: [50, 150]")
    print("  - max_depth: [5, 15]")
    print("  - min_samples_split: [2, 5]")
    print()

    tuner = TunerGrid(resolution=3)  # 3x3x2 = 18 configs

    print(f"Running grid search with {3*3*2} configurations...")
    start_time = time.time()

    result = tuner.tune(
        learner=learner,
        task=task,
        resampling=ResamplingCV(folds=5),
        measure=MeasureClassifAccuracy(),
        param_set=param_set
    )

    mlpy_time = time.time() - start_time

    print(f"\nMLPY tuning completed in {mlpy_time:.2f}s")
    print(f"Evaluated {len(result.configs)} configurations")
    print()

    print("Best configuration:")
    print(f"  - Accuracy: {result.best_score:.4f}")
    print(f"  - Parameters: {result.best_config}")
    print()

    # Show top 5
    print("Top 5 configurations:")
    sorted_indices = sorted(range(len(result.scores)), key=lambda i: result.scores[i], reverse=True)
    for i, idx in enumerate(sorted_indices[:5], 1):
        score = result.scores[idx]
        config = result.configs[idx]
        print(f"  {i}. Score: {score:.4f}, Params: {config}")
    print()

    # sklearn comparison (smaller grid)
    print("=" * 80)
    print("SKLEARN COMPARISON")
    print("=" * 80)
    print()

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }

    print(f"Parameter grid: {3*3*2} configurations")
    print()

    rf = RandomForestClassifier(random_state=42)

    print("Running sklearn GridSearchCV...")
    start_time = time.time()

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )

    grid_search.fit(X, y)
    sklearn_time = time.time() - start_time

    print(f"\nsklearn tuning completed in {sklearn_time:.2f}s")
    print()

    print("Best configuration:")
    print(f"  - Accuracy: {grid_search.best_score_:.4f}")
    print(f"  - Parameters: {grid_search.best_params_}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"MLPY:")
    print(f"  - Time: {mlpy_time:.2f}s")
    print(f"  - Best accuracy: {result.best_score:.4f}")
    print(f"  - Configurations: {len(result.configs)}")
    print()
    print(f"sklearn:")
    print(f"  - Time: {sklearn_time:.2f}s")
    print(f"  - Best accuracy: {grid_search.best_score_:.4f}")
    print(f"  - Configurations: {len(grid_search.cv_results_['params'])}")
    print()

    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    results = {
        'dataset': 'Iris',
        'n_samples': 150,
        'n_features': 4,
        'n_classes': 3,
        'cv_folds': 5,
        'mlpy': {
            'time': mlpy_time,
            'best_score': float(result.best_score),
            'best_config': result.best_config,
            'n_configs': len(result.configs),
            'all_configs': [{'config': c, 'score': float(s)} for c, s in zip(result.configs, result.scores)]
        },
        'sklearn': {
            'time': sklearn_time,
            'best_score': float(grid_search.best_score_),
            'best_params': grid_search.best_params_,
            'n_configs': len(grid_search.cv_results_['params'])
        }
    }

    results_file = output_dir / 'automl_tuning_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
