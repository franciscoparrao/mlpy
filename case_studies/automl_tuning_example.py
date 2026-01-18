"""
AutoML Hyperparameter Tuning Case Study

Demonstrates MLPY's automatic hyperparameter tuning capabilities
and compares with sklearn GridSearchCV.
"""

import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import json
from pathlib import Path

# MLPY imports
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resamplings import ResamplingCV
from mlpy.automl.tuning import TunerGrid, ParamSet, ParamInt, ParamCategorical
from mlpy.resample import resample


def load_iris_task():
    """Load Iris dataset as MLPY task."""
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    task = TaskClassif(data=df, target='species')
    return task, iris.data, iris.target


def mlpy_hyperparameter_tuning(task):
    """
    MLPY approach: Using TunerGrid for automatic hyperparameter tuning.
    """
    print("=" * 80)
    print("MLPY HYPERPARAMETER TUNING")
    print("=" * 80)
    print()

    # Define learner
    learner = LearnerRandomForestClassifier(random_state=42)

    # Define parameter space
    param_set = ParamSet([
        ParamInt('n_estimators', lower=50, upper=200),
        ParamInt('max_depth', lower=3, upper=20),
        ParamCategorical('min_samples_split', values=[2, 5, 10])
    ])

    print("Parameter space:")
    print("  - n_estimators: [50, 200]")
    print("  - max_depth: [3, 20]")
    print("  - min_samples_split: [2, 5, 10]")
    print()

    # Create tuner
    tuner = TunerGrid(resolution=5)

    # Run tuning
    print("Running grid search...")
    start_time = time.time()

    result = tuner.tune(
        learner=learner,
        task=task,
        resampling=ResamplingCV(folds=5),
        measure=MeasureClassifAccuracy(),
        param_set=param_set
    )

    elapsed = time.time() - start_time

    # Print results
    print(f"Tuning completed in {elapsed:.2f}s")
    print(f"Evaluated {len(result.configs)} configurations")
    print()

    print("Best configuration:")
    print(f"  - Accuracy: {result.best_score:.4f}")
    print(f"  - Parameters: {result.best_config}")
    print()

    # Show top 5 configurations
    print("Top 5 configurations:")
    # Sort configs by score (descending for accuracy)
    sorted_indices = sorted(range(len(result.scores)), key=lambda i: result.scores[i], reverse=True)
    for i, idx in enumerate(sorted_indices[:5], 1):
        score = result.scores[idx]
        config = result.configs[idx]
        print(f"  {i}. Score: {score:.4f}, Params: {config}")
    print()

    return result, elapsed


def sklearn_hyperparameter_tuning(X, y):
    """
    sklearn approach: Using GridSearchCV.
    """
    print("=" * 80)
    print("SKLEARN HYPERPARAMETER TUNING")
    print("=" * 80)
    print()

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 87, 125, 162, 200],  # 5 values
        'max_depth': [3, 7, 11, 15, 20],  # 5 values
        'min_samples_split': [2, 5, 10]  # 3 values
    }

    print("Parameter grid:")
    print(f"  - n_estimators: {param_grid['n_estimators']}")
    print(f"  - max_depth: {param_grid['max_depth']}")
    print(f"  - min_samples_split: {param_grid['min_samples_split']}")
    print(f"  Total configurations: {5 * 5 * 3} = 75")
    print()

    # Create model
    rf = RandomForestClassifier(random_state=42)

    # Run grid search
    print("Running grid search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1  # Sequential for fair comparison
    )

    grid_search.fit(X, y)

    elapsed = time.time() - start_time

    # Print results
    print(f"Tuning completed in {elapsed:.2f}s")
    print()

    print("Best configuration:")
    print(f"  - Accuracy: {grid_search.best_score_:.4f}")
    print(f"  - Parameters: {grid_search.best_params_}")
    print()

    # Show top 5
    print("Top 5 configurations:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5_idx = results_df['mean_test_score'].nlargest(5).index

    for i, idx in enumerate(top_5_idx, 1):
        score = results_df.loc[idx, 'mean_test_score']
        params = results_df.loc[idx, 'params']
        print(f"  {i}. Score: {score:.4f}, Params: {params}")
    print()

    return grid_search, elapsed


def compare_code_complexity():
    """
    Compare code complexity: MLPY vs sklearn.
    """
    print("=" * 80)
    print("CODE COMPARISON")
    print("=" * 80)
    print()

    mlpy_code = '''
# MLPY: Simple and declarative
from mlpy.automl.tuning import TunerGrid, ParamSet, ParamInt, ParamCategorical

param_set = ParamSet([
    ParamInt('n_estimators', lower=50, upper=200),
    ParamInt('max_depth', lower=3, upper=20),
    ParamCategorical('min_samples_split', values=[2, 5, 10])
])

tuner = TunerGrid(resolution=5)
result = tuner.tune(
    learner=learner,
    task=task,
    resampling=ResamplingCV(folds=5),
    measure=MeasureClassifAccuracy(),
    param_set=param_set
)

best_learner = result.best_learner
best_params = result.best_config
    '''

    sklearn_code = '''
# sklearn: Manual grid definition
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 87, 125, 162, 200],  # Manual calculation
    'max_depth': [3, 7, 11, 15, 20],          # Manual calculation
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
    '''

    mlpy_lines = len([l for l in mlpy_code.strip().split('\n') if l.strip() and not l.strip().startswith('#')])
    sklearn_lines = len([l for l in sklearn_code.strip().split('\n') if l.strip() and not l.strip().startswith('#')])

    print("Lines of code:")
    print(f"  MLPY: {mlpy_lines} lines")
    print(f"  sklearn: {sklearn_lines} lines")
    print(f"  Reduction: {((sklearn_lines - mlpy_lines) / sklearn_lines * 100):.1f}%")
    print()

    print("Key differences:")
    print("  MLPY:")
    print("    + Declarative parameter space (ParamInt, ParamChoice)")
    print("    + Automatic grid generation")
    print("    + Unified with task/learner/measure abstraction")
    print("    + Returns rich TuneResult object")
    print()
    print("  sklearn:")
    print("    + Manual grid calculation")
    print("    + Separate from task abstraction")
    print("    + Returns GridSearchCV object")
    print()


def save_results(mlpy_result, mlpy_time, sklearn_result, sklearn_time):
    """Save results for paper."""
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
            'best_score': mlpy_result.best_score,
            'best_config': mlpy_result.best_config,
            'n_configs': len(mlpy_result.configs)
        },
        'sklearn': {
            'time': sklearn_time,
            'best_score': sklearn_result.best_score_,
            'best_params': sklearn_result.best_params_,
            'n_configs': len(sklearn_result.cv_results_['params'])
        }
    }

    # Save JSON
    with open(output_dir / 'automl_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir / 'automl_tuning_results.json'}")


def main():
    """Run complete AutoML tuning case study."""
    print("\n")
    print("=" * 80)
    print("AUTOML HYPERPARAMETER TUNING CASE STUDY")
    print("=" * 80)
    print("\n")

    # Load data
    print("Loading Iris dataset...")
    task, X, y = load_iris_task()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, 3 classes")
    print()

    # MLPY tuning
    mlpy_result, mlpy_time = mlpy_hyperparameter_tuning(task)

    # sklearn tuning
    sklearn_result, sklearn_time = sklearn_hyperparameter_tuning(X, y)

    # Code comparison
    compare_code_complexity()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"MLPY:")
    print(f"  - Time: {mlpy_time:.2f}s")
    print(f"  - Best accuracy: {mlpy_result.best_score:.4f}")
    print(f"  - Configurations evaluated: {len(mlpy_result.configs)}")
    print()
    print(f"sklearn:")
    print(f"  - Time: {sklearn_time:.2f}s")
    print(f"  - Best accuracy: {sklearn_result.best_score_:.4f}")
    print(f"  - Configurations evaluated: {len(sklearn_result.cv_results_['params'])}")
    print()

    speedup = sklearn_time / mlpy_time
    print(f"MLPY is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than sklearn")
    print()

    print("Key findings:")
    print("  ✓ MLPY provides declarative parameter space definition")
    print("  ✓ Automatic grid generation from continuous ranges")
    print("  ✓ Unified API with task/learner/measure")
    print("  ✓ Similar performance and accuracy to sklearn")
    print()

    # Save results
    save_results(mlpy_result, mlpy_time, sklearn_result, sklearn_time)

    print("=" * 80)
    print("Case study complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
