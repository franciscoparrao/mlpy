"""
Usability Benchmark: MLPY vs scikit-learn

Compares code complexity (lines of code) for common ML tasks.

Run with: python benchmarks/usability_benchmark.py
"""

import pandas as pd


def count_loc(code):
    """Count lines of code (excluding empty lines and comments)"""
    lines = code.strip().split('\n')
    return len([l for l in lines if l.strip() and not l.strip().startswith('#')])


def analyze_usability():
    """Analyze code complexity for common tasks"""
    results = []

    print("=" * 80)
    print("MLPY vs scikit-learn Usability Benchmark")
    print("=" * 80)
    print()

    # ========================================================================
    # Task 1: Simple training
    # ========================================================================
    print("Task 1: Simple Training (load data + train model + predict)")
    print("-" * 80)

    sklearn_code = """
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)
"""

    mlpy_code = """
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.measures import MeasureClassifAccuracy
from sklearn import datasets

iris = datasets.load_iris()
task = TaskClassif(data=iris.data, target=iris.target)
learner = LearnerRandomForestClassifier(n_estimators=100, random_state=42)
learner.train(task)
predictions = learner.predict(task)
accuracy = MeasureClassifAccuracy().score(predictions.truth, predictions.response)
"""

    sklearn_loc = count_loc(sklearn_code)
    mlpy_loc = count_loc(mlpy_code)

    results.append({
        'task': 'Simple Training',
        'sklearn_loc': sklearn_loc,
        'mlpy_loc': mlpy_loc,
        'difference': mlpy_loc - sklearn_loc,
        'percent_change': ((mlpy_loc - sklearn_loc) / sklearn_loc * 100)
    })

    print(f"  sklearn: {sklearn_loc} LOC")
    print(f"  MLPY:    {mlpy_loc} LOC")
    print(f"  Change:  {mlpy_loc - sklearn_loc:+d} LOC ({((mlpy_loc - sklearn_loc) / sklearn_loc * 100):+.1f}%)")
    print()

    # ========================================================================
    # Task 2: Cross-validation
    # ========================================================================
    print("Task 2: Cross-Validation (5-fold CV with multiple metrics)")
    print("-" * 80)

    sklearn_code = """
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=100, random_state=42)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_weighted': make_scorer(f1_score, average='weighted')
}
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
mean_accuracy = cv_results['test_accuracy'].mean()
mean_f1 = cv_results['test_f1_weighted'].mean()
"""

    mlpy_code = """
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
from mlpy.resamplings import ResamplingCV
from mlpy import resample
from sklearn import datasets

iris = datasets.load_iris()
task = TaskClassif(data=iris.data, target=iris.target)
learner = LearnerRandomForestClassifier(n_estimators=100, random_state=42)
result = resample(task, learner, ResamplingCV(folds=5),
                  measures=[MeasureClassifAccuracy(), MeasureClassifF1()])
mean_accuracy = result.score('classif.acc', average='mean')
mean_f1 = result.score('classif.f1', average='mean')
"""

    sklearn_loc = count_loc(sklearn_code)
    mlpy_loc = count_loc(mlpy_code)

    results.append({
        'task': 'Cross-Validation',
        'sklearn_loc': sklearn_loc,
        'mlpy_loc': mlpy_loc,
        'difference': mlpy_loc - sklearn_loc,
        'percent_change': ((mlpy_loc - sklearn_loc) / sklearn_loc * 100)
    })

    print(f"  sklearn: {sklearn_loc} LOC")
    print(f"  MLPY:    {mlpy_loc} LOC")
    print(f"  Change:  {mlpy_loc - sklearn_loc:+d} LOC ({((mlpy_loc - sklearn_loc) / sklearn_loc * 100):+.1f}%)")
    print()

    # ========================================================================
    # Task 3: Pipeline with preprocessing
    # ========================================================================
    print("Task 3: Pipeline (preprocessing + model)")
    print("-" * 80)

    sklearn_code = """
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X, y)
scores = cross_val_score(pipeline, X, y, cv=5)
mean_score = scores.mean()
"""

    mlpy_code = """
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.pipelines import Graph, PipeOpScale, PipeOpPCA, linear_pipeline
from mlpy.pipelines.graph_learner import GraphLearner
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resamplings import ResamplingCV
from mlpy import resample
from sklearn import datasets

iris = datasets.load_iris()
task = TaskClassif(data=iris.data, target=iris.target)
graph = linear_pipeline(PipeOpScale(), PipeOpPCA(n_components=2))
learner = GraphLearner(graph, LearnerRandomForestClassifier(n_estimators=100, random_state=42))
result = resample(task, learner, ResamplingCV(folds=5), measures=[MeasureClassifAccuracy()])
mean_score = result.score('classif.acc', average='mean')
"""

    sklearn_loc = count_loc(sklearn_code)
    mlpy_loc = count_loc(mlpy_code)

    results.append({
        'task': 'Pipeline',
        'sklearn_loc': sklearn_loc,
        'mlpy_loc': mlpy_loc,
        'difference': mlpy_loc - sklearn_loc,
        'percent_change': ((mlpy_loc - sklearn_loc) / sklearn_loc * 100)
    })

    print(f"  sklearn: {sklearn_loc} LOC")
    print(f"  MLPY:    {mlpy_loc} LOC")
    print(f"  Change:  {mlpy_loc - sklearn_loc:+d} LOC ({((mlpy_loc - sklearn_loc) / sklearn_loc * 100):+.1f}%)")
    print()

    # ========================================================================
    # Task 4: Benchmarking multiple models
    # ========================================================================
    print("Task 4: Benchmarking (compare 3 models with CV)")
    print("-" * 80)

    sklearn_code = """
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import pandas as pd

iris = datasets.load_iris()
X, y = iris.data, iris.target
models = {
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LR': LogisticRegression(max_iter=1000, random_state=42)
}
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='weighted')
}
results = []
for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
    results.append({
        'model': name,
        'accuracy': cv_results['test_accuracy'].mean(),
        'f1': cv_results['test_f1'].mean()
    })
df_results = pd.DataFrame(results)
"""

    mlpy_code = """
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import (LearnerRandomForestClassifier,
                                    LearnerGradientBoostingClassifier,
                                    LearnerLogisticRegression)
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
from mlpy.resamplings import ResamplingCV
from mlpy import benchmark
from sklearn import datasets

iris = datasets.load_iris()
task = TaskClassif(data=iris.data, target=iris.target)
learners = [
    LearnerRandomForestClassifier(n_estimators=100, random_state=42),
    LearnerGradientBoostingClassifier(n_estimators=100, random_state=42),
    LearnerLogisticRegression(max_iter=1000, random_state=42)
]
result = benchmark(tasks=[task], learners=learners, resamplings=ResamplingCV(folds=5),
                   measures=[MeasureClassifAccuracy(), MeasureClassifF1()])
"""

    sklearn_loc = count_loc(sklearn_code)
    mlpy_loc = count_loc(mlpy_code)

    results.append({
        'task': 'Benchmarking',
        'sklearn_loc': sklearn_loc,
        'mlpy_loc': mlpy_loc,
        'difference': mlpy_loc - sklearn_loc,
        'percent_change': ((mlpy_loc - sklearn_loc) / sklearn_loc * 100)
    })

    print(f"  sklearn: {sklearn_loc} LOC")
    print(f"  MLPY:    {mlpy_loc} LOC")
    print(f"  Change:  {mlpy_loc - sklearn_loc:+d} LOC ({((mlpy_loc - sklearn_loc) / sklearn_loc * 100):+.1f}%)")
    print()

    # ========================================================================
    # Task 5: Hyperparameter tuning
    # ========================================================================
    print("Task 5: Hyperparameter Tuning (Grid Search with CV)")
    print("-" * 80)

    sklearn_code = """
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
X, y = iris.data, iris.target
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
"""

    mlpy_code = """
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.automl.tuning import TunerGridSearch
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resamplings import ResamplingCV
from sklearn import datasets

iris = datasets.load_iris()
task = TaskClassif(data=iris.data, target=iris.target)
learner = LearnerRandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
tuner = TunerGridSearch(param_grid)
result = tuner.tune(learner, task, ResamplingCV(folds=5), MeasureClassifAccuracy())
best_params = result['best_params']
best_score = result['best_score']
"""

    sklearn_loc = count_loc(sklearn_code)
    mlpy_loc = count_loc(mlpy_code)

    results.append({
        'task': 'Hyperparameter Tuning',
        'sklearn_loc': sklearn_loc,
        'mlpy_loc': mlpy_loc,
        'difference': mlpy_loc - sklearn_loc,
        'percent_change': ((mlpy_loc - sklearn_loc) / sklearn_loc * 100)
    })

    print(f"  sklearn: {sklearn_loc} LOC")
    print(f"  MLPY:    {mlpy_loc} LOC")
    print(f"  Change:  {mlpy_loc - sklearn_loc:+d} LOC ({((mlpy_loc - sklearn_loc) / sklearn_loc * 100):+.1f}%)")
    print()

    # ========================================================================
    # Save results
    # ========================================================================
    df_results = pd.DataFrame(results)
    df_results.to_csv('benchmarks/results/usability_results.csv', index=False)

    print("=" * 80)
    print("Results saved to: benchmarks/results/usability_results.csv")
    print("=" * 80)
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("SUMMARY")
    print("=" * 80)

    total_sklearn = df_results['sklearn_loc'].sum()
    total_mlpy = df_results['mlpy_loc'].sum()
    avg_percent = df_results['percent_change'].mean()

    print(f"\nTotal Lines of Code:")
    print(f"  sklearn: {total_sklearn} LOC")
    print(f"  MLPY:    {total_mlpy} LOC")
    print(f"  Change:  {total_mlpy - total_sklearn:+d} LOC ({((total_mlpy - total_sklearn) / total_sklearn * 100):+.1f}%)")

    print(f"\nAverage Change:")
    print(f"  {avg_percent:+.1f}% across all tasks")

    print(f"\nInterpretation:")
    if avg_percent < 0:
        print(f"  ✓ MLPY reduces code by {abs(avg_percent):.1f}% on average")
    else:
        print(f"  ✗ MLPY increases code by {avg_percent:.1f}% on average")

    print()
    print("=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)

    return df_results


if __name__ == '__main__':
    results = analyze_usability()
