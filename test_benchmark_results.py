"""
Test Benchmark Results
======================
"""

import numpy as np
import pandas as pd
from mlpy.tasks import TaskClassif
from mlpy.measures import MeasureClassifAccuracy
from mlpy.resamplings import ResamplingCV
from mlpy import benchmark
from mlpy.learners import LearnerClassifFeatureless

try:
    from mlpy.learners.sklearn import LearnerLogisticRegression, LearnerDecisionTree
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Create simple dataset
np.random.seed(42)
n = 200
X = np.random.normal(0, 1, (n, 2))
y = (X[:, 0] + X[:, 1] > 0)
data = pd.DataFrame(X, columns=['x1', 'x2'])
data['target'] = ['A' if yi else 'B' for yi in y]

# Create task
task = TaskClassif(data=data, target='target', id='test')

# Create learners
learners = [
    LearnerClassifFeatureless(id='baseline', method='mode')
]
if SKLEARN_AVAILABLE:
    learners.extend([
        LearnerLogisticRegression(id='logreg'),
        LearnerDecisionTree(id='tree', max_depth=5)
    ])

# Run benchmark
print("Running benchmark...")
bench_result = benchmark(
    tasks=[task],
    learners=learners,
    resampling=ResamplingCV(folds=3, stratify=True),
    measures=[MeasureClassifAccuracy()]
)

print("\nBenchmark result type:", type(bench_result))
print("Benchmark result attributes:", dir(bench_result))

# Try different ways to access results
print("\n\n1. Direct access to results:")
print("results attribute:", hasattr(bench_result, 'results'))
if hasattr(bench_result, 'results'):
    print("results type:", type(bench_result.results))
    print("results is dict:", isinstance(bench_result.results, dict))
    if isinstance(bench_result.results, dict):
        print("results keys:", list(bench_result.results.keys())[:5])
        print("Sample result:", list(bench_result.results.items())[0] if bench_result.results else "Empty")

print("\n\n2. Try rank_learners method:")
try:
    rankings = bench_result.rank_learners('classif.acc')
    print("Rankings type:", type(rankings))
    print("Rankings shape:", rankings.shape if hasattr(rankings, 'shape') else 'N/A')
    print("\nRankings:")
    print(rankings)
except Exception as e:
    print("Error ranking learners:", str(e))

print("\n\n3. Try rank_learners with task_id:")
try:
    rankings = bench_result.rank_learners('classif.acc', task_id='test')
    print("Rankings type:", type(rankings))
    print("Rankings shape:", rankings.shape if hasattr(rankings, 'shape') else 'N/A')
    print("\nRankings:")
    print(rankings)
except Exception as e:
    print("Error ranking learners with task_id:", str(e))

print("\n\n4. Try accessing columns directly:")
if hasattr(bench_result, 'results') and hasattr(bench_result.results, 'columns'):
    print("Available columns:", list(bench_result.results.columns))

print("\n\n5. Try aggregate method:")
if hasattr(bench_result, 'aggregate'):
    try:
        agg_results = bench_result.aggregate('classif.acc')
        print("Aggregate results:")
        print(agg_results)
    except Exception as e:
        print("Error aggregating:", str(e))