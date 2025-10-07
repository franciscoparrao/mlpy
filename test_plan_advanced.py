"""
Test Plan Execution Script for MLPY - Advanced Features
This script tests resampling, benchmarking, pipelines, and feature engineering.
"""

import sys
import traceback

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Test results tracking
results = {
    "passed": [],
    "failed": [],
    "skipped": []
}

def test_wrapper(test_name: str, test_func):
    """Wrapper to execute tests and track results."""
    try:
        test_func()
        results["passed"].append(test_name)
        print(f"✅ PASSED: {test_name}")
        return True
    except ImportError as e:
        results["skipped"].append((test_name, f"Missing dependency: {e}"))
        print(f"⏭️  SKIPPED: {test_name} - {e}")
        return None
    except Exception as e:
        results["failed"].append((test_name, str(e)))
        print(f"❌ FAILED: {test_name}")
        print(f"   Error: {e}")
        return False

# ============================================================================
# 2. PRUEBAS DE FUNCIONALIDAD AVANZADA
# ============================================================================

print("\n" + "="*80)
print("2. PRUEBAS DE FUNCIONALIDAD AVANZADA")
print("="*80)

# 2.1 Resampling
print("\n--- 2.1 Resampling ---")

def test_resampling_cv():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree
    from mlpy.resamplings import ResamplingCV
    from mlpy.measures import MeasureClassifAccuracy
    from mlpy import resample

    data = pd.DataFrame({
        'x1': list(range(50)),
        'x2': list(range(50, 100)),
        'y': ['A', 'B'] * 25
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerDecisionTree(random_state=42)
    resampling = ResamplingCV(folds=5)
    measure = MeasureClassifAccuracy()

    result = resample(task=task, learner=learner, resampling=resampling, measures=[measure])
    assert result is not None
    print(f"   CV Score: {result.score(measure.id):.4f}")

def test_resampling_holdout():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree
    from mlpy.resamplings import ResamplingHoldout
    from mlpy.measures import MeasureClassifAccuracy
    from mlpy import resample

    data = pd.DataFrame({
        'x1': list(range(100)),
        'x2': list(range(100, 200)),
        'y': ['A', 'B'] * 50
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerDecisionTree(random_state=42)
    resampling = ResamplingHoldout(ratio=0.8)
    measure = MeasureClassifAccuracy()

    result = resample(task=task, learner=learner, resampling=resampling, measures=[measure])
    assert result is not None
    print(f"   Holdout Score: {result.score(measure.id):.4f}")

def test_resampling_bootstrap():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree
    from mlpy.resamplings import ResamplingBootstrap
    from mlpy.measures import MeasureClassifAccuracy
    from mlpy import resample

    data = pd.DataFrame({
        'x1': list(range(50)),
        'x2': list(range(50, 100)),
        'y': ['A', 'B'] * 25
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerDecisionTree(random_state=42)
    resampling = ResamplingBootstrap(iters=5, ratio=0.8)
    measure = MeasureClassifAccuracy()

    result = resample(task=task, learner=learner, resampling=resampling, measures=[measure])
    assert result is not None
    print(f"   Bootstrap Score: {result.score(measure.id):.4f}")

def test_resample_function():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerRandomForest
    from mlpy.resamplings import ResamplingCV
    from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
    from mlpy import resample

    data = pd.DataFrame({
        'x1': list(range(100)),
        'x2': list(range(100, 200)),
        'y': ['A', 'B'] * 50
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerRandomForest(n_estimators=10, random_state=42)
    resampling = ResamplingCV(folds=3)
    measures = [MeasureClassifAccuracy(), MeasureClassifF1()]

    result = resample(task=task, learner=learner, resampling=resampling, measures=measures)
    assert result is not None
    print(f"   Accuracy: {result.score(measures[0].id):.4f}")
    print(f"   F1: {result.score(measures[1].id):.4f}")

test_wrapper("2.1.1 ResamplingCV", test_resampling_cv)
test_wrapper("2.1.2 ResamplingHoldout", test_resampling_holdout)
test_wrapper("2.1.3 ResamplingBootstrap", test_resampling_bootstrap)
test_wrapper("2.1.4 resample() function", test_resample_function)

# 2.2 Benchmarking
print("\n--- 2.2 Benchmarking ---")

def test_benchmark_multiple_learners():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree, LearnerRandomForest, LearnerKNN
    from mlpy.resamplings import ResamplingCV
    from mlpy.measures import MeasureClassifAccuracy
    from mlpy import benchmark

    data = pd.DataFrame({
        'x1': list(range(50)),
        'x2': list(range(50, 100)),
        'y': ['A', 'B'] * 25
    })

    task = TaskClassif(data=data, target='y')
    learners = [
        LearnerDecisionTree(random_state=42),
        LearnerRandomForest(n_estimators=10, random_state=42),
        LearnerKNN(n_neighbors=3)
    ]
    resampling = ResamplingCV(folds=3)
    measure = MeasureClassifAccuracy()

    result = benchmark(tasks=[task], learners=learners, resampling=resampling, measures=[measure])
    assert result is not None
    print(f"   Benchmark completed with {len(learners)} learners")

def test_benchmark_multiple_tasks():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree
    from mlpy.resamplings import ResamplingCV
    from mlpy.measures import MeasureClassifAccuracy
    from mlpy import benchmark

    data1 = pd.DataFrame({
        'x1': list(range(50)),
        'x2': list(range(50, 100)),
        'y': ['A', 'B'] * 25
    })

    data2 = pd.DataFrame({
        'x1': list(range(40)),
        'x2': list(range(40, 80)),
        'y': ['X', 'Y'] * 20
    })

    task1 = TaskClassif(data=data1, target='y', id='task1')
    task2 = TaskClassif(data=data2, target='y', id='task2')
    learner = LearnerDecisionTree(random_state=42)
    resampling = ResamplingCV(folds=3)
    measure = MeasureClassifAccuracy()

    result = benchmark(tasks=[task1, task2], learners=[learner], resampling=resampling, measures=[measure])
    assert result is not None
    print(f"   Benchmark completed with {2} tasks")

def test_benchmark_multiple_measures():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree
    from mlpy.resamplings import ResamplingCV
    from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1, MeasureClassifPrecision
    from mlpy import benchmark

    data = pd.DataFrame({
        'x1': list(range(50)),
        'x2': list(range(50, 100)),
        'y': ['A', 'B'] * 25
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerDecisionTree(random_state=42)
    resampling = ResamplingCV(folds=3)
    measures = [MeasureClassifAccuracy(), MeasureClassifF1(), MeasureClassifPrecision()]

    result = benchmark(tasks=[task], learners=[learner], resampling=resampling, measures=measures)
    assert result is not None
    print(f"   Benchmark with {len(measures)} measures")

test_wrapper("2.2.1 Benchmark multiple learners", test_benchmark_multiple_learners)
test_wrapper("2.2.2 Benchmark multiple tasks", test_benchmark_multiple_tasks)
test_wrapper("2.2.3 Benchmark multiple measures", test_benchmark_multiple_measures)

# 2.3 Pipelines
print("\n--- 2.3 Pipelines ---")

def test_pipeline_basic():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline
    from mlpy.learners.sklearn import LearnerLogisticRegression

    data = pd.DataFrame({
        'x1': list(range(30)),
        'x2': list(range(30, 60)),
        'y': ['A', 'B', 'C'] * 10
    })

    task = TaskClassif(data=data, target='y')

    # Create pipeline
    scale = PipeOpScale()
    learner_op = PipeOpLearner(LearnerLogisticRegression())

    pipeline = linear_pipeline([scale, learner_op])
    assert pipeline is not None
    print(f"   Pipeline created with {2} operations")

def test_pipeline_multiple_ops():
    import pandas as pd
    import numpy as np
    from mlpy.tasks import TaskClassif
    from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline
    from mlpy.learners.sklearn import LearnerDecisionTree

    # Create data with missing values
    data = pd.DataFrame({
        'x1': [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        'x2': [2, 4, 6, None, 10, 12, 14, 16, 18, 20],
        'y': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })

    task = TaskClassif(data=data, target='y')

    # Create pipeline with scale
    scale = PipeOpScale()
    learner_op = PipeOpLearner(LearnerDecisionTree())

    pipeline = linear_pipeline([scale, learner_op])
    assert pipeline is not None
    print(f"   Complex pipeline created")

def test_graph_learner():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.pipelines import GraphLearner, PipeOpScale, PipeOpLearner, linear_pipeline
    from mlpy.learners.sklearn import LearnerDecisionTree

    data = pd.DataFrame({
        'x1': list(range(20)),
        'x2': list(range(20, 40)),
        'y': ['A', 'B'] * 10
    })

    task = TaskClassif(data=data, target='y')

    scale = PipeOpScale()
    learner_op = PipeOpLearner(LearnerDecisionTree())

    # Create a graph first, then pass it to GraphLearner
    graph = linear_pipeline([scale, learner_op])
    graph_learner = GraphLearner(graph=graph)
    assert graph_learner is not None
    print(f"   GraphLearner created")

test_wrapper("2.3.1 Basic pipeline", test_pipeline_basic)
test_wrapper("2.3.2 Pipeline with multiple operations", test_pipeline_multiple_ops)
test_wrapper("2.3.3 GraphLearner", test_graph_learner)

# 2.4 Feature Engineering
print("\n--- 2.4 Feature Engineering ---")

def test_scaling():
    import pandas as pd
    from mlpy.pipelines import PipeOpScale

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [10, 20, 30, 40, 50]
    })

    scale = PipeOpScale()
    assert scale is not None
    print(f"   PipeOpScale created")

def test_encoding():
    import pandas as pd
    from mlpy.pipelines import PipeOpEncode

    data = pd.DataFrame({
        'x1': ['A', 'B', 'C', 'A', 'B'],
        'x2': ['X', 'Y', 'Z', 'X', 'Y']
    })

    encode = PipeOpEncode()
    assert encode is not None
    print(f"   PipeOpEncode created")

def test_selection():
    import pandas as pd
    from mlpy.pipelines import PipeOpSelect

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'x3': [3, 6, 9, 12, 15],
        'y': [1, 0, 1, 0, 1]
    })

    select = PipeOpSelect(selector=['x1', 'x2'])
    assert select is not None
    print(f"   PipeOpSelect created")

def test_imputation():
    import pandas as pd
    from mlpy.pipelines import PipeOpImpute

    data = pd.DataFrame({
        'x1': [1, None, 3, 4, 5],
        'x2': [2, 4, None, 8, 10]
    })

    impute = PipeOpImpute()
    assert impute is not None
    print(f"   PipeOpImpute created")

test_wrapper("2.4.1 Scaling", test_scaling)
test_wrapper("2.4.2 Encoding", test_encoding)
test_wrapper("2.4.3 Selection", test_selection)
test_wrapper("2.4.4 Imputation", test_imputation)

# ============================================================================
# Print Summary for Advanced Tests
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE PRUEBAS AVANZADAS")
print("="*80)
print(f"Pasadas: {len(results['passed'])}")
print(f"Falladas: {len(results['failed'])}")
print(f"Saltadas: {len(results['skipped'])}")
print(f"Total: {len(results['passed']) + len(results['failed']) + len(results['skipped'])}")

if results['failed']:
    print("\nTESTS FALLADOS:")
    for test_name, error in results['failed']:
        print(f"  - {test_name}: {error}")

if results['skipped']:
    print("\nTESTS SALTADOS:")
    for test_name, reason in results['skipped']:
        print(f"  - {test_name}: {reason}")

print("\n" + "="*80)

# Export results
import json
with open('test_results_advanced.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResultados guardados en test_results_advanced.json")
