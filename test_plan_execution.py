"""
Test Plan Execution Script for MLPY
This script executes all tests from the test plan and updates the results.
"""

import sys
import traceback
from typing import Tuple, List

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
        traceback.print_exc()
        return False

# ============================================================================
# 1. PRUEBAS DE FUNCIONALIDAD BÁSICA
# ============================================================================

print("\n" + "="*80)
print("1️⃣  PRUEBAS DE FUNCIONALIDAD BÁSICA")
print("="*80)

# 1.1 Importación del Paquete
print("\n--- 1.1 Importación del Paquete ---")

def test_import_mlpy():
    import mlpy
    assert mlpy is not None

def test_mlpy_version():
    import mlpy
    assert hasattr(mlpy, '__version__')
    print(f"   Version: {mlpy.__version__}")

def test_import_core_modules():
    from mlpy import resample, benchmark
    from mlpy.tasks import TaskClassif, TaskRegr
    from mlpy.learners import Learner
    from mlpy.measures import MeasureClassifAccuracy

test_wrapper("1.1.1 Import mlpy base", test_import_mlpy)
test_wrapper("1.1.2 Verify version", test_mlpy_version)
test_wrapper("1.1.3 Import core modules", test_import_core_modules)

# 1.2 Tasks
print("\n--- 1.2 Tasks (Tareas) ---")

def test_create_task_classif():
    import pandas as pd
    from mlpy.tasks import TaskClassif

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': ['A', 'B', 'A', 'B', 'A']
    })

    task = TaskClassif(data=data, target='y')
    assert task is not None
    assert task.target_name == 'y'
    print(f"   Task created: {task.nrow} rows, {task.ncol} cols")

def test_create_task_regr():
    import pandas as pd
    from mlpy.tasks import TaskRegr

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': [1.5, 2.5, 3.5, 4.5, 5.5]
    })

    task = TaskRegr(data=data, target='y')
    assert task is not None
    assert task.target_name == 'y'

def test_task_properties():
    import pandas as pd
    from mlpy.tasks import TaskClassif

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': ['A', 'B', 'A', 'B', 'A']
    })

    task = TaskClassif(data=data, target='y')
    assert task.nrow == 5
    assert task.ncol == 3
    assert 'x1' in task.feature_names
    assert 'x2' in task.feature_names

def test_validated_task():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.validation import ValidatedTask

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': ['A', 'B', 'A', 'B', 'A']
    })

    task = TaskClassif(data=data, target='y')
    validated_task = ValidatedTask(task)
    assert validated_task is not None

test_wrapper("1.2.1 Create TaskClassif", test_create_task_classif)
test_wrapper("1.2.2 Create TaskRegr", test_create_task_regr)
test_wrapper("1.2.3 Verify task properties", test_task_properties)
test_wrapper("1.2.4 Validate task with ValidatedTask", test_validated_task)

# 1.3 Learners
print("\n--- 1.3 Learners (Aprendices) ---")

def test_create_learner_classif():
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from sklearn.ensemble import RandomForestClassifier

    learner = LearnerClassifSklearn(RandomForestClassifier(n_estimators=10, random_state=42))
    assert learner is not None

def test_create_learner_regr():
    from mlpy.learners.sklearn import LearnerRegrSklearn
    from sklearn.ensemble import RandomForestRegressor

    learner = LearnerRegrSklearn(RandomForestRegressor(n_estimators=10, random_state=42))
    assert learner is not None

def test_train_learner():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from sklearn.tree import DecisionTreeClassifier

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    learner.train(task)

    assert learner.model is not None
    print(f"   Learner trained successfully")

def test_generate_predictions():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from sklearn.tree import DecisionTreeClassifier

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    learner.train(task)

    predictions = learner.predict(task)
    assert predictions is not None
    assert len(predictions.response) == 10

def test_prediction_structure():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from sklearn.tree import DecisionTreeClassifier

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    learner.train(task)

    predictions = learner.predict(task)
    assert hasattr(predictions, 'truth')
    assert hasattr(predictions, 'response')

test_wrapper("1.3.1 Create classification learner", test_create_learner_classif)
test_wrapper("1.3.2 Create regression learner", test_create_learner_regr)
test_wrapper("1.3.3 Train learner", test_train_learner)
test_wrapper("1.3.4 Generate predictions", test_generate_predictions)
test_wrapper("1.3.5 Verify prediction structure", test_prediction_structure)

# 1.4 Measures
print("\n--- 1.4 Measures (Métricas) ---")

def test_measure_accuracy():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from mlpy.measures import MeasureClassifAccuracy
    from sklearn.tree import DecisionTreeClassifier

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    learner.train(task)
    predictions = learner.predict(task)

    measure = MeasureClassifAccuracy()
    score = measure.score(predictions.truth, predictions.response)
    assert 0 <= score <= 1
    print(f"   Accuracy: {score:.4f}")

def test_measure_mse():
    import pandas as pd
    from mlpy.tasks import TaskRegr
    from mlpy.learners.sklearn import LearnerRegrSklearn
    from mlpy.measures import MeasureRegrMSE
    from sklearn.tree import DecisionTreeRegressor

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    })

    task = TaskRegr(data=data, target='y')
    learner = LearnerRegrSklearn(DecisionTreeRegressor(random_state=42))
    learner.train(task)
    predictions = learner.predict(task)

    measure = MeasureRegrMSE()
    score = measure.score(predictions.truth, predictions.response)
    assert score >= 0
    print(f"   MSE: {score:.4f}")

def test_multiple_measures():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
    from sklearn.tree import DecisionTreeClassifier

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'y': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    learner.train(task)
    predictions = learner.predict(task)

    acc = MeasureClassifAccuracy()
    f1 = MeasureClassifF1()

    acc_score = acc.score(predictions.truth, predictions.response)
    f1_score = f1.score(predictions.truth, predictions.response)

    assert 0 <= acc_score <= 1
    assert 0 <= f1_score <= 1
    print(f"   Accuracy: {acc_score:.4f}, F1: {f1_score:.4f}")

test_wrapper("1.4.1 Calculate accuracy", test_measure_accuracy)
test_wrapper("1.4.2 Calculate MSE/RMSE", test_measure_mse)
test_wrapper("1.4.3 Calculate multiple measures", test_multiple_measures)

# 1.5 Predictions
print("\n--- 1.5 Predictions ---")

def test_prediction_classif():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from sklearn.tree import DecisionTreeClassifier

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': ['A', 'B', 'A', 'B', 'A']
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    learner.train(task)
    predictions = learner.predict(task)

    from mlpy.predictions import PredictionClassif
    assert isinstance(predictions, PredictionClassif)

def test_prediction_regr():
    import pandas as pd
    from mlpy.tasks import TaskRegr
    from mlpy.learners.sklearn import LearnerRegrSklearn
    from sklearn.tree import DecisionTreeRegressor

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': [1.5, 2.5, 3.5, 4.5, 5.5]
    })

    task = TaskRegr(data=data, target='y')
    learner = LearnerRegrSklearn(DecisionTreeRegressor(random_state=42))
    learner.train(task)
    predictions = learner.predict(task)

    from mlpy.predictions import PredictionRegr
    assert isinstance(predictions, PredictionRegr)

def test_prediction_truth_response():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerClassifSklearn
    from sklearn.tree import DecisionTreeClassifier

    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': ['A', 'B', 'A', 'B', 'A']
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerClassifSklearn(DecisionTreeClassifier(random_state=42))
    learner.train(task)
    predictions = learner.predict(task)

    assert predictions.truth is not None
    assert predictions.response is not None
    assert len(predictions.truth) == len(predictions.response)

test_wrapper("1.5.1 Verify PredictionClassif", test_prediction_classif)
test_wrapper("1.5.2 Verify PredictionRegr", test_prediction_regr)
test_wrapper("1.5.3 Access truth and response", test_prediction_truth_response)

# ============================================================================
# Print Summary for Basic Tests
# ============================================================================

print("\n" + "="*80)
print("📊 RESUMEN DE PRUEBAS BÁSICAS")
print("="*80)
print(f"✅ Pasadas: {len(results['passed'])}")
print(f"❌ Falladas: {len(results['failed'])}")
print(f"⏭️  Saltadas: {len(results['skipped'])}")
print(f"📈 Total: {len(results['passed']) + len(results['failed']) + len(results['skipped'])}")

if results['failed']:
    print("\n❌ TESTS FALLADOS:")
    for test_name, error in results['failed']:
        print(f"  - {test_name}: {error}")

if results['skipped']:
    print("\n⏭️  TESTS SALTADOS:")
    for test_name, reason in results['skipped']:
        print(f"  - {test_name}: {reason}")

print("\n" + "="*80)
