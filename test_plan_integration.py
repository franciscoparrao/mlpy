"""
Test Plan Execution Script for MLPY - Integration & Optional Features
This script tests end-to-end workflows and optional features.
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
        print(f"⏭️  SKIPPED: {test_name}")
        return None
    except Exception as e:
        results["failed"].append((test_name, str(e)))
        print(f"❌ FAILED: {test_name}")
        print(f"   Error: {e}")
        return False

# ============================================================================
# 3. PRUEBAS DE INTEGRACIÓN
# ============================================================================

print("\n" + "="*80)
print("3. PRUEBAS DE INTEGRACION")
print("="*80)

# 3.1 Workflows Completos
print("\n--- 3.1 Workflows Completos ---")

def test_classification_workflow():
    import pandas as pd
    from sklearn.datasets import make_classification
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerRandomForest
    from mlpy.measures import MeasureClassifAccuracy

    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)

    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(20)])
    data['target'] = y

    # Create task
    task = TaskClassif(data=data, target='target')

    # Create and train learner
    learner = LearnerRandomForest(n_estimators=50, random_state=42)
    learner.train(task)

    # Make predictions
    predictions = learner.predict(task)

    # Evaluate
    measure = MeasureClassifAccuracy()
    score = measure.score(predictions.truth, predictions.response)

    assert 0.5 <= score <= 1.0
    print(f"   Classification Score: {score:.4f}")

def test_regression_workflow():
    import pandas as pd
    from sklearn.datasets import make_regression
    from mlpy.tasks import TaskRegr
    from mlpy.learners.sklearn import LearnerRandomForestRegressor
    from mlpy.measures import MeasureRegrMSE

    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=20, n_informative=15,
                          random_state=42)

    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(20)])
    data['target'] = y

    # Create task
    task = TaskRegr(data=data, target='target')

    # Create and train learner
    learner = LearnerRandomForestRegressor(n_estimators=50, random_state=42)
    learner.train(task)

    # Make predictions
    predictions = learner.predict(task)

    # Evaluate
    measure = MeasureRegrMSE()
    score = measure.score(predictions.truth, predictions.response)

    assert score >= 0
    print(f"   Regression MSE: {score:.4f}")

def test_multiclass_classification():
    import pandas as pd
    from sklearn.datasets import make_classification
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerRandomForest
    from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1

    # Generate multiclass data
    X, y = make_classification(n_samples=100, n_features=20, n_informative=15,
                               n_classes=3, n_clusters_per_class=1, random_state=42)

    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(20)])
    data['target'] = y

    # Create task
    task = TaskClassif(data=data, target='target')

    # Create and train learner
    learner = LearnerRandomForest(n_estimators=50, random_state=42)
    learner.train(task)

    # Make predictions
    predictions = learner.predict(task)

    # Evaluate with multiple measures
    acc = MeasureClassifAccuracy()
    f1 = MeasureClassifF1()

    acc_score = acc.score(predictions.truth, predictions.response)
    f1_score = f1.score(predictions.truth, predictions.response)

    assert 0 <= acc_score <= 1
    assert 0 <= f1_score <= 1
    print(f"   Accuracy: {acc_score:.4f}, F1: {f1_score:.4f}")

test_wrapper("3.1.1 Classification workflow end-to-end", test_classification_workflow)
test_wrapper("3.1.2 Regression workflow end-to-end", test_regression_workflow)
test_wrapper("3.1.3 Multiclass classification", test_multiclass_classification)

# 3.2 Interoperabilidad
print("\n--- 3.2 Interoperabilidad ---")

def test_sklearn_integration():
    import pandas as pd
    from sklearn.datasets import load_iris
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree

    # Load sklearn dataset
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    # Use with MLPY
    task = TaskClassif(data=data, target='target')
    learner = LearnerDecisionTree(random_state=42)
    learner.train(task)

    predictions = learner.predict(task)
    assert len(predictions.response) == len(iris.target)
    print(f"   Sklearn integration: {len(predictions.response)} predictions")

def test_pandas_dataframe():
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerKNN

    # Create pandas DataFrame
    df = pd.DataFrame({
        'feat1': list(range(50)),
        'feat2': list(range(50, 100)),
        'feat3': list(range(100, 150)),
        'label': ['A', 'B'] * 25
    })

    task = TaskClassif(data=df, target='label')
    learner = LearnerKNN(n_neighbors=5)
    learner.train(task)

    predictions = learner.predict(task)
    assert predictions is not None
    print(f"   Pandas compatibility: {len(predictions.response)} samples")

def test_numpy_arrays():
    import numpy as np
    import pandas as pd
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree

    # Create numpy arrays
    X = np.random.rand(50, 3)
    y = np.array(['A', 'B'] * 25)

    # Convert to DataFrame (MLPY expects DataFrames)
    data = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    data['target'] = y

    task = TaskClassif(data=data, target='target')
    learner = LearnerDecisionTree(random_state=42)
    learner.train(task)

    predictions = learner.predict(task)
    assert predictions is not None
    print(f"   Numpy compatibility: {len(predictions.response)} samples")

test_wrapper("3.2.1 Sklearn integration", test_sklearn_integration)
test_wrapper("3.2.2 Pandas DataFrame compatibility", test_pandas_dataframe)
test_wrapper("3.2.3 Numpy array compatibility", test_numpy_arrays)

# ============================================================================
# 4. PRUEBAS DE CARACTERÍSTICAS OPCIONALES
# ============================================================================

print("\n" + "="*80)
print("4. PRUEBAS DE CARACTERISTICAS OPCIONALES")
print("="*80)

# 4.1 Visualización
print("\n--- 4.1 Visualizacion ---")

def test_matplotlib_available():
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"   Matplotlib version: {matplotlib.__version__}")

def test_viz_imports():
    try:
        from mlpy.visualizations import plot_resample_boxplot
        print(f"   Visualization module available")
    except ImportError as e:
        raise ImportError(f"Visualization not available: {e}")

test_wrapper("4.1.1 Matplotlib available", test_matplotlib_available)
test_wrapper("4.1.2 Visualization imports", test_viz_imports)

# 4.2 Interpretabilidad
print("\n--- 4.2 Interpretabilidad ---")

def test_shap_available():
    import shap
    print(f"   SHAP version: {shap.__version__}")

def test_lime_available():
    import lime
    print(f"   LIME available")

def test_interpretability_imports():
    from mlpy.interpretability import Interpreter
    print(f"   Interpretability module available")

test_wrapper("4.2.1 SHAP available", test_shap_available)
test_wrapper("4.2.2 LIME available", test_lime_available)
test_wrapper("4.2.3 Interpretability imports", test_interpretability_imports)

# 4.3 Persistencia
print("\n--- 4.3 Persistencia ---")

def test_persistence_imports():
    from mlpy.persistence import save_model, load_model
    print(f"   Persistence module available")

def test_save_load_model():
    import pandas as pd
    import tempfile
    import os
    from mlpy.tasks import TaskClassif
    from mlpy.learners.sklearn import LearnerDecisionTree
    from mlpy.persistence import save_model, load_model

    # Train a model
    data = pd.DataFrame({
        'x1': list(range(30)),
        'x2': list(range(30, 60)),
        'y': ['A', 'B', 'C'] * 10
    })

    task = TaskClassif(data=data, target='y')
    learner = LearnerDecisionTree(random_state=42)
    learner.train(task)

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_model.pkl')
        save_model(learner, filepath)

        # Load model
        loaded_learner = load_model(filepath)
        assert loaded_learner is not None
        print(f"   Model saved and loaded successfully")

test_wrapper("4.3.1 Persistence imports", test_persistence_imports)
test_wrapper("4.3.2 Save and load model", test_save_load_model)

# 4.4 Backends Alternativos
print("\n--- 4.4 Backends Alternativos ---")

def test_dask_available():
    import dask
    import dask.dataframe as dd
    print(f"   Dask version: {dask.__version__}")

def test_vaex_available():
    import vaex
    print(f"   Vaex version: {vaex.__version__}")

def test_pandas_backend():
    from mlpy.backends import PandasBackend
    backend = PandasBackend()
    assert backend is not None
    print(f"   Pandas backend available")

def test_numpy_backend():
    from mlpy.backends import NumpyBackend
    backend = NumpyBackend()
    assert backend is not None
    print(f"   Numpy backend available")

test_wrapper("4.4.1 Dask available", test_dask_available)
test_wrapper("4.4.2 Vaex available", test_vaex_available)
test_wrapper("4.4.3 Pandas backend", test_pandas_backend)
test_wrapper("4.4.4 Numpy backend", test_numpy_backend)

# 4.5 Learners Avanzados
print("\n--- 4.5 Learners Avanzados ---")

def test_xgboost_available():
    import xgboost as xgb
    print(f"   XGBoost version: {xgb.__version__}")

def test_lightgbm_available():
    import lightgbm as lgb
    print(f"   LightGBM version: {lgb.__version__}")

def test_catboost_available():
    import catboost
    print(f"   CatBoost version: {catboost.__version__}")

test_wrapper("4.5.1 XGBoost available", test_xgboost_available)
test_wrapper("4.5.2 LightGBM available", test_lightgbm_available)
test_wrapper("4.5.3 CatBoost available", test_catboost_available)

# 4.6 CLI
print("\n--- 4.6 CLI ---")

def test_cli_module():
    from mlpy.cli import main
    assert main is not None
    print(f"   CLI module available")

test_wrapper("4.6.1 CLI module", test_cli_module)

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE PRUEBAS DE INTEGRACION Y OPCIONALES")
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
with open('test_results_integration.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResultados guardados en test_results_integration.json")
