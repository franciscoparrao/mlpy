"""
Verificación completa de funcionalidad de MLPY
Este script verifica sistemáticamente todos los componentes implementados.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("VERIFICACIÓN COMPLETA DE MLPY")
print("="*60)

# Lista de componentes a verificar
componentes = {
    "1. Core": [
        ("Base", "mlpy.base.MLPYObject"),
        ("Registry", "mlpy.utils.registry.Registry"),
        ("Logging", "mlpy.utils.logging")
    ],
    
    "2. Data Backend": [
        ("DataBackend Base", "mlpy.backends.base.DataBackend"),
        ("Pandas Backend", "mlpy.backends.pandas_backend.DataBackendPandas"),
        ("NumPy Backend", "mlpy.backends.numpy_backend.DataBackendNumPy"),
        ("Dask Backend", "mlpy.backends.dask_backend.DataBackendDask"),
        ("Vaex Backend", "mlpy.backends.vaex_backend.DataBackendVaex")
    ],
    
    "3. Tasks": [
        ("Task Base", "mlpy.tasks.base.Task"),
        ("TaskClassif", "mlpy.tasks.supervised.TaskClassif"),
        ("TaskRegr", "mlpy.tasks.supervised.TaskRegr"),
        ("Big Data Tasks", "mlpy.tasks.big_data.create_dask_task")
    ],
    
    "4. Learners": [
        ("Learner Base", "mlpy.learners.base.Learner"),
        ("Sklearn Wrapper", "mlpy.learners.sklearn.learner_sklearn"),
        ("Native Decision Tree", "mlpy.learners.native.decision_tree.DecisionTreeClassifier"),
        ("Native Linear Regression", "mlpy.learners.native.linear_regression.LinearRegression"),
        ("Native KNN", "mlpy.learners.native.knn.KNeighborsClassifier"),
        ("Native Naive Bayes", "mlpy.learners.native.naive_bayes.GaussianNB"),
        ("TGPY Wrapper", "mlpy.learners.tgpy_wrapper.LearnerTGPRegressor")
    ],
    
    "5. Measures": [
        ("Measure Base", "mlpy.measures.base.Measure"),
        ("Classification Measures", "mlpy.measures.classification.MeasureClassifAcc"),
        ("Regression Measures", "mlpy.measures.regression.MeasureRegrRMSE")
    ],
    
    "6. Resampling": [
        ("Resampling Base", "mlpy.resamplings.base.Resampling"),
        ("CV", "mlpy.resamplings.cv.ResamplingCV"),
        ("Holdout", "mlpy.resamplings.holdout.ResamplingHoldout"),
        ("Bootstrap", "mlpy.resamplings.bootstrap.ResamplingBootstrap")
    ],
    
    "7. Pipelines": [
        ("PipeOp Base", "mlpy.pipelines.base.PipeOp"),
        ("Basic Operators", "mlpy.pipelines.operators.PipeOpScale"),
        ("Advanced Operators", "mlpy.pipelines.advanced_operators.PipeOpPCA"),
        ("Lazy Operators", "mlpy.pipelines.lazy_ops.LazyPipeOp"),
        ("GraphLearner", "mlpy.pipelines.graph.GraphLearner")
    ],
    
    "8. AutoML": [
        ("Tuning", "mlpy.automl.tuning.TunerGrid"),
        ("Feature Engineering", "mlpy.automl.feature_engineering.AutoFeaturesNumeric")
    ],
    
    "9. Parallel": [
        ("Backend Base", "mlpy.parallel.base.Backend"),
        ("Threading", "mlpy.parallel.threading.BackendThreading"),
        ("Multiprocessing", "mlpy.parallel.multiprocessing.BackendMultiprocessing"),
        ("Joblib", "mlpy.parallel.joblib.BackendJoblib")
    ],
    
    "10. Callbacks": [
        ("Callback Base", "mlpy.callbacks.base.Callback"),
        ("History", "mlpy.callbacks.history.CallbackHistory"),
        ("Progress", "mlpy.callbacks.progress.CallbackProgress"),
        ("Early Stopping", "mlpy.callbacks.early_stopping.CallbackEarlyStopping")
    ],
    
    "11. Visualization": [
        ("Visualizer Base", "mlpy.visualizations.base.Visualizer"),
        ("Benchmark Viz", "mlpy.visualizations.benchmark.BenchmarkVisualizer"),
        ("Resample Viz", "mlpy.visualizations.resample.ResampleVisualizer"),
        ("Tuning Viz", "mlpy.visualizations.tuning.TuningVisualizer")
    ],
    
    "12. Interpretability": [
        ("Interpreter Base", "mlpy.interpretability.base.Interpreter"),
        ("SHAP", "mlpy.interpretability.shap_interpreter.SHAPInterpreter"),
        ("LIME", "mlpy.interpretability.lime_interpreter.LIMEInterpreter")
    ],
    
    "13. Persistence": [
        ("Save/Load", "mlpy.persistence.base.save_model"),
        ("Serializers", "mlpy.persistence.serializers.PickleSerializer"),
        ("Model Registry", "mlpy.persistence.utils.ModelRegistry"),
        ("Export Package", "mlpy.persistence.utils.export_model_package")
    ],
    
    "14. Core Functions": [
        ("Resample", "mlpy.resample.resample"),
        ("Benchmark", "mlpy.benchmark.benchmark")
    ]
}

# Verificar cada componente
total = 0
exitosos = 0
fallidos = []

for categoria, items in componentes.items():
    print(f"\n{categoria}")
    print("-" * 40)
    
    for nombre, path in items:
        total += 1
        try:
            # Intentar importar el componente
            parts = path.split('.')
            module_path = '.'.join(parts[:-1])
            class_name = parts[-1]
            
            module = __import__(module_path, fromlist=[class_name])
            component = getattr(module, class_name)
            
            print(f"[OK] {nombre:<30}")
            exitosos += 1
            
        except Exception as e:
            print(f"[FALLO] {nombre:<30}: {str(e)[:50]}")
            fallidos.append((nombre, path, str(e)))

# Resumen
print("\n" + "="*60)
print("RESUMEN DE VERIFICACIÓN")
print("="*60)
print(f"Total de componentes: {total}")
print(f"Exitosos: {exitosos} ({exitosos/total*100:.1f}%)")
print(f"Fallidos: {len(fallidos)} ({len(fallidos)/total*100:.1f}%)")

if fallidos:
    print("\nComponentes con fallos:")
    for nombre, path, error in fallidos[:10]:  # Mostrar máximo 10
        print(f"  - {nombre}: {error[:80]}")

# Verificar ejemplos y tests
print("\n" + "="*60)
print("VERIFICACIÓN DE ARCHIVOS")
print("="*60)

import os
from pathlib import Path

base_path = Path(".")

# Contar archivos
file_counts = {
    "Python (mlpy/)": len(list(base_path.glob("mlpy/**/*.py"))),
    "Tests": len(list(base_path.glob("tests/**/*.py"))),
    "Examples": len(list(base_path.glob("examples/**/*.py"))),
    "Notebooks": len(list(base_path.glob("examples/notebooks/*.ipynb"))),
    "Documentation": len(list(base_path.glob("docs/**/*.md")))
}

for tipo, count in file_counts.items():
    print(f"{tipo:<20}: {count} archivos")

# Verificar dependencias opcionales
print("\n" + "="*60)
print("DEPENDENCIAS OPCIONALES")
print("="*60)

optional_deps = {
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "dask": "dask",
    "vaex": "vaex",
    "shap": "shap",
    "lime": "lime"
}

for name, module in optional_deps.items():
    try:
        __import__(module)
        print(f"[OK] {name:<15} instalado")
    except ImportError:
        print(f"[NO] {name:<15} no instalado")

print("\n" + "="*60)
print("VERIFICACIÓN COMPLETADA")
print("="*60)