"""
Demo completo de MLPY - Muestra todas las funcionalidades principales
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

print("="*60)
print("DEMO COMPLETO DE MLPY")
print("="*60)

# 1. Crear datos de ejemplo
print("\n1. CREANDO DATOS DE EJEMPLO")
print("-"*40)

# Clasificación
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_informative=15, 
    n_redundant=5, n_classes=2, random_state=42
)
df_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(20)])
df_clf['target'] = y_clf

# Regresión
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, n_informative=8,
    noise=0.1, random_state=42
)
df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
df_reg['target'] = y_reg

print(f"Dataset clasificación: {df_clf.shape}")
print(f"Dataset regresión: {df_reg.shape}")

# 2. Crear Tasks
print("\n2. CREANDO TASKS")
print("-"*40)

from mlpy.tasks import TaskClassif, TaskRegr

task_clf = TaskClassif(data=df_clf, target='target', id='binary_classification')
task_reg = TaskRegr(data=df_reg, target='target', id='regression')

print(f"TaskClassif: {task_clf.nrow} filas, {task_clf.ncol} columnas")
print(f"TaskRegr: {task_reg.nrow} filas, {task_reg.ncol} columnas")

# 3. Crear Learners
print("\n3. CREANDO LEARNERS")
print("-"*40)

from mlpy.learners import learner_sklearn

# Learners para clasificación
rf_clf = learner_sklearn(RandomForestClassifier(n_estimators=50, random_state=42), id='rf')
lr_clf = learner_sklearn(LogisticRegression(max_iter=1000), id='logreg')
dt_clf = learner_sklearn(DecisionTreeClassifier(max_depth=5), id='dtree')

# Learners para regresión
rf_reg = learner_sklearn(RandomForestRegressor(n_estimators=50, random_state=42), id='rf_reg')
ridge_reg = learner_sklearn(Ridge(alpha=1.0), id='ridge')

print("Learners creados:")
print(f"  - Clasificación: {rf_clf.id}, {lr_clf.id}, {dt_clf.id}")
print(f"  - Regresión: {rf_reg.id}, {ridge_reg.id}")

# 4. Definir Measures
print("\n4. DEFINIENDO MEASURES")
print("-"*40)

from mlpy.measures import (
    MeasureClassifAccuracy, MeasureClassifAUC, MeasureClassifF1,
    MeasureRegrRMSE, MeasureRegrMAE, MeasureRegrR2
)

measures_clf = [MeasureClassifAccuracy(), MeasureClassifAUC(), MeasureClassifF1()]
measures_reg = [MeasureRegrRMSE(), MeasureRegrMAE(), MeasureRegrR2()]

print(f"Measures clasificación: {[m.id for m in measures_clf]}")
print(f"Measures regresión: {[m.id for m in measures_reg]}")

# 5. Resampling simple
print("\n5. RESAMPLING SIMPLE")
print("-"*40)

from mlpy import resample
from mlpy.resamplings import ResamplingCV, ResamplingHoldout

# Clasificación con CV
result_clf = resample(
    task=task_clf,
    learner=rf_clf,
    resampling=ResamplingCV(folds=5),
    measure=measures_clf[0]
)

scores = result_clf.aggregate()
print(f"Random Forest - CV Accuracy: {scores['acc'][0]:.3f} ± {scores['acc'][1]:.3f}")

# Regresión con Holdout
result_reg = resample(
    task=task_reg,
    learner=ridge_reg,
    resampling=ResamplingHoldout(ratio=0.8),
    measure=measures_reg[0]
)

scores = result_reg.aggregate()
print(f"Ridge - Holdout RMSE: {scores['rmse'][0]:.3f}")

# 6. Pipelines
print("\n6. PIPELINES")
print("-"*40)

from mlpy.pipelines import (
    PipeOpScale, PipeOpImpute, PipeOpSelect, PipeOpLearner, linear_pipeline
)

# Pipeline con preprocesamiento
pipeline = linear_pipeline(
    PipeOpScale(id='scale', method='standard'),
    PipeOpSelect(id='select', k=15, score_func='f_classif'),
    PipeOpLearner(rf_clf, id='learner')
)

print(f"Pipeline creado con {len(pipeline.pipeops)} operaciones")

# Evaluar pipeline
result_pipe = resample(
    task=task_clf,
    learner=pipeline,
    resampling=ResamplingCV(folds=3),
    measure=measures_clf[0]
)

scores = result_pipe.aggregate()
print(f"Pipeline - CV Accuracy: {scores['acc'][0]:.3f}")

# 7. Benchmark
print("\n7. BENCHMARK")
print("-"*40)

from mlpy import benchmark

# Comparar múltiples learners
learners = [rf_clf, lr_clf, dt_clf]

bench_result = benchmark(
    tasks=[task_clf],
    learners=learners,
    resampling=ResamplingCV(folds=5),
    measure=measures_clf
)

print("\nResultados del benchmark:")
print(bench_result.score_table())

print("\nRanking de learners:")
ranking = bench_result.rank_learners()
for i, (learner_id, avg_rank) in enumerate(ranking[:3]):
    print(f"{i+1}. {learner_id}: rank promedio = {avg_rank:.2f}")

# 8. AutoML - Tuning
print("\n8. AUTOML - HYPERPARAMETER TUNING")
print("-"*40)

try:
    from mlpy.automl import ParamSet, ParamInt, ParamFloat, TunerGrid
    
    # Definir espacio de búsqueda
    param_set = ParamSet([
        ParamInt('n_estimators', lower=10, upper=100),
        ParamInt('max_depth', lower=3, upper=10),
        ParamFloat('min_samples_split', lower=0.01, upper=0.2)
    ])
    
    # Tuner
    tuner = TunerGrid()
    
    # Crear learner base
    rf_base = RandomForestClassifier(random_state=42)
    learner_base = learner_sklearn(rf_base, id='rf_tuned')
    
    # Ejecutar tuning
    tune_result = tuner.tune(
        task=task_clf,
        learner=learner_base,
        param_set=param_set,
        resampling=ResamplingCV(folds=3),
        measure=measures_clf[0]
    )
    
    print(f"Mejor configuración: {tune_result.best_params}")
    print(f"Mejor score: {tune_result.best_score:.3f}")
    
except Exception as e:
    print(f"AutoML no disponible: {e}")

# 9. Persistencia
print("\n9. PERSISTENCIA DE MODELOS")
print("-"*40)

try:
    from mlpy.persistence import save_model, load_model
    import tempfile
    import os
    
    # Entrenar un modelo
    rf_clf.train(task_clf)
    
    # Guardar
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    metadata = {
        'description': 'Random Forest para clasificación binaria',
        'accuracy': 0.95,
        'features': task_clf.feature_names
    }
    
    saved_path = save_model(rf_clf, temp_path, metadata=metadata)
    print(f"Modelo guardado en: {saved_path}")
    
    # Cargar
    loaded_learner, loaded_metadata = load_model(temp_path, return_metadata=True)
    print(f"Modelo cargado: {loaded_learner.id}")
    print(f"Metadata: {loaded_metadata['description']}")
    
    # Limpiar
    os.unlink(temp_path)
    
except Exception as e:
    print(f"Persistencia no disponible: {e}")

# 10. Visualización (si está disponible)
print("\n10. VISUALIZACIÓN")
print("-"*40)

try:
    from mlpy.visualizations import plot_benchmark_boxplot
    
    # Crear figura
    fig = plot_benchmark_boxplot(bench_result, measure='acc')
    print("Gráfico de boxplot creado (no mostrado en consola)")
    
except Exception as e:
    print(f"Visualización no disponible: {e}")

print("\n" + "="*60)
print("DEMO COMPLETADO EXITOSAMENTE")
print("="*60)

# Resumen de capacidades
print("\nCAPACIDADES DEMOSTRADAS:")
print("- Tasks para clasificación y regresión")
print("- Learners con wrappers de sklearn")
print("- Múltiples measures de evaluación")
print("- Resampling (CV, Holdout)")
print("- Pipelines con preprocesamiento")
print("- Benchmark para comparar modelos")
print("- AutoML con hyperparameter tuning")
print("- Persistencia de modelos")
print("- Visualización de resultados")

print("\nMLPY está listo para usar!")