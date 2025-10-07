"""
Ejemplos Prácticos de MLPY
==========================

Este archivo contiene ejemplos de uso de las principales funcionalidades
del framework MLPY, incluyendo las características que fueron corregidas:
- Resampling (CV, Holdout, Bootstrap)
- Pipelines
- Benchmarking
- Clasificación multiclase
"""

import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, make_classification

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# =============================================================================
# Ejemplo 1: Clasificación Básica con Cross-Validation
# =============================================================================
print("="*80)
print("EJEMPLO 1: Clasificación con Cross-Validation")
print("="*80)

from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerDecisionTree
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy
from mlpy import resample

# Cargar datos
iris = load_iris()
data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
data['species'] = iris.target_names[iris.target]

# Crear task
task = TaskClassif(data=data, target='species', id='iris')
print(f"Task creado: {task.nrow} muestras, {task.ncol} features")

# Crear learner
learner = LearnerDecisionTree(max_depth=3, random_state=42)

# Evaluar con 5-fold CV
resampling = ResamplingCV(folds=5)
measure = MeasureClassifAccuracy()

result = resample(
    task=task,
    learner=learner,
    resampling=resampling,
    measures=[measure]
)

print(f"\n✅ Resultados de 5-fold CV:")
print(f"   Accuracy media: {result.score('classif.acc', average='mean'):.4f}")
print(f"   Desviación estándar: {result.score('classif.acc', average='std'):.4f}")
print(f"   Min: {result.score('classif.acc', average='min'):.4f}")
print(f"   Max: {result.score('classif.acc', average='max'):.4f}")
print(f"   Iteraciones completadas: {result.n_iters}")
print(f"   Errores: {result.n_errors}")

# =============================================================================
# Ejemplo 2: Pipeline con Feature Engineering
# =============================================================================
print("\n" + "="*80)
print("EJEMPLO 2: Pipeline con Scaling y Learner")
print("="*80)

from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline, GraphLearner
from mlpy.learners.sklearn import LearnerLogisticRegression

# Crear datos sintéticos
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
data['target'] = y

task = TaskClassif(data=data, target='target', id='synthetic')

# Crear pipeline: Scale -> Learner
scale_op = PipeOpScale()
learner_op = PipeOpLearner(LearnerLogisticRegression(max_iter=1000))

# Opción 1: Crear pipeline directamente
pipeline = linear_pipeline([scale_op, learner_op])
print(f"✅ Pipeline creado con {len([scale_op, learner_op])} operaciones")

# Opción 2: Usar GraphLearner para entrenar
graph_learner = GraphLearner(graph=pipeline)
graph_learner.train(task)
predictions = graph_learner.predict(task)

print(f"✅ Pipeline entrenado y predicciones generadas: {len(predictions.response)} muestras")

# =============================================================================
# Ejemplo 3: Benchmarking - Comparar Múltiples Modelos
# =============================================================================
print("\n" + "="*80)
print("EJEMPLO 3: Benchmarking - Comparar Múltiples Modelos")
print("="*80)

from mlpy.learners.sklearn import LearnerRandomForest, LearnerKNN
from mlpy.measures import MeasureClassifF1, MeasureClassifPrecision
from mlpy import benchmark

# Usar dataset wine
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['wine_class'] = wine.target_names[wine.target]

task = TaskClassif(data=data, target='wine_class', id='wine')

# Definir learners a comparar
learners = [
    LearnerDecisionTree(max_depth=5, random_state=42),
    LearnerRandomForest(n_estimators=50, max_depth=5, random_state=42),
    LearnerKNN(n_neighbors=5)
]

# Definir métricas
measures = [
    MeasureClassifAccuracy(),
    MeasureClassifF1(),  # Auto-detecta multiclase y usa average='weighted'
    MeasureClassifPrecision()
]

# Ejecutar benchmark
resampling = ResamplingCV(folds=3)

benchmark_result = benchmark(
    tasks=[task],
    learners=learners,
    resampling=resampling,
    measures=measures
)

print(f"\n✅ Benchmark completado:")
print(f"   Experimentos: {len(benchmark_result.results)}")
print(f"\n   Resultados por learner:")

# Mostrar resultados para cada métrica
for learner in learners:
    result = benchmark_result.get_result('wine', learner.id)
    if result:
        print(f"\n   {learner.id}:")
        print(f"      Accuracy: {result.score('classif.acc'):.4f}")
        print(f"      F1 Score: {result.score('classif.f1'):.4f}")
        print(f"      Precision: {result.score('classif.precision'):.4f}")

# =============================================================================
# Ejemplo 4: Resampling Strategies
# =============================================================================
print("\n" + "="*80)
print("EJEMPLO 4: Diferentes Estrategias de Resampling")
print("="*80)

from mlpy.resamplings import ResamplingHoldout, ResamplingBootstrap

# Datos simples
data = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100),
    'y': np.random.choice(['A', 'B'], 100)
})

task = TaskClassif(data=data, target='y', id='simple')
learner = LearnerDecisionTree(random_state=42)
measure = MeasureClassifAccuracy()

# 1. Holdout (80-20 split)
print("\n📊 Holdout (80-20 split):")
resampling_holdout = ResamplingHoldout(ratio=0.8)
result_holdout = resample(task, learner, resampling_holdout, [measure])
print(f"   Accuracy: {result_holdout.score('classif.acc'):.4f}")

# 2. 10-Fold Cross-Validation
print("\n📊 10-Fold Cross-Validation:")
resampling_cv = ResamplingCV(folds=10)
result_cv = resample(task, learner, resampling_cv, [measure])
print(f"   Accuracy media: {result_cv.score('classif.acc', average='mean'):.4f} ± {result_cv.score('classif.acc', average='std'):.4f}")

# 3. Bootstrap (10 iteraciones)
print("\n📊 Bootstrap (10 iteraciones, 80% muestra):")
resampling_bootstrap = ResamplingBootstrap(iters=10, ratio=0.8)
result_bootstrap = resample(task, learner, resampling_bootstrap, [measure])
print(f"   Accuracy media: {result_bootstrap.score('classif.acc', average='mean'):.4f} ± {result_bootstrap.score('classif.acc', average='std'):.4f}")

# =============================================================================
# Ejemplo 5: Clasificación Multiclase con Múltiples Métricas
# =============================================================================
print("\n" + "="*80)
print("EJEMPLO 5: Clasificación Multiclase (Auto-detección)")
print("="*80)

from mlpy.measures import MeasureClassifRecall

# Usar dataset iris (3 clases)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target_names[iris.target]

task = TaskClassif(data=data, target='species', id='iris_multiclass')
learner = LearnerRandomForest(n_estimators=100, random_state=42)

# Métricas con auto-detección de multiclase
measures = [
    MeasureClassifAccuracy(),
    MeasureClassifF1(),         # Auto-detecta 3 clases -> average='weighted'
    MeasureClassifPrecision(),
    MeasureClassifRecall()
]

result = resample(
    task=task,
    learner=learner,
    resampling=ResamplingCV(folds=5),
    measures=measures
)

print(f"\n✅ Resultados de clasificación multiclase (3 clases):")
print(f"   Accuracy:  {result.score('classif.acc'):.4f}")
print(f"   F1 Score:  {result.score('classif.f1'):.4f} (weighted)")
print(f"   Precision: {result.score('classif.precision'):.4f} (weighted)")
print(f"   Recall:    {result.score('classif.recall'):.4f} (weighted)")

# =============================================================================
# Ejemplo 6: Pipeline Completo con Feature Engineering
# =============================================================================
print("\n" + "="*80)
print("EJEMPLO 6: Pipeline Completo con Feature Engineering")
print("="*80)

from mlpy.pipelines import PipeOpEncode, PipeOpSelect, PipeOpImpute

# Crear datos con valores faltantes y categóricos
data = pd.DataFrame({
    'numeric1': [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
    'numeric2': [10, None, 30, 40, 50, 60, 70, 80, 90, 100],
    'numeric3': np.random.randn(10),
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

task = TaskClassif(data=data, target='target', id='complex')

# Pipeline: Impute -> Encode -> Scale -> Select -> Learner
impute_op = PipeOpImpute()
encode_op = PipeOpEncode()
scale_op = PipeOpScale()
select_op = PipeOpSelect(selector=['numeric1', 'numeric2', 'numeric3'])  # Seleccionar solo features numéricas después de encoding
learner_op = PipeOpLearner(LearnerLogisticRegression(max_iter=1000))

pipeline = linear_pipeline([
    impute_op,
    encode_op,
    scale_op,
    learner_op
])

graph_learner = GraphLearner(graph=pipeline)

# Evaluar pipeline completo
result = resample(
    task=task,
    learner=graph_learner,
    resampling=ResamplingCV(folds=5),
    measures=[MeasureClassifAccuracy()]
)

print(f"✅ Pipeline completo ejecutado:")
print(f"   Operaciones: Impute -> Encode -> Scale -> Learner")
print(f"   Accuracy: {result.score('classif.acc'):.4f}")

# =============================================================================
# Ejemplo 7: Workflow Completo de ML
# =============================================================================
print("\n" + "="*80)
print("EJEMPLO 7: Workflow Completo de Machine Learning")
print("="*80)

# 1. Cargar y preparar datos
print("\n1️⃣ Preparación de datos...")
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['quality'] = wine.target

task = TaskClassif(data=data, target='quality', id='wine_quality')
print(f"   ✅ Dataset: {task.nrow} muestras, {task.ncol-1} features, {len(np.unique(data['quality']))} clases")

# 2. Definir modelos candidatos
print("\n2️⃣ Definición de modelos...")
learners = {
    'Decision Tree': LearnerDecisionTree(max_depth=5, random_state=42),
    'Random Forest': LearnerRandomForest(n_estimators=100, max_depth=5, random_state=42),
    'KNN': LearnerKNN(n_neighbors=7)
}
print(f"   ✅ {len(learners)} modelos a comparar")

# 3. Benchmark con múltiples métricas
print("\n3️⃣ Ejecutando benchmark...")
measures = [
    MeasureClassifAccuracy(),
    MeasureClassifF1(),
    MeasureClassifPrecision(),
    MeasureClassifRecall()
]

benchmark_result = benchmark(
    tasks=[task],
    learners=list(learners.values()),
    resampling=ResamplingCV(folds=5),
    measures=measures
)

# 4. Analizar resultados
print("\n4️⃣ Resultados del benchmark:\n")

best_accuracy = 0
best_learner_name = None
best_learner = None

for learner_name, learner in learners.items():
    result = benchmark_result.get_result('wine_quality', learner.id)
    if result:
        acc = result.score('classif.acc')
        print(f"   {learner_name}:")
        print(f"      Accuracy:  {acc:.4f}")
        print(f"      F1:        {result.score('classif.f1'):.4f}")
        print(f"      Precision: {result.score('classif.precision'):.4f}")
        print(f"      Recall:    {result.score('classif.recall'):.4f}")
        print()

        if acc > best_accuracy:
            best_accuracy = acc
            best_learner_name = learner_name
            best_learner = learner

print(f"5️⃣ Mejor modelo: {best_learner_name}")
print(f"   Accuracy: {best_accuracy:.4f}")

# 6. Entrenar modelo final
print(f"\n6️⃣ Entrenando modelo final con todo el dataset...")
best_learner.train(task)
print(f"   ✅ Modelo entrenado y listo para producción")

print("\n" + "="*80)
print("¡Todos los ejemplos ejecutados exitosamente! 🎉")
print("="*80)
