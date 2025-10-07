"""
Benchmark Final MLPY - Todos los Modelos
========================================
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BENCHMARK COMPLETO DE MLPY")
print("="*80)
print(f"Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Importar MLPY
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE, MeasureRegrR2
from mlpy.resamplings import ResamplingCV
from mlpy import benchmark

# Todos los learners
from mlpy.learners import (
    LearnerClassifFeatureless, LearnerRegrFeatureless
)

try:
    from mlpy.learners.sklearn import (
        # Clasificación
        LearnerLogisticRegression,
        LearnerDecisionTree,
        LearnerRandomForest,
        LearnerGradientBoosting,
        LearnerSVM,
        LearnerKNN,
        LearnerNaiveBayes,
        LearnerMLPClassifier,
        # Regresión
        LearnerLinearRegression,
        LearnerRidge,
        LearnerLasso,
        LearnerElasticNet,
        LearnerDecisionTreeRegressor,
        LearnerRandomForestRegressor,
        LearnerGradientBoostingRegressor,
        LearnerSVR,
        LearnerKNNRegressor,
        LearnerMLPRegressor
    )
    SKLEARN_AVAILABLE = True
    print("+ Modelos de scikit-learn disponibles")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("- Modelos de scikit-learn no disponibles")

# Crear datasets
np.random.seed(42)

# Dataset clasificación
n = 1000
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
X3 = np.random.uniform(-1, 1, n)
y = (2*X1 + X2 - 0.5*X3 + np.random.normal(0, 0.3, n)) > 0
data_classif = pd.DataFrame({
    'x1': X1, 'x2': X2, 'x3': X3,
    'target': ['Clase1' if yi else 'Clase0' for yi in y]
})

# Dataset regresión
X1 = np.random.uniform(-2, 2, n)
X2 = np.random.normal(0, 1, n)
X3 = np.random.exponential(1, n)
y = 3*X1 + X2**2 - 0.5*X3 + np.random.normal(0, 0.5, n)
data_regr = pd.DataFrame({
    'x1': X1, 'x2': X2, 'x3': X3,
    'target': y
})

print("\n1. DATASETS")
print("-" * 40)
print(f"Clasificación: {data_classif.shape[0]} muestras, {data_classif.shape[1]-1} características")
print(f"Regresión: {data_regr.shape[0]} muestras, {data_regr.shape[1]-1} características")

# Crear tareas
task_classif = TaskClassif(data=data_classif, target='target', id='classif')
task_regr = TaskRegr(data=data_regr, target='target', id='regr')

# Configurar todos los modelos
print("\n2. MODELOS")
print("-" * 40)

modelos_classif = [
    LearnerClassifFeatureless(id='baseline_mode', method='mode'),
    LearnerClassifFeatureless(id='baseline_weighted', method='weighted'),
]

if SKLEARN_AVAILABLE:
    modelos_classif.extend([
        # Regresión Logística (3 variantes)
        LearnerLogisticRegression(id='logreg', C=1.0),
        LearnerLogisticRegression(id='logreg_weak', C=10.0),
        LearnerLogisticRegression(id='logreg_strong', C=0.01),
        
        # Árboles de Decisión (3 variantes)
        LearnerDecisionTree(id='tree_shallow', max_depth=3),
        LearnerDecisionTree(id='tree_medium', max_depth=10),
        LearnerDecisionTree(id='tree_deep', max_depth=None),
        
        # Random Forest (3 variantes)
        LearnerRandomForest(id='rf_small', n_estimators=10),
        LearnerRandomForest(id='rf_medium', n_estimators=50),
        LearnerRandomForest(id='rf_large', n_estimators=100),
        
        # Gradient Boosting (2 variantes)
        LearnerGradientBoosting(id='gb_weak', n_estimators=50, learning_rate=0.1),
        LearnerGradientBoosting(id='gb_strong', n_estimators=100, learning_rate=0.05),
        
        # SVM (3 variantes)
        LearnerSVM(id='svm_linear', kernel='linear'),
        LearnerSVM(id='svm_rbf', kernel='rbf'),
        LearnerSVM(id='svm_poly', kernel='poly', degree=3),
        
        # KNN (3 variantes)
        LearnerKNN(id='knn_1', n_neighbors=1),
        LearnerKNN(id='knn_5', n_neighbors=5),
        LearnerKNN(id='knn_10', n_neighbors=10),
        
        # Otros
        LearnerNaiveBayes(id='naive_bayes'),
        LearnerMLPClassifier(id='mlp_small', hidden_layer_sizes=(50,)),
        LearnerMLPClassifier(id='mlp_large', hidden_layer_sizes=(100, 50)),
    ])

print(f"Modelos de clasificación: {len(modelos_classif)}")

modelos_regr = [
    LearnerRegrFeatureless(id='baseline_mean', method='mean'),
    LearnerRegrFeatureless(id='baseline_median', method='median'),
]

if SKLEARN_AVAILABLE:
    modelos_regr.extend([
        # Lineales (6 variantes)
        LearnerLinearRegression(id='linear'),
        LearnerRidge(id='ridge_weak', alpha=0.01),
        LearnerRidge(id='ridge', alpha=1.0),
        LearnerRidge(id='ridge_strong', alpha=10.0),
        LearnerLasso(id='lasso', alpha=0.01),
        LearnerElasticNet(id='elastic', alpha=0.1, l1_ratio=0.5),
        
        # Árboles (2 variantes)
        LearnerDecisionTreeRegressor(id='tree_reg', max_depth=10),
        LearnerDecisionTreeRegressor(id='tree_reg_deep', max_depth=None),
        
        # Ensemble (4 variantes)
        LearnerRandomForestRegressor(id='rf_reg_small', n_estimators=10),
        LearnerRandomForestRegressor(id='rf_reg', n_estimators=50),
        LearnerGradientBoostingRegressor(id='gb_reg', n_estimators=50),
        LearnerGradientBoostingRegressor(id='gb_reg_strong', n_estimators=100, learning_rate=0.05),
        
        # Otros (4 variantes)
        LearnerSVR(id='svr_linear', kernel='linear'),
        LearnerSVR(id='svr_rbf', kernel='rbf'),
        LearnerKNNRegressor(id='knn_reg_5', n_neighbors=5),
        LearnerMLPRegressor(id='mlp_reg', hidden_layer_sizes=(50,)),
    ])

print(f"Modelos de regresión: {len(modelos_regr)}")

# Ejecutar benchmarks
print("\n3. EJECUTANDO BENCHMARKS")
print("-" * 40)

# Clasificación - separar por velocidad
modelos_classif_rapidos = [m for m in modelos_classif if not any(x in m.id for x in ['svm', 'mlp'])]
modelos_classif_lentos = [m for m in modelos_classif if any(x in m.id for x in ['svm', 'mlp'])]

print("\n>>> Benchmark Clasificación (modelos rápidos)...")
bench_classif = benchmark(
    tasks=[task_classif],
    learners=modelos_classif_rapidos,
    resampling=ResamplingCV(folds=5, stratify=True),
    measures=[MeasureClassifAccuracy()]
)

if modelos_classif_lentos and SKLEARN_AVAILABLE:
    print("\n>>> Benchmark Clasificación (modelos lentos)...")
    bench_classif_lento = benchmark(
        tasks=[task_classif],
        learners=modelos_classif_lentos,
        resampling=ResamplingCV(folds=3, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )

# Regresión - separar por velocidad
modelos_regr_rapidos = [m for m in modelos_regr if not any(x in m.id for x in ['svr', 'mlp'])]
modelos_regr_lentos = [m for m in modelos_regr if any(x in m.id for x in ['svr', 'mlp'])]

print("\n>>> Benchmark Regresión (modelos rápidos)...")
bench_regr = benchmark(
    tasks=[task_regr],
    learners=modelos_regr_rapidos,
    resampling=ResamplingCV(folds=5),
    measures=[MeasureRegrR2(), MeasureRegrMSE()]
)

if modelos_regr_lentos and SKLEARN_AVAILABLE:
    print("\n>>> Benchmark Regresión (modelos lentos)...")
    bench_regr_lento = benchmark(
        tasks=[task_regr],
        learners=modelos_regr_lentos,
        resampling=ResamplingCV(folds=3),
        measures=[MeasureRegrR2(), MeasureRegrMSE()]
    )

# Resultados
print("\n" + "="*80)
print("RESULTADOS FINALES")
print("="*80)

# Clasificación
print("\n>>> CLASIFICACIÓN - Ranking por Accuracy")
print("-" * 70)

todos_classif = []
rankings = bench_classif.rank_learners('classif.acc')
for _, row in rankings.iterrows():
    todos_classif.append((row['learner'], row['mean_score'], 'CV-5'))

if 'bench_classif_lento' in locals():
    rankings = bench_classif_lento.rank_learners('classif.acc')
    for _, row in rankings.iterrows():
        todos_classif.append((row['learner'], row['mean_score'], 'CV-3'))

todos_classif.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Modelo':<25} {'Accuracy':<12} {'CV':<6}")
print("-" * 50)
for i, (modelo, mean, cv) in enumerate(todos_classif, 1):
    print(f"{i:<6} {modelo:<25} {mean:.4f}      {cv:<6}")

# Regresión
print("\n\n>>> REGRESIÓN - Ranking por R²")
print("-" * 70)

todos_regr = []
rankings = bench_regr.rank_learners('regr.rsq')
for _, row in rankings.iterrows():
    todos_regr.append((row['learner'], row['mean_score'], 'CV-5'))

if 'bench_regr_lento' in locals():
    rankings = bench_regr_lento.rank_learners('regr.rsq')
    for _, row in rankings.iterrows():
        todos_regr.append((row['learner'], row['mean_score'], 'CV-3'))

todos_regr.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Modelo':<25} {'R²':<12} {'CV':<6}")
print("-" * 50)
for i, (modelo, mean, cv) in enumerate(todos_regr, 1):
    print(f"{i:<6} {modelo:<25} {mean:.4f}      {cv:<6}")

# Análisis adicional
print("\n\n>>> ANÁLISIS ADICIONAL")
print("-" * 70)

# Mejor modelo de cada categoría
categorias_classif = {
    'Baseline': ['baseline'],
    'Lineal': ['logreg'],
    'Árboles': ['tree', 'rf', 'gb'],
    'Distancia': ['knn', 'svm'],
    'Probabilístico': ['naive'],
    'Neural': ['mlp']
}

print("\nMejor modelo por categoría (Clasificación):")
for cat, keywords in categorias_classif.items():
    mejor = max([m for m in todos_classif if any(k in m[0] for k in keywords)], 
                key=lambda x: x[1], default=None)
    if mejor:
        print(f"  {cat:<15}: {mejor[0]:<25} (Acc={mejor[1]:.4f})")

categorias_regr = {
    'Baseline': ['baseline'],
    'Lineal': ['linear', 'ridge', 'lasso', 'elastic'],
    'Árboles': ['tree', 'rf', 'gb'],
    'Otros': ['svr', 'knn', 'mlp']
}

print("\nMejor modelo por categoría (Regresión):")
for cat, keywords in categorias_regr.items():
    mejor = max([m for m in todos_regr if any(k in m[0] for k in keywords)], 
                key=lambda x: x[1], default=None)
    if mejor:
        print(f"  {cat:<15}: {mejor[0]:<25} (R²={mejor[1]:.4f})")

# Resumen final
print("\n\nRESUMEN FINAL:")
print("-" * 40)
print(f"Total modelos evaluados: {len(modelos_classif) + len(modelos_regr)}")
print(f"  - Clasificación: {len(modelos_classif)}")
print(f"  - Regresión: {len(modelos_regr)}")

if todos_classif:
    print(f"\nMejor modelo clasificación: {todos_classif[0][0]} (Acc={todos_classif[0][1]:.4f})")
if todos_regr:
    print(f"Mejor modelo regresión: {todos_regr[0][0]} (R²={todos_regr[0][1]:.4f})")

print(f"\nFinalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n+ Benchmark completado con éxito!")