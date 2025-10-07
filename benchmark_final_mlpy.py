"""
Benchmark Final MLPY
====================

Evaluación exhaustiva de todos los modelos disponibles en MLPY.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BENCHMARK FINAL MLPY - TODOS LOS MODELOS")
print("="*80)
print(f"Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Importar MLPY
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE, MeasureRegrR2
from mlpy.resamplings import ResamplingCV
from mlpy import benchmark

# Importar learners base
from mlpy.learners import LearnerClassifFeatureless, LearnerRegrFeatureless

# Intentar importar sklearn learners
try:
    from mlpy.learners.sklearn import (
        # Clasificación
        LearnerLogisticRegression, LearnerDecisionTree, LearnerRandomForest,
        LearnerGradientBoosting, LearnerSVM, LearnerKNN, LearnerNaiveBayes,
        LearnerMLPClassifier,
        # Regresión
        LearnerLinearRegression, LearnerRidge, LearnerLasso, LearnerElasticNet,
        LearnerDecisionTreeRegressor, LearnerRandomForestRegressor,
        LearnerGradientBoostingRegressor, LearnerSVR, LearnerKNNRegressor,
        LearnerMLPRegressor
    )
    SKLEARN_AVAILABLE = True
    print("+ Modelos de scikit-learn disponibles")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("- Modelos de scikit-learn no disponibles")

# ============================================================================
# 1. CREAR DATASETS
# ============================================================================
print("\n1. DATASETS SINTÉTICOS")
print("-" * 40)

np.random.seed(42)

# Dataset 1: Clasificación binaria (linealmente separable)
n1 = 1000
X1 = np.random.normal(0, 1, (n1, 3))
y1 = (2*X1[:, 0] + X1[:, 1] - 0.5*X1[:, 2] + np.random.normal(0, 0.3, n1)) > 0
data_bin_simple = pd.DataFrame(X1, columns=['x1', 'x2', 'x3'])
data_bin_simple['target'] = ['Clase1' if yi else 'Clase0' for yi in y1]

# Dataset 2: Clasificación binaria (no lineal)
n2 = 1200
X2 = np.random.uniform(-3, 3, (n2, 4))
y2 = ((X2[:, 0]**2 + X2[:, 1]**2 < 4) & (X2[:, 2] > 0)) | (X2[:, 3] > 1)
data_bin_complex = pd.DataFrame(X2, columns=['x1', 'x2', 'x3', 'x4'])
data_bin_complex['target'] = ['Interior' if yi else 'Exterior' for yi in y2]

# Dataset 3: Regresión simple
n3 = 800
X3 = np.random.uniform(-2, 2, (n3, 3))
y3 = 3*X3[:, 0] + X3[:, 1]**2 - 0.5*X3[:, 2] + np.random.normal(0, 0.5, n3)
data_reg_simple = pd.DataFrame(X3, columns=['x1', 'x2', 'x3'])
data_reg_simple['target'] = y3

# Dataset 4: Regresión compleja
n4 = 1000
X4 = np.random.normal(0, 1, (n4, 5))
y4 = (np.sin(2*X4[:, 0]) + X4[:, 1]**2 + np.exp(-np.abs(X4[:, 2])) + 
      0.3*X4[:, 3]*X4[:, 4] + np.random.normal(0, 0.3, n4))
data_reg_complex = pd.DataFrame(X4, columns=[f'x{i+1}' for i in range(5)])
data_reg_complex['target'] = y4

datasets = {
    'classif_simple': data_bin_simple,
    'classif_complex': data_bin_complex,
    'regr_simple': data_reg_simple,
    'regr_complex': data_reg_complex
}

for nombre, df in datasets.items():
    print(f"\n{nombre}:")
    print(f"  - Muestras: {len(df)}")
    print(f"  - Características: {df.shape[1]-1}")
    if 'classif' in nombre:
        dist = dict(df['target'].value_counts())
        print(f"  - Distribución: {dist}")

# ============================================================================
# 2. CREAR TAREAS
# ============================================================================
print("\n2. CONFIGURACIÓN DE TAREAS")
print("-" * 40)

# Tareas de clasificación
task_c1 = TaskClassif(data=data_bin_simple, target='target', id='classif_simple')
task_c2 = TaskClassif(data=data_bin_complex, target='target', id='classif_complex')

# Tareas de regresión
task_r1 = TaskRegr(data=data_reg_simple, target='target', id='regr_simple')
task_r2 = TaskRegr(data=data_reg_complex, target='target', id='regr_complex')

print("+ 2 tareas de clasificación creadas")
print("+ 2 tareas de regresión creadas")

# ============================================================================
# 3. CONFIGURAR LEARNERS
# ============================================================================
print("\n3. CONFIGURACIÓN DE MODELOS")
print("-" * 40)

# CLASIFICACIÓN
learners_classif = [
    # Baseline
    LearnerClassifFeatureless(id='baseline_mode', method='mode'),
    LearnerClassifFeatureless(id='baseline_weighted', method='weighted'),
]

if SKLEARN_AVAILABLE:
    learners_classif.extend([
        # Regresión Logística
        LearnerLogisticRegression(id='logreg', max_iter=1000),
        LearnerLogisticRegression(id='logreg_l2_weak', C=10.0, max_iter=1000),
        LearnerLogisticRegression(id='logreg_l2_strong', C=0.01, max_iter=1000),
        
        # Árboles de Decisión
        LearnerDecisionTree(id='tree_shallow', max_depth=3),
        LearnerDecisionTree(id='tree_medium', max_depth=10),
        LearnerDecisionTree(id='tree_deep', max_depth=None),
        
        # Random Forest
        LearnerRandomForest(id='rf_small', n_estimators=10, max_depth=5),
        LearnerRandomForest(id='rf_medium', n_estimators=50, max_depth=10),
        LearnerRandomForest(id='rf_large', n_estimators=100),
        
        # Gradient Boosting
        LearnerGradientBoosting(id='gb_weak', n_estimators=50, learning_rate=0.1, max_depth=3),
        LearnerGradientBoosting(id='gb_strong', n_estimators=100, learning_rate=0.05, max_depth=5),
        
        # SVM
        LearnerSVM(id='svm_linear', kernel='linear', max_iter=2000),
        LearnerSVM(id='svm_rbf', kernel='rbf', gamma='scale', max_iter=2000),
        
        # K-Vecinos
        LearnerKNN(id='knn_3', n_neighbors=3),
        LearnerKNN(id='knn_10', n_neighbors=10),
        
        # Otros
        LearnerNaiveBayes(id='naive_bayes'),
        LearnerMLPClassifier(id='mlp_small', hidden_layer_sizes=(50,), max_iter=500),
        LearnerMLPClassifier(id='mlp_medium', hidden_layer_sizes=(100, 50), max_iter=500),
    ])

print(f"- Modelos de clasificación: {len(learners_classif)}")

# REGRESIÓN
learners_regr = [
    # Baseline
    LearnerRegrFeatureless(id='baseline_mean', method='mean'),
    LearnerRegrFeatureless(id='baseline_median', method='median'),
]

if SKLEARN_AVAILABLE:
    learners_regr.extend([
        # Modelos Lineales
        LearnerLinearRegression(id='linear'),
        LearnerRidge(id='ridge_weak', alpha=0.01),
        LearnerRidge(id='ridge', alpha=1.0),
        LearnerRidge(id='ridge_strong', alpha=10.0),
        LearnerLasso(id='lasso_weak', alpha=0.001),
        LearnerLasso(id='lasso', alpha=0.01),
        LearnerElasticNet(id='elastic', alpha=0.1, l1_ratio=0.5),
        
        # Árboles
        LearnerDecisionTreeRegressor(id='tree_reg', max_depth=10),
        LearnerRandomForestRegressor(id='rf_reg_small', n_estimators=50),
        LearnerRandomForestRegressor(id='rf_reg', n_estimators=100),
        LearnerGradientBoostingRegressor(id='gb_reg', n_estimators=100, learning_rate=0.1),
        
        # SVM
        LearnerSVR(id='svr_linear', kernel='linear'),
        LearnerSVR(id='svr_rbf', kernel='rbf', gamma='scale'),
        
        # Otros
        LearnerKNNRegressor(id='knn_reg', n_neighbors=5),
        LearnerMLPRegressor(id='mlp_reg', hidden_layer_sizes=(50,), max_iter=500),
    ])

print(f"- Modelos de regresión: {len(learners_regr)}")

# ============================================================================
# 4. EJECUTAR BENCHMARKS
# ============================================================================
print("\n4. EJECUTANDO BENCHMARKS")
print("-" * 40)

# CLASIFICACIÓN
print("\n>>> Evaluando modelos de CLASIFICACIÓN...")

# Separar modelos por velocidad
classif_rapidos = [l for l in learners_classif if 'svm' not in l.id and 'mlp' not in l.id]
classif_lentos = [l for l in learners_classif if 'svm' in l.id or 'mlp' in l.id]

bench_results_classif = []

if classif_rapidos:
    print(f"- Evaluando {len(classif_rapidos)} modelos rápidos (5-fold CV)...")
    bench_c_rapido = benchmark(
        tasks=[task_c1, task_c2],
        learners=classif_rapidos,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )
    bench_results_classif.append(bench_c_rapido)

if classif_lentos and SKLEARN_AVAILABLE:
    print(f"- Evaluando {len(classif_lentos)} modelos lentos (3-fold CV)...")
    bench_c_lento = benchmark(
        tasks=[task_c1, task_c2],
        learners=classif_lentos,
        resampling=ResamplingCV(folds=3, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )
    bench_results_classif.append(bench_c_lento)

# REGRESIÓN
print("\n>>> Evaluando modelos de REGRESIÓN...")

regr_rapidos = [l for l in learners_regr if 'svr' not in l.id and 'mlp' not in l.id]
regr_lentos = [l for l in learners_regr if 'svr' in l.id or 'mlp' in l.id]

bench_results_regr = []

if regr_rapidos:
    print(f"- Evaluando {len(regr_rapidos)} modelos rápidos (5-fold CV)...")
    bench_r_rapido = benchmark(
        tasks=[task_r1, task_r2],
        learners=regr_rapidos,
        resampling=ResamplingCV(folds=5),
        measures=[MeasureRegrMSE(), MeasureRegrR2()]
    )
    bench_results_regr.append(bench_r_rapido)

if regr_lentos and SKLEARN_AVAILABLE:
    print(f"- Evaluando {len(regr_lentos)} modelos lentos (3-fold CV)...")
    bench_r_lento = benchmark(
        tasks=[task_r1, task_r2],
        learners=regr_lentos,
        resampling=ResamplingCV(folds=3),
        measures=[MeasureRegrMSE(), MeasureRegrR2()]
    )
    bench_results_regr.append(bench_r_lento)

# ============================================================================
# 5. RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("RESULTADOS DEL BENCHMARK")
print("="*80)

# CLASIFICACIÓN
print("\n>>> CLASIFICACIÓN - Ranking por Accuracy")
print("-" * 70)

# Combinar resultados de clasificación
all_classif_results = []
for bench in bench_results_classif:
    rankings = bench.rank_learners('classif.acc')
    for _, row in rankings.iterrows():
        all_classif_results.append({
            'learner': row['learner'],
            'mean_score': row['mean_score'],
            'rank': row['rank']
        })

# Ordenar por score
all_classif_results.sort(key=lambda x: x['mean_score'], reverse=True)

# Mostrar tabla combinada
print(f"\n{'Rank':<6} {'Modelo':<30} {'Accuracy':<12} {'Tipo':<15}")
print("-" * 65)
for i, result in enumerate(all_classif_results[:20], 1):
    learner = result['learner']
    score = result['mean_score']
    
    # Determinar tipo
    if 'baseline' in learner:
        tipo = 'Baseline'
    elif 'logreg' in learner:
        tipo = 'Logística'
    elif 'tree' in learner and 'rf' not in learner:
        tipo = 'Árbol'
    elif 'rf' in learner:
        tipo = 'Random Forest'
    elif 'gb' in learner:
        tipo = 'Gradient Boost'
    elif 'svm' in learner:
        tipo = 'SVM'
    elif 'knn' in learner:
        tipo = 'K-Vecinos'
    elif 'naive' in learner:
        tipo = 'Naive Bayes'
    elif 'mlp' in learner:
        tipo = 'Red Neuronal'
    else:
        tipo = 'Otro'
    
    print(f"{i:<6} {learner:<30} {score:.4f}      {tipo:<15}")

if len(all_classif_results) > 20:
    print(f"\n... y {len(all_classif_results) - 20} modelos más")

# REGRESIÓN
print("\n\n>>> REGRESIÓN - Ranking por R²")
print("-" * 70)

# Combinar resultados de regresión
all_regr_results = []
for bench in bench_results_regr:
    rankings = bench.rank_learners('regr.r2')
    for _, row in rankings.iterrows():
        all_regr_results.append({
            'learner': row['learner'],
            'mean_score': row['mean_score'],
            'rank': row['rank']
        })

# Ordenar por R²
all_regr_results.sort(key=lambda x: x['mean_score'], reverse=True)

# Mostrar tabla
print(f"\n{'Rank':<6} {'Modelo':<30} {'R²':<12} {'Tipo':<15}")
print("-" * 65)
for i, result in enumerate(all_regr_results[:20], 1):
    learner = result['learner']
    score = result['mean_score']
    
    # Determinar tipo
    if 'baseline' in learner:
        tipo = 'Baseline'
    elif 'linear' in learner:
        tipo = 'Lineal'
    elif 'ridge' in learner:
        tipo = 'Ridge'
    elif 'lasso' in learner:
        tipo = 'Lasso'
    elif 'elastic' in learner:
        tipo = 'ElasticNet'
    elif 'tree' in learner and 'rf' not in learner:
        tipo = 'Árbol'
    elif 'rf' in learner:
        tipo = 'Random Forest'
    elif 'gb' in learner:
        tipo = 'Gradient Boost'
    elif 'svr' in learner:
        tipo = 'SVR'
    elif 'knn' in learner:
        tipo = 'K-Vecinos'
    elif 'mlp' in learner:
        tipo = 'Red Neuronal'
    else:
        tipo = 'Otro'
    
    print(f"{i:<6} {learner:<30} {score:.4f}      {tipo:<15}")

if len(all_regr_results) > 20:
    print(f"\n... y {len(all_regr_results) - 20} modelos más")

# ============================================================================
# 6. ANÁLISIS POR DATASET
# ============================================================================
print("\n\n>>> ANÁLISIS POR DATASET")
print("-" * 70)

# Para cada dataset de clasificación
for task_id in ['classif_simple', 'classif_complex']:
    print(f"\n{task_id.upper()}:")
    
    # Obtener mejores modelos para este dataset
    task_results = []
    for bench in bench_results_classif:
        # Obtener resultados agregados
        agg = bench.aggregate('classif.acc')
        if task_id in agg.index:
            scores = agg.loc[task_id]
            for learner, score in scores.items():
                if not pd.isna(score):
                    task_results.append((learner, score))
    
    # Ordenar y mostrar top 5
    task_results.sort(key=lambda x: x[1], reverse=True)
    for i, (learner, score) in enumerate(task_results[:5], 1):
        print(f"  {i}. {learner:<25} Accuracy={score:.4f}")

# Para cada dataset de regresión
for task_id in ['regr_simple', 'regr_complex']:
    print(f"\n{task_id.upper()}:")
    
    # Obtener mejores modelos
    task_results = []
    for bench in bench_results_regr:
        agg = bench.aggregate('regr.r2')
        if task_id in agg.index:
            scores = agg.loc[task_id]
            for learner, score in scores.items():
                if not pd.isna(score):
                    task_results.append((learner, score))
    
    # Ordenar y mostrar top 5
    task_results.sort(key=lambda x: x[1], reverse=True)
    for i, (learner, score) in enumerate(task_results[:5], 1):
        print(f"  {i}. {learner:<25} R²={score:.4f}")

# ============================================================================
# 7. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

# Estadísticas generales
total_models = len(learners_classif) + len(learners_regr)
total_tasks = 4
total_experiments = len(learners_classif) * 2 + len(learners_regr) * 2

print(f"\nESTADÍSTICAS:")
print(f"- Total de modelos evaluados: {total_models}")
print(f"  - Clasificación: {len(learners_classif)}")
print(f"  - Regresión: {len(learners_regr)}")
print(f"- Datasets evaluados: {total_tasks}")
print(f"- Total de experimentos: {total_experiments}")

# Mejor modelo global
if all_classif_results:
    print(f"\nMEJOR MODELO DE CLASIFICACIÓN:")
    best_c = all_classif_results[0]
    print(f"- {best_c['learner']} (Accuracy promedio: {best_c['mean_score']:.4f})")

if all_regr_results:
    print(f"\nMEJOR MODELO DE REGRESIÓN:")
    best_r = all_regr_results[0]
    print(f"- {best_r['learner']} (R² promedio: {best_r['mean_score']:.4f})")

print(f"\nFinalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n+ Benchmark completado exitosamente!")