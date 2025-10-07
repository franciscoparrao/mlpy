"""
Benchmark Simplificado de MLPY
==============================

Comparación exhaustiva de modelos con manejo correcto de métricas.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BENCHMARK DE MODELOS MLPY")
print("="*80)
print(f"Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Importar MLPY
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import MeasureClassifAccuracy, MeasureRegrMSE, MeasureRegrR2
from mlpy.resamplings import ResamplingCV
from mlpy import benchmark

# Importar learners
from mlpy.learners import LearnerClassifFeatureless, LearnerRegrFeatureless

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

# ============================================================================
# 1. DATASETS
# ============================================================================
print("\n1. CREANDO DATASETS")
print("-" * 40)

np.random.seed(42)

# Dataset 1: Clasificación binaria (linealmente separable)
n1 = 1000
X1_1 = np.random.normal(0, 1, n1)
X1_2 = np.random.normal(0, 1, n1)
X1_3 = np.random.uniform(-1, 1, n1)
y1 = (2*X1_1 + X1_2 - 0.5*X1_3 + np.random.normal(0, 0.3, n1)) > 0
data_classif_simple = pd.DataFrame({
    'x1': X1_1, 'x2': X1_2, 'x3': X1_3,
    'target': ['Positivo' if yi else 'Negativo' for yi in y1]
})

# Dataset 2: Clasificación binaria (no lineal)
n2 = 1200
X2_1 = np.random.uniform(-3, 3, n2)
X2_2 = np.random.uniform(-3, 3, n2)
X2_3 = np.random.normal(0, 1, n2)
X2_4 = np.random.exponential(1, n2)
y2 = (X2_1**2 + X2_2**2 - 9 + 0.5*X2_3 + 0.3*X2_4 + np.random.normal(0, 1, n2)) > 0
data_classif_compleja = pd.DataFrame({
    'x1': X2_1, 'x2': X2_2, 'x3': X2_3, 'x4': X2_4,
    'target': ['Dentro' if yi else 'Fuera' for yi in y2]
})

# Dataset 3: Regresión simple
n3 = 800
X3_1 = np.random.uniform(-2, 2, n3)
X3_2 = np.random.normal(0, 1, n3)
X3_3 = np.random.exponential(1, n3)
y3 = 3*X3_1 + X3_2**2 - 0.5*X3_3 + np.random.normal(0, 0.5, n3)
data_regr_simple = pd.DataFrame({
    'x1': X3_1, 'x2': X3_2, 'x3': X3_3,
    'target': y3
})

# Dataset 4: Regresión no lineal
n4 = 1000
X4_1 = np.random.uniform(-3, 3, n4)
X4_2 = np.random.uniform(-3, 3, n4)
X4_3 = np.random.normal(0, 1, n4)
y4 = np.sin(X4_1) + X4_2**2 + 0.1*X4_1*X4_2*X4_3 + np.random.normal(0, 0.3, n4)
data_regr_compleja = pd.DataFrame({
    'x1': X4_1, 'x2': X4_2, 'x3': X4_3,
    'target': y4
})

datasets = {
    'classif_simple': data_classif_simple,
    'classif_compleja': data_classif_compleja,
    'regr_simple': data_regr_simple,
    'regr_compleja': data_regr_compleja
}

for nombre, df in datasets.items():
    print(f"\n{nombre}:")
    print(f"  - Muestras: {len(df)}")
    print(f"  - Características: {df.shape[1]-1}")
    if 'classif' in nombre:
        print(f"  - Distribución: {dict(df['target'].value_counts())}")

# ============================================================================
# 2. TAREAS
# ============================================================================
print("\n2. CREANDO TAREAS MLPY")
print("-" * 40)

tareas_classif = []
tareas_regr = []

# Tareas de clasificación
for nombre in ['classif_simple', 'classif_compleja']:
    task = TaskClassif(
        data=datasets[nombre],
        target='target',
        id=nombre
    )
    tareas_classif.append(task)
    print(f"+ Tarea: {task.id} ({task.task_type})")

# Tareas de regresión
for nombre in ['regr_simple', 'regr_compleja']:
    task = TaskRegr(
        data=datasets[nombre],
        target='target',
        id=nombre
    )
    tareas_regr.append(task)
    print(f"+ Tarea: {task.id} ({task.task_type})")

# ============================================================================
# 3. CONFIGURAR MODELOS
# ============================================================================
print("\n3. CONFIGURANDO MODELOS")
print("-" * 40)

# Modelos de clasificación
modelos_classif = [
    # Baseline
    LearnerClassifFeatureless(id='baseline', method='mode'),
]

if SKLEARN_AVAILABLE:
    modelos_classif.extend([
        # Lineales
        LearnerLogisticRegression(id='logreg', max_iter=1000),
        LearnerLogisticRegression(id='logreg_l2_weak', C=10.0, max_iter=1000),
        LearnerLogisticRegression(id='logreg_l2_strong', C=0.01, max_iter=1000),
        
        # Árboles
        LearnerDecisionTree(id='tree_shallow', max_depth=3),
        LearnerDecisionTree(id='tree_medium', max_depth=10),
        LearnerDecisionTree(id='tree_deep', max_depth=None),
        
        # Ensemble
        LearnerRandomForest(id='rf_small', n_estimators=10, max_depth=5),
        LearnerRandomForest(id='rf_medium', n_estimators=50, max_depth=10),
        LearnerRandomForest(id='rf_large', n_estimators=100, max_depth=None),
        
        LearnerGradientBoosting(id='gb_weak', n_estimators=50, learning_rate=0.1, max_depth=3),
        LearnerGradientBoosting(id='gb_strong', n_estimators=100, learning_rate=0.05, max_depth=5),
        
        # Otros
        LearnerSVM(id='svm_linear', kernel='linear'),
        LearnerSVM(id='svm_rbf', kernel='rbf', gamma='scale'),
        LearnerKNN(id='knn_3', n_neighbors=3),
        LearnerKNN(id='knn_10', n_neighbors=10),
        LearnerNaiveBayes(id='naive_bayes'),
        LearnerMLPClassifier(id='mlp', hidden_layer_sizes=(100,), max_iter=500),
    ])

print(f"Modelos de clasificación: {len(modelos_classif)}")

# Modelos de regresión
modelos_regr = [
    # Baseline
    LearnerRegrFeatureless(id='baseline_mean', method='mean'),
    LearnerRegrFeatureless(id='baseline_median', method='median'),
]

if SKLEARN_AVAILABLE:
    modelos_regr.extend([
        # Lineales
        LearnerLinearRegression(id='linear'),
        LearnerRidge(id='ridge_weak', alpha=0.01),
        LearnerRidge(id='ridge', alpha=1.0),
        LearnerRidge(id='ridge_strong', alpha=10.0),
        LearnerLasso(id='lasso_weak', alpha=0.001),
        LearnerLasso(id='lasso', alpha=0.01),
        LearnerElasticNet(id='elastic', alpha=0.1, l1_ratio=0.5),
        
        # Árboles
        LearnerDecisionTreeRegressor(id='tree_reg', max_depth=10),
        LearnerRandomForestRegressor(id='rf_reg_small', n_estimators=10),
        LearnerRandomForestRegressor(id='rf_reg', n_estimators=50),
        LearnerGradientBoostingRegressor(id='gb_reg', n_estimators=100, learning_rate=0.1),
        
        # Otros
        LearnerSVR(id='svr_linear', kernel='linear'),
        LearnerSVR(id='svr_rbf', kernel='rbf'),
        LearnerKNNRegressor(id='knn_reg', n_neighbors=5),
        LearnerMLPRegressor(id='mlp_reg', hidden_layer_sizes=(50,), max_iter=500),
    ])

print(f"Modelos de regresión: {len(modelos_regr)}")

# ============================================================================
# 4. EJECUTAR BENCHMARKS
# ============================================================================
print("\n4. EJECUTANDO BENCHMARKS")
print("-" * 40)

# CLASIFICACIÓN
print("\n>>> Benchmark de Clasificación")
print("Evaluando con Accuracy (funciona para binario y multiclase)...")

# Separar modelos rápidos y lentos
modelos_classif_rapidos = [m for m in modelos_classif if 'svm' not in m.id and 'mlp' not in m.id]
modelos_classif_lentos = [m for m in modelos_classif if 'svm' in m.id or 'mlp' in m.id]

# Benchmark rápido
if modelos_classif_rapidos:
    bench_classif_rapido = benchmark(
        tasks=tareas_classif,
        learners=modelos_classif_rapidos,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )
    print("+ Modelos rápidos completados")

# Benchmark lento
if modelos_classif_lentos and SKLEARN_AVAILABLE:
    bench_classif_lento = benchmark(
        tasks=tareas_classif,
        learners=modelos_classif_lentos,
        resampling=ResamplingCV(folds=3, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )
    print("+ Modelos lentos completados")

# REGRESIÓN
print("\n>>> Benchmark de Regresión")
print("Evaluando con MSE y R²...")

# Separar modelos
modelos_regr_rapidos = [m for m in modelos_regr if 'svr' not in m.id and 'mlp' not in m.id]
modelos_regr_lentos = [m for m in modelos_regr if 'svr' in m.id or 'mlp' in m.id]

# Benchmark rápido
if modelos_regr_rapidos:
    bench_regr_rapido = benchmark(
        tasks=tareas_regr,
        learners=modelos_regr_rapidos,
        resampling=ResamplingCV(folds=5),
        measures=[MeasureRegrMSE(), MeasureRegrR2()]
    )
    print("+ Modelos rápidos completados")

# Benchmark lento
if modelos_regr_lentos and SKLEARN_AVAILABLE:
    bench_regr_lento = benchmark(
        tasks=tareas_regr,
        learners=modelos_regr_lentos,
        resampling=ResamplingCV(folds=3),
        measures=[MeasureRegrMSE(), MeasureRegrR2()]
    )
    print("+ Modelos lentos completados")

# ============================================================================
# 5. RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("RESULTADOS DEL BENCHMARK")
print("="*80)

# CLASIFICACIÓN
print("\n>>> CLASIFICACIÓN (Ordenado por Accuracy)")
print("-" * 80)

for tarea in tareas_classif:
    print(f"\nDataset: {tarea.id}")
    print("~" * 50)
    
    # Recopilar todos los resultados
    todos_resultados = []
    
    if 'bench_classif_rapido' in locals():
        try:
            rankings = bench_classif_rapido.rank_learners('classif.acc', task_id=tarea.id)
            for _, row in rankings.iterrows():
                todos_resultados.append((row['learner'], row['mean_score'], row['std_score']))
        except:
            pass
    
    if 'bench_classif_lento' in locals():
        try:
            rankings = bench_classif_lento.rank_learners('classif.acc', task_id=tarea.id)
            for _, row in rankings.iterrows():
                # Verificar que no esté duplicado
                if not any(r[0] == row['learner'] for r in todos_resultados):
                    todos_resultados.append((row['learner'], row['mean_score'], row['std_score']))
        except:
            pass
    
    # Ordenar por score
    todos_resultados.sort(key=lambda x: x[1], reverse=True)
    
    # Mostrar tabla
    print(f"\n{'Rank':<6} {'Modelo':<25} {'Accuracy':<12} {'Std':<10}")
    print("-" * 55)
    for i, (modelo, mean, std) in enumerate(todos_resultados[:15], 1):
        print(f"{i:<6} {modelo:<25} {mean:.4f}      {std:.4f}")
    
    if len(todos_resultados) > 15:
        print(f"\n... y {len(todos_resultados) - 15} modelos más")

# REGRESIÓN
print("\n\n>>> REGRESIÓN (Ordenado por R²)")
print("-" * 80)

for tarea in tareas_regr:
    print(f"\nDataset: {tarea.id}")
    print("~" * 50)
    
    # Recopilar resultados de R²
    todos_resultados = []
    
    if 'bench_regr_rapido' in locals():
        try:
            rankings = bench_regr_rapido.rank_learners('regr.r2', task_id=tarea.id)
            for _, row in rankings.iterrows():
                todos_resultados.append((row['learner'], row['mean_score'], row['std_score']))
        except:
            pass
    
    if 'bench_regr_lento' in locals():
        try:
            rankings = bench_regr_lento.rank_learners('regr.r2', task_id=tarea.id)
            for _, row in rankings.iterrows():
                if not any(r[0] == row['learner'] for r in todos_resultados):
                    todos_resultados.append((row['learner'], row['mean_score'], row['std_score']))
        except:
            pass
    
    # Ordenar por R²
    todos_resultados.sort(key=lambda x: x[1], reverse=True)
    
    # Mostrar tabla
    print(f"\n{'Rank':<6} {'Modelo':<25} {'R²':<12} {'Std':<10}")
    print("-" * 55)
    for i, (modelo, mean, std) in enumerate(todos_resultados[:15], 1):
        print(f"{i:<6} {modelo:<25} {mean:.4f}      {std:.4f}")
    
    if len(todos_resultados) > 15:
        print(f"\n... y {len(todos_resultados) - 15} modelos más")

# ============================================================================
# 6. RESUMEN
# ============================================================================
print("\n" + "="*80)
print("RESUMEN GENERAL")
print("="*80)

# Mejor modelo por dataset
print("\nMEJOR MODELO POR DATASET:")
print("-" * 70)
print(f"{'Dataset':<20} {'Mejor Modelo':<25} {'Métrica':<15} {'Score':<10}")
print("-" * 70)

# Clasificación
for tarea in tareas_classif:
    mejor_score = 0
    mejor_modelo = 'baseline'
    
    if 'bench_classif_rapido' in locals():
        try:
            rankings = bench_classif_rapido.rank_learners('classif.acc', task_id=tarea.id)
            if len(rankings) > 0:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if 'bench_classif_lento' in locals():
        try:
            rankings = bench_classif_lento.rank_learners('classif.acc', task_id=tarea.id)
            if len(rankings) > 0 and rankings.iloc[0]['mean_score'] > mejor_score:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    print(f"{tarea.id:<20} {mejor_modelo:<25} {'Accuracy':<15} {mejor_score:.4f}")

# Regresión
for tarea in tareas_regr:
    mejor_score = -np.inf
    mejor_modelo = 'baseline_mean'
    
    if 'bench_regr_rapido' in locals():
        try:
            rankings = bench_regr_rapido.rank_learners('regr.r2', task_id=tarea.id)
            if len(rankings) > 0:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if 'bench_regr_lento' in locals():
        try:
            rankings = bench_regr_lento.rank_learners('regr.r2', task_id=tarea.id)
            if len(rankings) > 0 and rankings.iloc[0]['mean_score'] > mejor_score:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    print(f"{tarea.id:<20} {mejor_modelo:<25} {'R²':<15} {mejor_score:.4f}")

# Estadísticas finales
total_experimentos = (len(tareas_classif) * len(modelos_classif) + 
                     len(tareas_regr) * len(modelos_regr))

print(f"\n\nESTADÍSTICAS:")
print(f"- Datasets evaluados: {len(datasets)}")
print(f"- Modelos de clasificación: {len(modelos_classif)}")
print(f"- Modelos de regresión: {len(modelos_regr)}")
print(f"- Total de experimentos: {total_experimentos}")

print(f"\nFinalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n+ Benchmark completado exitosamente!")