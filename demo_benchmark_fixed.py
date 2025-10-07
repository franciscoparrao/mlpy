"""
Benchmark Fijo de MLPY
=====================

Versión corregida del benchmark que maneja correctamente las métricas
para clasificación binaria y multiclase.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BENCHMARK DE MLPY - VERSIÓN CORREGIDA")
print("="*80)
print(f"Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Importar MLPY
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.measures import (
    # Clasificación - solo accuracy funciona para ambos
    MeasureClassifAccuracy,
    # Regresión
    MeasureRegrMSE, MeasureRegrMAE, MeasureRegrR2
)
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
# 1. DATASETS SINTÉTICOS
# ============================================================================
print("\n1. CREANDO DATASETS SINTÉTICOS")
print("-" * 40)

np.random.seed(42)

# Dataset 1: Clasificación binaria (simple)
n1 = 800
X1 = np.random.normal(0, 1, (n1, 3))
coef = np.array([1.5, -1.0, 0.5])
y1 = (X1 @ coef + np.random.normal(0, 0.5, n1)) > 0
data_binaria = pd.DataFrame(X1, columns=['x1', 'x2', 'x3'])
data_binaria['target'] = ['Positivo' if yi else 'Negativo' for yi in y1]

# Dataset 2: Clasificación binaria (no lineal)
n2 = 1000
X2 = np.random.uniform(-2, 2, (n2, 4))
y2 = ((X2[:, 0]**2 + X2[:, 1]**2 < 2) & (X2[:, 2] > 0)) | (X2[:, 3] > 1)
data_no_lineal = pd.DataFrame(X2, columns=['x1', 'x2', 'x3', 'x4'])
data_no_lineal['target'] = ['ClaseA' if yi else 'ClaseB' for yi in y2]

# Dataset 3: Regresión simple
n3 = 600
X3 = np.random.uniform(-3, 3, (n3, 3))
y3 = 2*X3[:, 0] - X3[:, 1]**2 + 0.5*X3[:, 2] + np.random.normal(0, 0.5, n3)
data_regr_simple = pd.DataFrame(X3, columns=['x1', 'x2', 'x3'])
data_regr_simple['target'] = y3

# Dataset 4: Regresión compleja
n4 = 800
X4 = np.random.normal(0, 1, (n4, 5))
y4 = (np.sin(2*X4[:, 0]) + X4[:, 1]**2 + 
      np.exp(-np.abs(X4[:, 2])) + 0.5*X4[:, 3]*X4[:, 4] + 
      np.random.normal(0, 0.3, n4))
data_regr_compleja = pd.DataFrame(X4, columns=[f'x{i+1}' for i in range(5)])
data_regr_compleja['target'] = y4

datasets = {
    'binaria': data_binaria,
    'no_lineal': data_no_lineal,
    'regr_simple': data_regr_simple,
    'regr_compleja': data_regr_compleja
}

for nombre, df in datasets.items():
    print(f"\n{nombre}:")
    print(f"  - Muestras: {len(df)}")
    print(f"  - Características: {df.shape[1]-1}")
    if 'regr' not in nombre:
        dist = dict(df['target'].value_counts())
        print(f"  - Clases: {len(dist)}")
        print(f"  - Balance: {dist}")

# ============================================================================
# 2. CREAR TAREAS
# ============================================================================
print("\n2. CREANDO TAREAS MLPY")
print("-" * 40)

tareas_classif = []
tareas_regr = []

# Tareas de clasificación binaria
for nombre in ['binaria', 'no_lineal']:
    task = TaskClassif(
        data=datasets[nombre],
        target='target',
        id=f'task_{nombre}'
    )
    tareas_classif.append(task)
    print(f"+ Tarea: {task.id} (clasificación)")

# Tareas de regresión
for nombre in ['regr_simple', 'regr_compleja']:
    task = TaskRegr(
        data=datasets[nombre],
        target='target',
        id=f'task_{nombre}'
    )
    tareas_regr.append(task)
    print(f"+ Tarea: {task.id} (regresión)")

# ============================================================================
# 3. CONFIGURAR LEARNERS
# ============================================================================
print("\n3. CONFIGURANDO LEARNERS")
print("-" * 40)

# Learners de clasificación
learners_classif = [
    # Baseline
    LearnerClassifFeatureless(id='baseline_mode', method='mode'),
    LearnerClassifFeatureless(id='baseline_weighted', method='weighted'),
]

if SKLEARN_AVAILABLE:
    learners_classif.extend([
        # Lineales
        LearnerLogisticRegression(id='logreg', max_iter=1000),
        LearnerLogisticRegression(id='logreg_l2', C=0.1, max_iter=1000),
        
        # Árboles
        LearnerDecisionTree(id='tree_shallow', max_depth=5),
        LearnerDecisionTree(id='tree_medium', max_depth=10),
        LearnerDecisionTree(id='tree_deep', max_depth=20),
        
        # Ensemble
        LearnerRandomForest(id='rf_small', n_estimators=50, max_depth=10),
        LearnerRandomForest(id='rf_large', n_estimators=100),
        LearnerGradientBoosting(id='gb', n_estimators=100, learning_rate=0.1, max_depth=5),
        
        # SVM (menos configuraciones para acelerar)
        LearnerSVM(id='svm_linear', kernel='linear', max_iter=1000),
        LearnerSVM(id='svm_rbf', kernel='rbf', gamma='scale', max_iter=1000),
        
        # Vecinos
        LearnerKNN(id='knn_5', n_neighbors=5),
        LearnerKNN(id='knn_10', n_neighbors=10),
        
        # Otros
        LearnerNaiveBayes(id='naive_bayes'),
        LearnerMLPClassifier(id='mlp', hidden_layer_sizes=(100,), max_iter=500),
    ])

print(f"Learners de clasificación: {len(learners_classif)}")

# Learners de regresión
learners_regr = [
    # Baseline
    LearnerRegrFeatureless(id='baseline_mean', method='mean'),
    LearnerRegrFeatureless(id='baseline_median', method='median'),
]

if SKLEARN_AVAILABLE:
    learners_regr.extend([
        # Lineales
        LearnerLinearRegression(id='linear'),
        LearnerRidge(id='ridge', alpha=1.0),
        LearnerRidge(id='ridge_strong', alpha=10.0),
        LearnerLasso(id='lasso', alpha=0.01),
        LearnerElasticNet(id='elastic', alpha=0.1, l1_ratio=0.5),
        
        # Árboles
        LearnerDecisionTreeRegressor(id='tree_reg', max_depth=10),
        LearnerRandomForestRegressor(id='rf_reg', n_estimators=100),
        LearnerGradientBoostingRegressor(id='gb_reg', n_estimators=100, learning_rate=0.1),
        
        # SVM
        LearnerSVR(id='svr_linear', kernel='linear'),
        LearnerSVR(id='svr_rbf', kernel='rbf', gamma='scale'),
        
        # Otros
        LearnerKNNRegressor(id='knn_reg', n_neighbors=5),
        LearnerMLPRegressor(id='mlp_reg', hidden_layer_sizes=(50,), max_iter=500),
    ])

print(f"Learners de regresión: {len(learners_regr)}")

# ============================================================================
# 4. EJECUTAR BENCHMARKS
# ============================================================================
print("\n4. EJECUTANDO BENCHMARKS")
print("-" * 40)

# CLASIFICACIÓN
print("\n>>> BENCHMARK DE CLASIFICACIÓN")
print("Evaluando con Accuracy (funciona para todos los casos)...")

# Separar modelos por velocidad
learners_rapidos = [l for l in learners_classif if 'svm' not in l.id and 'mlp' not in l.id]
learners_lentos = [l for l in learners_classif if 'svm' in l.id or 'mlp' in l.id]

resultados_classif = []

if learners_rapidos:
    print(f"\nEvaluando {len(learners_rapidos)} modelos rápidos (5-fold CV)...")
    bench_rapido = benchmark(
        tasks=tareas_classif,
        learners=learners_rapidos,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )
    resultados_classif.append(bench_rapido)
    print("+ Completado")

if learners_lentos and SKLEARN_AVAILABLE:
    print(f"\nEvaluando {len(learners_lentos)} modelos lentos (3-fold CV)...")
    bench_lento = benchmark(
        tasks=tareas_classif,
        learners=learners_lentos,
        resampling=ResamplingCV(folds=3, stratify=True),
        measures=[MeasureClassifAccuracy()]
    )
    resultados_classif.append(bench_lento)
    print("+ Completado")

# REGRESIÓN
print("\n>>> BENCHMARK DE REGRESIÓN")
print("Evaluando con MSE, MAE y R²...")

learners_regr_rapidos = [l for l in learners_regr if 'svr' not in l.id and 'mlp' not in l.id]
learners_regr_lentos = [l for l in learners_regr if 'svr' in l.id or 'mlp' in l.id]

resultados_regr = []

if learners_regr_rapidos:
    print(f"\nEvaluando {len(learners_regr_rapidos)} modelos rápidos (5-fold CV)...")
    bench_regr_rapido = benchmark(
        tasks=tareas_regr,
        learners=learners_regr_rapidos,
        resampling=ResamplingCV(folds=5),
        measures=[MeasureRegrMSE(), MeasureRegrMAE(), MeasureRegrR2()]
    )
    resultados_regr.append(bench_regr_rapido)
    print("+ Completado")

if learners_regr_lentos and SKLEARN_AVAILABLE:
    print(f"\nEvaluando {len(learners_regr_lentos)} modelos lentos (3-fold CV)...")
    bench_regr_lento = benchmark(
        tasks=tareas_regr,
        learners=learners_regr_lentos,
        resampling=ResamplingCV(folds=3),
        measures=[MeasureRegrMSE(), MeasureRegrMAE(), MeasureRegrR2()]
    )
    resultados_regr.append(bench_regr_lento)
    print("+ Completado")

# ============================================================================
# 5. RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("RESULTADOS DEL BENCHMARK")
print("="*80)

# CLASIFICACIÓN
print("\n>>> RESULTADOS DE CLASIFICACIÓN")
print("-" * 70)

for tarea in tareas_classif:
    print(f"\nDataset: {tarea.id}")
    print("~" * 40)
    
    # Combinar todos los resultados
    todos = []
    for bench in resultados_classif:
        try:
            rankings = bench.rank_learners('classif.acc', task_id=tarea.id)
            for _, row in rankings.iterrows():
                todos.append({
                    'learner': row['learner'],
                    'mean': row['mean_score'],
                    'std': row.get('std_score', 0)
                })
        except:
            pass
    
    # Eliminar duplicados y ordenar
    vistos = set()
    unicos = []
    for item in todos:
        if item['learner'] not in vistos:
            vistos.add(item['learner'])
            unicos.append(item)
    
    unicos.sort(key=lambda x: x['mean'], reverse=True)
    
    # Mostrar top 10
    print(f"\n{'Rank':<6} {'Modelo':<25} {'Accuracy':<12} {'Std':<10}")
    print("-" * 55)
    for i, item in enumerate(unicos[:10], 1):
        print(f"{i:<6} {item['learner']:<25} {item['mean']:.4f}      ± {item['std']:.4f}")
    
    if len(unicos) > 10:
        print(f"\n... y {len(unicos)-10} modelos más")

# REGRESIÓN
print("\n\n>>> RESULTADOS DE REGRESIÓN")
print("-" * 70)

for tarea in tareas_regr:
    print(f"\nDataset: {tarea.id}")
    print("~" * 40)
    
    # Combinar resultados por R²
    todos = []
    for bench in resultados_regr:
        try:
            rankings = bench.rank_learners('regr.r2', task_id=tarea.id)
            for _, row in rankings.iterrows():
                todos.append({
                    'learner': row['learner'],
                    'mean': row['mean_score'],
                    'std': row.get('std_score', 0)
                })
        except:
            pass
    
    # Eliminar duplicados
    vistos = set()
    unicos = []
    for item in todos:
        if item['learner'] not in vistos:
            vistos.add(item['learner'])
            unicos.append(item)
    
    unicos.sort(key=lambda x: x['mean'], reverse=True)
    
    # Mostrar top 10
    print(f"\n{'Rank':<6} {'Modelo':<25} {'R²':<12} {'Std':<10}")
    print("-" * 55)
    for i, item in enumerate(unicos[:10], 1):
        print(f"{i:<6} {item['learner']:<25} {item['mean']:.4f}      ± {item['std']:.4f}")
    
    if len(unicos) > 10:
        print(f"\n... y {len(unicos)-10} modelos más")

# ============================================================================
# 6. RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

# Mejor modelo por dataset
print("\nMEJOR MODELO POR DATASET:")
print("-" * 60)
print(f"{'Dataset':<25} {'Mejor Modelo':<25} {'Score':<15}")
print("-" * 60)

# Clasificación
for tarea in tareas_classif:
    mejor = None
    mejor_score = -np.inf
    
    for bench in resultados_classif:
        try:
            rankings = bench.rank_learners('classif.acc', task_id=tarea.id)
            if len(rankings) > 0 and rankings.iloc[0]['mean_score'] > mejor_score:
                mejor = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if mejor:
        print(f"{tarea.id:<25} {mejor:<25} Acc={mejor_score:.4f}")

# Regresión
for tarea in tareas_regr:
    mejor = None
    mejor_score = -np.inf
    
    for bench in resultados_regr:
        try:
            rankings = bench.rank_learners('regr.r2', task_id=tarea.id)
            if len(rankings) > 0 and rankings.iloc[0]['mean_score'] > mejor_score:
                mejor = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if mejor:
        print(f"{tarea.id:<25} {mejor:<25} R²={mejor_score:.4f}")

# Estadísticas
total_evals = len(tareas_classif) * len(learners_classif) + len(tareas_regr) * len(learners_regr)

print(f"\n\nESTADÍSTICAS:")
print(f"- Datasets evaluados: {len(datasets)}")
print(f"- Modelos de clasificación: {len(learners_classif)}")
print(f"- Modelos de regresión: {len(learners_regr)}")
print(f"- Total de evaluaciones: {total_evals}")

print(f"\nCompletado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n+ Benchmark ejecutado exitosamente!")