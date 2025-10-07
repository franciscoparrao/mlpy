"""
Benchmark Completo de MLPY
==========================

Este script ejecuta un benchmark exhaustivo comparando todos los modelos
disponibles en MLPY en múltiples datasets.
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
from mlpy.measures import (
    # Clasificación
    MeasureClassifAccuracy, MeasureClassifF1, MeasureClassifAUC,
    MeasureClassifPrecision, MeasureClassifRecall,
    # Regresión
    MeasureRegrMSE, MeasureRegrMAE, MeasureRegrR2
)
from mlpy.resamplings import ResamplingCV
from mlpy import benchmark

# Importar todos los learners disponibles
from mlpy.learners import (
    # Baseline
    LearnerClassifFeatureless, LearnerRegrFeatureless,
    LearnerClassifDebug, LearnerRegrDebug
)

# Intentar importar sklearn learners
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
# 1. CREAR DATASETS DE PRUEBA
# ============================================================================
print("\n1. CREANDO DATASETS DE PRUEBA")
print("-" * 40)

np.random.seed(42)

# Dataset 1: Clasificación binaria simple
def crear_dataset_binario_simple(n=1000):
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.uniform(-1, 1, n)
    y = (X1 + X2 + 0.5*X3 + np.random.normal(0, 0.3, n)) > 0
    
    return pd.DataFrame({
        'x1': X1, 'x2': X2, 'x3': X3,
        'target': ['Clase1' if yi else 'Clase0' for yi in y]
    })

# Dataset 2: Clasificación multiclase
def crear_dataset_multiclase(n=1200):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=1,
        n_classes=3, n_clusters_per_class=2, random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(6)])
    df['target'] = ['Clase_' + str(yi) for yi in y]
    return df

# Dataset 3: Regresión simple
def crear_dataset_regresion_simple(n=800):
    X1 = np.random.uniform(-2, 2, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.exponential(1, n)
    y = 2*X1 + X2**2 - 0.5*X3 + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'x1': X1, 'x2': X2, 'x3': X3,
        'target': y
    })

# Dataset 4: Regresión con no linealidades
def crear_dataset_regresion_compleja(n=1000):
    X1 = np.random.uniform(-3, 3, n)
    X2 = np.random.uniform(-3, 3, n)
    X3 = np.random.normal(0, 1, n)
    X4 = np.random.gamma(2, 1, n)
    
    y = (np.sin(X1) + X2**2 + 0.5*X3*X4 + 
         0.1*X1*X2 + np.random.normal(0, 0.3, n))
    
    return pd.DataFrame({
        'x1': X1, 'x2': X2, 'x3': X3, 'x4': X4,
        'target': y
    })

# Crear datasets
datasets = {
    'binario_simple': crear_dataset_binario_simple(1000),
    'multiclase': crear_dataset_multiclase(1200),
    'regresion_simple': crear_dataset_regresion_simple(800),
    'regresion_compleja': crear_dataset_regresion_compleja(1000)
}

for nombre, df in datasets.items():
    print(f"\n{nombre}:")
    print(f"  - Forma: {df.shape}")
    print(f"  - Características: {df.shape[1]-1}")
    if 'regresion' not in nombre:
        print(f"  - Clases: {df['target'].nunique()}")
        print(f"  - Distribución: {dict(df['target'].value_counts())}")
    else:
        print(f"  - Rango target: [{df['target'].min():.2f}, {df['target'].max():.2f}]")

# ============================================================================
# 2. CREAR TAREAS
# ============================================================================
print("\n2. CREANDO TAREAS MLPY")
print("-" * 40)

tareas_classif = []
tareas_regr = []

# Tareas de clasificación
for nombre in ['binario_simple', 'multiclase']:
    task = TaskClassif(
        data=datasets[nombre],
        target='target',
        id=f'task_{nombre}'
    )
    tareas_classif.append(task)
    print(f"+ Tarea creada: {task.id} (clasificación)")

# Tareas de regresión
for nombre in ['regresion_simple', 'regresion_compleja']:
    task = TaskRegr(
        data=datasets[nombre],
        target='target',
        id=f'task_{nombre}'
    )
    tareas_regr.append(task)
    print(f"+ Tarea creada: {task.id} (regresión)")

# ============================================================================
# 3. DEFINIR LEARNERS
# ============================================================================
print("\n3. CONFIGURANDO LEARNERS")
print("-" * 40)

# Learners de clasificación
learners_classif = [
    # Baseline
    LearnerClassifFeatureless(id='baseline_mode', method='mode'),
    LearnerClassifFeatureless(id='baseline_sample', method='sample'),
    LearnerClassifFeatureless(id='baseline_weighted', method='weighted'),
]

if SKLEARN_AVAILABLE:
    learners_classif.extend([
        # Lineales
        LearnerLogisticRegression(id='logreg', C=1.0, max_iter=1000),
        LearnerLogisticRegression(id='logreg_l1', penalty='l1', solver='liblinear', C=1.0),
        LearnerLogisticRegression(id='logreg_elastic', penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000),
        
        # Árboles
        LearnerDecisionTree(id='tree_shallow', max_depth=5),
        LearnerDecisionTree(id='tree_deep', max_depth=20),
        LearnerDecisionTree(id='tree_pruned', max_depth=10, min_samples_split=20, min_samples_leaf=10),
        
        # Ensemble
        LearnerRandomForest(id='rf_small', n_estimators=50, max_depth=10),
        LearnerRandomForest(id='rf_large', n_estimators=200, max_depth=None),
        LearnerGradientBoosting(id='gb_weak', n_estimators=50, learning_rate=0.1, max_depth=3),
        LearnerGradientBoosting(id='gb_strong', n_estimators=150, learning_rate=0.05, max_depth=5),
        
        # SVM
        LearnerSVM(id='svm_linear', kernel='linear', C=1.0),
        LearnerSVM(id='svm_rbf', kernel='rbf', C=1.0, gamma='scale'),
        LearnerSVM(id='svm_poly', kernel='poly', degree=3, C=1.0),
        
        # Vecinos
        LearnerKNN(id='knn_5', n_neighbors=5),
        LearnerKNN(id='knn_10', n_neighbors=10),
        LearnerKNN(id='knn_weighted', n_neighbors=10, weights='distance'),
        
        # Probabilísticos
        LearnerNaiveBayes(id='naive_bayes'),
        
        # Redes Neuronales
        LearnerMLPClassifier(id='mlp_small', hidden_layer_sizes=(50,), max_iter=500),
        LearnerMLPClassifier(id='mlp_medium', hidden_layer_sizes=(100, 50), max_iter=500),
        LearnerMLPClassifier(id='mlp_large', hidden_layer_sizes=(100, 50, 25), max_iter=500),
    ])

print(f"Learners de clasificación configurados: {len(learners_classif)}")

# Learners de regresión
learners_regr = [
    # Baseline
    LearnerRegrFeatureless(id='baseline_mean', method='mean'),
    LearnerRegrFeatureless(id='baseline_median', method='median'),
    LearnerRegrFeatureless(id='baseline_sample', method='sample'),
]

if SKLEARN_AVAILABLE:
    learners_regr.extend([
        # Lineales
        LearnerLinearRegression(id='linear'),
        LearnerRidge(id='ridge_weak', alpha=0.1),
        LearnerRidge(id='ridge_strong', alpha=10.0),
        LearnerLasso(id='lasso_weak', alpha=0.01),
        LearnerLasso(id='lasso_strong', alpha=1.0),
        LearnerElasticNet(id='elastic_balanced', alpha=0.1, l1_ratio=0.5),
        LearnerElasticNet(id='elastic_l1', alpha=0.1, l1_ratio=0.9),
        
        # Árboles
        LearnerDecisionTreeRegressor(id='tree_reg_shallow', max_depth=5),
        LearnerDecisionTreeRegressor(id='tree_reg_deep', max_depth=20),
        
        # Ensemble
        LearnerRandomForestRegressor(id='rf_reg_small', n_estimators=50, max_depth=10),
        LearnerRandomForestRegressor(id='rf_reg_large', n_estimators=200),
        LearnerGradientBoostingRegressor(id='gb_reg_weak', n_estimators=50, learning_rate=0.1),
        LearnerGradientBoostingRegressor(id='gb_reg_strong', n_estimators=150, learning_rate=0.05),
        
        # SVM
        LearnerSVR(id='svr_linear', kernel='linear', C=1.0),
        LearnerSVR(id='svr_rbf', kernel='rbf', C=1.0, gamma='scale'),
        
        # Vecinos
        LearnerKNNRegressor(id='knn_reg_5', n_neighbors=5),
        LearnerKNNRegressor(id='knn_reg_weighted', n_neighbors=10, weights='distance'),
        
        # Redes Neuronales
        LearnerMLPRegressor(id='mlp_reg_small', hidden_layer_sizes=(50,), max_iter=500),
        LearnerMLPRegressor(id='mlp_reg_large', hidden_layer_sizes=(100, 50), max_iter=500),
    ])

print(f"Learners de regresión configurados: {len(learners_regr)}")

# ============================================================================
# 4. EJECUTAR BENCHMARKS
# ============================================================================
print("\n4. EJECUTANDO BENCHMARKS")
print("-" * 40)

# Benchmark de clasificación
print("\n>>> BENCHMARK DE CLASIFICACIÓN")
print("Esto puede tomar varios minutos...")

medidas_classif = [
    MeasureClassifAccuracy(),
    MeasureClassifF1(),
    MeasureClassifPrecision(),
    MeasureClassifRecall()
]

# Usar menos folds para modelos lentos
learners_rapidos = [l for l in learners_classif if 'svm' not in l.id and 'mlp' not in l.id]
learners_lentos = [l for l in learners_classif if 'svm' in l.id or 'mlp' in l.id]

# Benchmark con learners rápidos (5 folds)
if learners_rapidos:
    print(f"\nEvaluando {len(learners_rapidos)} modelos rápidos con 5-fold CV...")
    benchmark_classif_rapido = benchmark(
        tasks=tareas_classif,
        learners=learners_rapidos,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=medidas_classif
    )
    print("+ Benchmark de modelos rápidos completado")

# Benchmark con learners lentos (3 folds)
if learners_lentos and SKLEARN_AVAILABLE:
    print(f"\nEvaluando {len(learners_lentos)} modelos lentos con 3-fold CV...")
    benchmark_classif_lento = benchmark(
        tasks=tareas_classif,
        learners=learners_lentos,
        resampling=ResamplingCV(folds=3, stratify=True),
        measures=medidas_classif
    )
    print("+ Benchmark de modelos lentos completado")

# Benchmark de regresión
print("\n>>> BENCHMARK DE REGRESIÓN")

medidas_regr = [
    MeasureRegrMSE(),
    MeasureRegrMAE(),
    MeasureRegrR2()
]

# Similar división para regresión
learners_regr_rapidos = [l for l in learners_regr if 'svr' not in l.id and 'mlp' not in l.id]
learners_regr_lentos = [l for l in learners_regr if 'svr' in l.id or 'mlp' in l.id]

if learners_regr_rapidos:
    print(f"\nEvaluando {len(learners_regr_rapidos)} modelos rápidos con 5-fold CV...")
    benchmark_regr_rapido = benchmark(
        tasks=tareas_regr,
        learners=learners_regr_rapidos,
        resampling=ResamplingCV(folds=5),
        measures=medidas_regr
    )
    print("+ Benchmark de regresión (rápido) completado")

if learners_regr_lentos and SKLEARN_AVAILABLE:
    print(f"\nEvaluando {len(learners_regr_lentos)} modelos lentos con 3-fold CV...")
    benchmark_regr_lento = benchmark(
        tasks=tareas_regr,
        learners=learners_regr_lentos,
        resampling=ResamplingCV(folds=3),
        measures=medidas_regr
    )
    print("+ Benchmark de regresión (lento) completado")

# ============================================================================
# 5. ANÁLISIS DE RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("ANÁLISIS DE RESULTADOS")
print("="*80)

# Resultados de clasificación
print("\n>>> RESULTADOS DE CLASIFICACIÓN")
print("-" * 80)

for tarea in tareas_classif:
    print(f"\nDataset: {tarea.id}")
    print("~" * 40)
    
    # Combinar resultados si tenemos ambos benchmarks
    todos_resultados = []
    
    if 'benchmark_classif_rapido' in locals():
        for medida in ['classif.acc', 'classif.f1']:
            try:
                rankings = benchmark_classif_rapido.rank_learners(medida, task_id=tarea.id)
                todos_resultados.extend([(row['learner'], row['mean_score'], medida) 
                                       for _, row in rankings.iterrows()])
            except:
                pass
    
    if 'benchmark_classif_lento' in locals() and SKLEARN_AVAILABLE:
        for medida in ['classif.acc', 'classif.f1']:
            try:
                rankings = benchmark_classif_lento.rank_learners(medida, task_id=tarea.id)
                todos_resultados.extend([(row['learner'], row['mean_score'], medida) 
                                       for _, row in rankings.iterrows()])
            except:
                pass
    
    # Mostrar top 10 por F1
    f1_scores = [(l, s) for l, s, m in todos_resultados if m == 'classif.f1']
    f1_scores = list(dict.fromkeys(f1_scores))  # Eliminar duplicados
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 modelos por F1 Score:")
    print(f"{'Rank':<6} {'Modelo':<25} {'F1 Score':<10}")
    print("-" * 45)
    for i, (learner, score) in enumerate(f1_scores[:10], 1):
        print(f"{i:<6} {learner:<25} {score:.4f}")

# Resultados de regresión
print("\n\n>>> RESULTADOS DE REGRESIÓN")
print("-" * 80)

for tarea in tareas_regr:
    print(f"\nDataset: {tarea.id}")
    print("~" * 40)
    
    # Combinar resultados
    todos_resultados = []
    
    if 'benchmark_regr_rapido' in locals():
        try:
            rankings = benchmark_regr_rapido.rank_learners('regr.r2', task_id=tarea.id)
            todos_resultados.extend([(row['learner'], row['mean_score']) 
                                   for _, row in rankings.iterrows()])
        except:
            pass
    
    if 'benchmark_regr_lento' in locals() and SKLEARN_AVAILABLE:
        try:
            rankings = benchmark_regr_lento.rank_learners('regr.r2', task_id=tarea.id)
            todos_resultados.extend([(row['learner'], row['mean_score']) 
                                   for _, row in rankings.iterrows()])
        except:
            pass
    
    # Eliminar duplicados y ordenar
    todos_resultados = list(dict.fromkeys(todos_resultados))
    todos_resultados.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 modelos por R² Score:")
    print(f"{'Rank':<6} {'Modelo':<25} {'R² Score':<10}")
    print("-" * 45)
    for i, (learner, score) in enumerate(todos_resultados[:10], 1):
        print(f"{i:<6} {learner:<25} {score:.4f}")

# ============================================================================
# 6. ANÁLISIS AGREGADO
# ============================================================================
print("\n" + "="*80)
print("ANÁLISIS AGREGADO")
print("="*80)

# Mejor modelo por dataset
print("\nMEJOR MODELO POR DATASET:")
print("-" * 60)
print(f"{'Dataset':<25} {'Mejor Modelo':<25} {'Score':<10}")
print("-" * 60)

# Para cada tarea de clasificación
for tarea in tareas_classif:
    mejor_modelo = None
    mejor_score = -np.inf
    
    if 'benchmark_classif_rapido' in locals():
        try:
            rankings = benchmark_classif_rapido.rank_learners('classif.f1', task_id=tarea.id)
            if len(rankings) > 0:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if 'benchmark_classif_lento' in locals() and SKLEARN_AVAILABLE:
        try:
            rankings = benchmark_classif_lento.rank_learners('classif.f1', task_id=tarea.id)
            if len(rankings) > 0 and rankings.iloc[0]['mean_score'] > mejor_score:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if mejor_modelo:
        print(f"{tarea.id:<25} {mejor_modelo:<25} {mejor_score:.4f}")

# Para cada tarea de regresión
for tarea in tareas_regr:
    mejor_modelo = None
    mejor_score = -np.inf
    
    if 'benchmark_regr_rapido' in locals():
        try:
            rankings = benchmark_regr_rapido.rank_learners('regr.r2', task_id=tarea.id)
            if len(rankings) > 0:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if 'benchmark_regr_lento' in locals() and SKLEARN_AVAILABLE:
        try:
            rankings = benchmark_regr_lento.rank_learners('regr.r2', task_id=tarea.id)
            if len(rankings) > 0 and rankings.iloc[0]['mean_score'] > mejor_score:
                mejor_modelo = rankings.iloc[0]['learner']
                mejor_score = rankings.iloc[0]['mean_score']
        except:
            pass
    
    if mejor_modelo:
        print(f"{tarea.id:<25} {mejor_modelo:<25} {mejor_score:.4f}")

# Resumen de modelos evaluados
print(f"\n\nRESUMEN:")
print(f"- Tareas de clasificación evaluadas: {len(tareas_classif)}")
print(f"- Tareas de regresión evaluadas: {len(tareas_regr)}")
print(f"- Modelos de clasificación evaluados: {len(learners_classif)}")
print(f"- Modelos de regresión evaluados: {len(learners_regr)}")
print(f"- Total de evaluaciones: {len(tareas_classif) * len(learners_classif) + len(tareas_regr) * len(learners_regr)}")

print(f"\nCompletado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n+ Benchmark completo ejecutado exitosamente!")