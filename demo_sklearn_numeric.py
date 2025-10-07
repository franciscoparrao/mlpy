"""
Demostración de MLPY con modelos reales (solo características numéricas)
======================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime

print("=== MLPY Demo con Modelos Reales ===\n")

# 1. Generar dataset con solo características numéricas
print("1. GENERANDO DATASET")
print("-" * 40)

np.random.seed(42)
n_samples = 1500

# Generar características numéricas
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 2, n_samples)
X3 = np.random.exponential(1, n_samples)
X4 = np.random.gamma(2, 2, n_samples)
X5 = X1 * X2 + np.random.normal(0, 0.5, n_samples)  # Interacción

# Target binario con relación no lineal
y_score = (
    2 * X1 - 0.5 * X2 + 0.3 * X3 - 0.2 * X4 + 
    0.5 * X1 * X2 +  # Interacción
    0.1 * X3**2 +    # No linealidad
    np.random.normal(0, 0.5, n_samples)
)
y = ['Positivo' if score > np.percentile(y_score, 40) else 'Negativo' for score in y_score]

data = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'feature3': X3,
    'feature4': X4,
    'feature5': X5,
    'target': y
})

print(f"Dataset: {data.shape}")
print(f"Distribución del target:")
print(data['target'].value_counts())
print(f"Proporción clase positiva: {(data['target'] == 'Positivo').mean():.1%}")

# 2. Configurar MLPY
print("\n2. CONFIGURANDO MLPY")
print("-" * 40)

from mlpy.tasks import TaskClassif
from mlpy.measures import (
    MeasureClassifAccuracy, MeasureClassifF1,
    MeasureClassifPrecision, MeasureClassifRecall
)
from mlpy.resamplings import ResamplingCV, ResamplingHoldout
from mlpy import resample, benchmark

# Importar modelos
from mlpy.learners import LearnerClassifFeatureless

try:
    from mlpy.learners.sklearn import (
        LearnerLogisticRegression,
        LearnerDecisionTree,
        LearnerRandomForest,
        LearnerGradientBoosting,
        LearnerSVM,
        LearnerKNN
    )
    SKLEARN_AVAILABLE = True
    print("+ Modelos de scikit-learn disponibles")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("- Modelos de scikit-learn no disponibles")

# 3. Crear tarea
print("\n3. CREANDO TAREA DE CLASIFICACIÓN")
print("-" * 40)

task = TaskClassif(data=data, target='target', id='clasificacion_binaria')
print(f"Tarea: {task.id}")
print(f"Clases: {task.class_names}")
print(f"Características: {len(task.feature_names)}")

# 4. Evaluar modelos
print("\n4. EVALUANDO MODELOS")
print("-" * 40)

# Medidas a usar
measures = [
    MeasureClassifAccuracy(),
    MeasureClassifF1(),
    MeasureClassifPrecision(),
    MeasureClassifRecall()
]

# Diccionario para almacenar resultados
resultados = {}

# Modelo baseline
print("\n>>> Modelo Baseline (Featureless):")
baseline = LearnerClassifFeatureless(id='baseline', method='mode')
result_baseline = resample(
    task=task,
    learner=baseline,
    resampling=ResamplingCV(folds=5, stratify=True),
    measures=measures
)
scores_baseline = result_baseline.aggregate()
print(scores_baseline)
resultados['baseline'] = scores_baseline

if SKLEARN_AVAILABLE:
    # Regresión Logística
    print("\n>>> Regresión Logística:")
    logreg = LearnerLogisticRegression(
        id='logreg',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    result_logreg = resample(
        task=task,
        learner=logreg,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=measures
    )
    scores_logreg = result_logreg.aggregate()
    print(scores_logreg)
    resultados['logreg'] = scores_logreg
    
    # Árbol de Decisión
    print("\n>>> Árbol de Decisión:")
    tree = LearnerDecisionTree(
        id='tree',
        max_depth=10,
        min_samples_split=20,
        random_state=42
    )
    result_tree = resample(
        task=task,
        learner=tree,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=measures
    )
    scores_tree = result_tree.aggregate()
    print(scores_tree)
    resultados['tree'] = scores_tree
    
    # Random Forest
    print("\n>>> Random Forest:")
    rf = LearnerRandomForest(
        id='rf',
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42
    )
    result_rf = resample(
        task=task,
        learner=rf,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=measures
    )
    scores_rf = result_rf.aggregate()
    print(scores_rf)
    resultados['rf'] = scores_rf
    
    # Gradient Boosting
    print("\n>>> Gradient Boosting:")
    gb = LearnerGradientBoosting(
        id='gb',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    result_gb = resample(
        task=task,
        learner=gb,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=measures
    )
    scores_gb = result_gb.aggregate()
    print(scores_gb)
    resultados['gb'] = scores_gb
    
    # SVM
    print("\n>>> Support Vector Machine:")
    svm = LearnerSVM(
        id='svm',
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    result_svm = resample(
        task=task,
        learner=svm,
        resampling=ResamplingCV(folds=3, stratify=True),  # Menos folds porque SVM es lento
        measures=measures
    )
    scores_svm = result_svm.aggregate()
    print(scores_svm)
    resultados['svm'] = scores_svm
    
    # KNN
    print("\n>>> K-Nearest Neighbors:")
    knn = LearnerKNN(
        id='knn',
        n_neighbors=10,
        weights='distance'
    )
    result_knn = resample(
        task=task,
        learner=knn,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=measures
    )
    scores_knn = result_knn.aggregate()
    print(scores_knn)
    resultados['knn'] = scores_knn

# 5. Resumen comparativo
print("\n" + "="*80)
print("RESUMEN COMPARATIVO DE MODELOS")
print("="*80)

print(f"\nDataset: {n_samples} muestras, {len(task.feature_names)} características numéricas")
print(f"Distribución de clases: {(data['target'] == 'Positivo').mean():.1%} positivos")

print("\nMétricas por modelo (media ± desviación estándar):")
print("-" * 80)
print(f"{'Modelo':<20} {'Accuracy':<20} {'F1 Score':<20} {'Precision':<20} {'Recall':<20}")
print("-" * 80)

for nombre, scores in resultados.items():
    acc = scores[scores['measure'] == 'classif.acc']
    f1 = scores[scores['measure'] == 'classif.f1']
    prec = scores[scores['measure'] == 'classif.precision']
    rec = scores[scores['measure'] == 'classif.recall']
    
    print(f"{nombre:<20} "
          f"{acc['mean'].values[0]:.3f} ± {acc['std'].values[0]:.3f}      "
          f"{f1['mean'].values[0]:.3f} ± {f1['std'].values[0]:.3f}      "
          f"{prec['mean'].values[0]:.3f} ± {prec['std'].values[0]:.3f}      "
          f"{rec['mean'].values[0]:.3f} ± {rec['std'].values[0]:.3f}")

# 6. Mejor modelo
print("\n" + "-"*80)

# Encontrar mejor modelo por F1
mejor_f1 = 0
mejor_modelo = 'baseline'
for nombre, scores in resultados.items():
    f1_mean = scores[scores['measure'] == 'classif.f1']['mean'].values[0]
    if f1_mean > mejor_f1:
        mejor_f1 = f1_mean
        mejor_modelo = nombre

print(f"\nMejor modelo (por F1 Score): {mejor_modelo.upper()}")
mejor_scores = resultados[mejor_modelo]
print(f"  - Accuracy:  {mejor_scores[mejor_scores['measure'] == 'classif.acc']['mean'].values[0]:.3f}")
print(f"  - F1 Score:  {mejor_scores[mejor_scores['measure'] == 'classif.f1']['mean'].values[0]:.3f}")
print(f"  - Precision: {mejor_scores[mejor_scores['measure'] == 'classif.precision']['mean'].values[0]:.3f}")
print(f"  - Recall:    {mejor_scores[mejor_scores['measure'] == 'classif.recall']['mean'].values[0]:.3f}")

# Calcular mejora sobre baseline
baseline_f1 = resultados['baseline'][resultados['baseline']['measure'] == 'classif.f1']['mean'].values[0]
mejora = ((mejor_f1 - baseline_f1) / baseline_f1 * 100)
print(f"\nMejora sobre baseline: {mejora:+.1f}%")

# 7. Análisis de errores con el mejor modelo
if SKLEARN_AVAILABLE and mejor_modelo != 'baseline':
    print("\n" + "="*80)
    print("ANÁLISIS DEL MEJOR MODELO")
    print("="*80)
    
    # Crear el mejor modelo
    if mejor_modelo == 'logreg':
        mejor_learner = logreg
    elif mejor_modelo == 'tree':
        mejor_learner = tree
    elif mejor_modelo == 'rf':
        mejor_learner = rf
    elif mejor_modelo == 'gb':
        mejor_learner = gb
    elif mejor_modelo == 'svm':
        mejor_learner = svm
    elif mejor_modelo == 'knn':
        mejor_learner = knn
    
    # Evaluar en holdout para análisis final
    print(f"\nEvaluando {mejor_modelo} en conjunto de prueba (80/20 split)...")
    final_result = resample(
        task=task,
        learner=mejor_learner,
        resampling=ResamplingHoldout(ratio=0.8, stratify=True),
        measures=measures
    )
    
    final_scores = final_result.aggregate()
    print("\nResultados en conjunto de prueba:")
    print(final_scores)
    
    # Entrenar en todos los datos para análisis
    mejor_learner.train(task)
    
    # Hacer predicciones en una muestra
    indices_muestra = list(range(20))
    predicciones = mejor_learner.predict(task, row_ids=indices_muestra)
    
    print("\nMuestra de predicciones:")
    print("-" * 60)
    print(f"{'Índice':<8} {'Real':<10} {'Predicción':<12} {'Correcto':<10}")
    print("-" * 60)
    
    aciertos = 0
    for i, idx in enumerate(indices_muestra):
        real = predicciones.truth[i]
        pred = predicciones.response[i]
        correcto = real == pred
        aciertos += correcto
        print(f"{idx:<8} {real:<10} {pred:<12} {'Sí' if correcto else 'No':<10}")
    
    print(f"\nAciertos en la muestra: {aciertos}/{len(indices_muestra)} ({aciertos/len(indices_muestra):.1%})")

print("\n" + "="*80)
print(f"Completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("+ Demo completada exitosamente!")