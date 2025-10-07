"""
Demostración simple de MLPY con modelos reales
==============================================
"""

import numpy as np
import pandas as pd
from datetime import datetime

print("=== MLPY Demo con Modelos Reales ===\n")

# 1. Generar dataset
print("1. GENERANDO DATASET")
print("-" * 40)

np.random.seed(42)
n_samples = 1000

# Dataset simplificado
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 2, n_samples) 
X3 = np.random.choice(['A', 'B', 'C'], n_samples)

# Target binario con relación clara
y_score = 2 * X1 - 0.5 * X2 + (X3 == 'A').astype(float) + np.random.normal(0, 0.5, n_samples)
y = ['Si' if score > 0 else 'No' for score in y_score]

data = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'categoria': X3,
    'target': y
})

print(f"Dataset: {data.shape}")
print(f"Distribución del target:")
print(data['target'].value_counts())

# 2. Configurar MLPY
print("\n2. CONFIGURANDO MLPY")
print("-" * 40)

from mlpy.tasks import TaskClassif
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
from mlpy.resamplings import ResamplingCV
from mlpy import resample

# Importar solo modelos básicos para evitar problemas con pipelines
from mlpy.learners import LearnerClassifFeatureless

try:
    from mlpy.learners.sklearn import (
        LearnerLogisticRegression,
        LearnerDecisionTree,
        LearnerRandomForest
    )
    SKLEARN_AVAILABLE = True
    print("+ Modelos de scikit-learn disponibles")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("- Modelos de scikit-learn no disponibles")

# 3. Crear tarea
print("\n3. CREANDO TAREA")
print("-" * 40)

task = TaskClassif(data=data, target='target', id='clasificacion')
print(f"Tarea: {task.id}")
print(f"Clases: {task.class_names}")

# 4. Crear y evaluar modelos
print("\n4. EVALUANDO MODELOS")
print("-" * 40)

# Modelo baseline
print("\nModelo Baseline:")
baseline = LearnerClassifFeatureless(id='baseline', method='mode')

result_baseline = resample(
    task=task,
    learner=baseline,
    resampling=ResamplingCV(folds=5, stratify=True),
    measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
)

scores_baseline = result_baseline.aggregate()
print(scores_baseline)

if SKLEARN_AVAILABLE:
    # Logistic Regression
    print("\nRegresión Logística:")
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
        measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
    )
    
    scores_logreg = result_logreg.aggregate()
    print(scores_logreg)
    
    # Decision Tree
    print("\nÁrbol de Decisión:")
    tree = LearnerDecisionTree(
        id='tree',
        max_depth=5,
        random_state=42
    )
    
    result_tree = resample(
        task=task,
        learner=tree,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
    )
    
    scores_tree = result_tree.aggregate()
    print(scores_tree)
    
    # Random Forest
    print("\nRandom Forest:")
    rf = LearnerRandomForest(
        id='rf',
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    result_rf = resample(
        task=task,
        learner=rf,
        resampling=ResamplingCV(folds=5, stratify=True),
        measures=[MeasureClassifAccuracy(), MeasureClassifF1()]
    )
    
    scores_rf = result_rf.aggregate()
    print(scores_rf)

# 5. Resumen
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS")
print("="*60)

print(f"\nDataset: {n_samples} muestras")
print(f"Características: {len(task.feature_names)}")

print("\nComparación de modelos:")
print("-" * 60)
print(f"{'Modelo':<20} {'Accuracy':<15} {'F1 Score':<15}")
print("-" * 60)

# Baseline
acc_baseline = scores_baseline[scores_baseline['measure'] == 'classif.acc']['mean'].values[0]
f1_baseline = scores_baseline[scores_baseline['measure'] == 'classif.f1']['mean'].values[0]
print(f"{'Baseline':<20} {acc_baseline:<15.3f} {f1_baseline:<15.3f}")

if SKLEARN_AVAILABLE:
    # Logistic Regression
    acc_logreg = scores_logreg[scores_logreg['measure'] == 'classif.acc']['mean'].values[0]
    f1_logreg = scores_logreg[scores_logreg['measure'] == 'classif.f1']['mean'].values[0]
    print(f"{'Regresión Logística':<20} {acc_logreg:<15.3f} {f1_logreg:<15.3f}")
    
    # Decision Tree
    acc_tree = scores_tree[scores_tree['measure'] == 'classif.acc']['mean'].values[0]
    f1_tree = scores_tree[scores_tree['measure'] == 'classif.f1']['mean'].values[0]
    print(f"{'Árbol de Decisión':<20} {acc_tree:<15.3f} {f1_tree:<15.3f}")
    
    # Random Forest
    acc_rf = scores_rf[scores_rf['measure'] == 'classif.acc']['mean'].values[0]
    f1_rf = scores_rf[scores_rf['measure'] == 'classif.f1']['mean'].values[0]
    print(f"{'Random Forest':<20} {acc_rf:<15.3f} {f1_rf:<15.3f}")
    
    # Calcular mejora
    mejor_acc = max(acc_logreg, acc_tree, acc_rf)
    mejora_acc = ((mejor_acc - acc_baseline) / acc_baseline * 100)
    print(f"\nMejora sobre baseline: {mejora_acc:+.1f}%")

print("\n+ Demo completada exitosamente!")