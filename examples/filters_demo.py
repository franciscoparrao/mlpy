"""
Demo del sistema de filtros de características en MLPY.

Inspirado en mlr3filters, muestra cómo usar diferentes
métodos de selección de características.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.filters import (
    create_filter, list_filters, filter_features,
    FilterEnsemble, FilterStability
)
from mlpy.pipelines import PipeOpFilter, PipeOpFilterCorr, linear_pipeline, GraphLearner, PipeOpLearner
from mlpy.learners import learner_sklearn
from mlpy import benchmark
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy

print("="*60)
print("SISTEMA DE FILTROS MLPY (estilo mlr3filters)")
print("="*60)

# Crear dataset con muchas características
print("\nCreando dataset con 100 características (20 informativas)...")
X, y = make_classification(
    n_samples=500,
    n_features=100,
    n_informative=20,
    n_redundant=30,
    n_repeated=10,
    n_classes=3,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feat_{i:03d}" for i in range(100)])
df['target'] = y

task = TaskClassif(data=df, target='target')
print(f"Task: {task.nrow} muestras, {len(task.feature_names)} características, {task.n_classes} clases")

# Ejemplo 1: Filtros Univariados
print("\n" + "="*60)
print("EJEMPLO 1: Filtros Univariados")
print("="*60)

print("\nFiltros disponibles:")
print(list_filters())

# ANOVA
print("\n1.1 ANOVA Filter:")
anova_filter = create_filter('anova')
anova_result = anova_filter.calculate(task)

print(f"Top 10 características por ANOVA:")
for i, feat in enumerate(anova_result.features[:10]):
    print(f"  {i+1}. {feat}: {anova_result.scores[feat]:.4f}")

# Mutual Information
print("\n1.2 Mutual Information Filter:")
mi_filter = create_filter('mutual_info', n_neighbors=5)
mi_result = mi_filter.calculate(task)

print(f"Top 10 características por MI:")
for i, feat in enumerate(mi_result.features[:10]):
    print(f"  {i+1}. {feat}: {mi_result.scores[feat]:.4f}")

# Correlation
print("\n1.3 Correlation Filter:")
corr_filter = create_filter('correlation', method='spearman')
corr_result = corr_filter.calculate(task)

print(f"Top 10 características por correlación:")
for i, feat in enumerate(corr_result.features[:10]):
    print(f"  {i+1}. {feat}: {corr_result.scores[feat]:.4f}")

# Ejemplo 2: Filtros Multivariados
print("\n" + "="*60)
print("EJEMPLO 2: Filtros Multivariados")
print("="*60)

# Importance (Random Forest)
print("\n2.1 Feature Importance (Random Forest):")
importance_filter = create_filter('importance', n_estimators=50)
imp_result = importance_filter.calculate(task)

print(f"Top 10 características por importancia:")
for i, feat in enumerate(imp_result.features[:10]):
    print(f"  {i+1}. {feat}: {imp_result.scores[feat]:.4f}")

# mRMR
print("\n2.2 mRMR (Maximum Relevance Minimum Redundancy):")
mrmr_filter = create_filter('mrmr', n_features=20, relevance_func='mutual_info')
mrmr_result = mrmr_filter.calculate(task)

print(f"Top 10 características por mRMR:")
for i, feat in enumerate(mrmr_result.features[:10]):
    print(f"  {i+1}. {feat}: {mrmr_result.scores[feat]:.4f}")

# Ejemplo 3: Filtros Ensemble
print("\n" + "="*60)
print("EJEMPLO 3: Filtros Ensemble")
print("="*60)

# Ensemble de múltiples filtros
print("\n3.1 Ensemble de filtros:")
ensemble_filter = FilterEnsemble(
    filters=['anova', 'mutual_info', 'correlation', 'importance'],
    aggregation='rank_mean',
    normalize=True
)

ensemble_result = ensemble_filter.calculate(task)

print(f"Top 10 características por ensemble:")
for i, feat in enumerate(ensemble_result.features[:10]):
    print(f"  {i+1}. {feat}: {ensemble_result.scores[feat]:.4f}")

# Stability Selection
print("\n3.2 Stability Selection:")
stability_filter = FilterStability(
    filter='anova',
    n_iterations=50,
    sample_fraction=0.5,
    n_features=30
)

print("Ejecutando stability selection (50 iteraciones)...")
stability_result = stability_filter.calculate(task)

print(f"\nCaracterísticas más estables (frecuencia de selección):")
stable_features = stability_result.scores[stability_result.scores > 0.6]
for feat, freq in stable_features.items():
    print(f"  {feat}: {freq:.2%}")

# Ejemplo 4: Integración con Pipelines
print("\n" + "="*60)
print("EJEMPLO 4: Filtros en Pipelines")
print("="*60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Pipeline 1: Filter + Classifier
print("\n4.1 Pipeline simple con filtro:")
pipeline1 = linear_pipeline(
    PipeOpFilter(method='anova', k=20),
    PipeOpLearner(learner_sklearn(RandomForestClassifier(n_estimators=50)))
)
learner1 = GraphLearner(pipeline1, id="anova_rf")

# Pipeline 2: Correlation removal + Filter + Classifier  
print("\n4.2 Pipeline con eliminación de correlación:")
pipeline2 = linear_pipeline(
    PipeOpFilterCorr(threshold=0.9),
    PipeOpFilter(method='mutual_info', k=15),
    PipeOpLearner(learner_sklearn(SVC(probability=True)))
)
learner2 = GraphLearner(pipeline2, id="decorr_mi_svm")

# Pipeline 3: Auto filter
print("\n4.3 Pipeline con selección automática:")
pipeline3 = linear_pipeline(
    PipeOpFilter(method='auto', k=25),
    PipeOpLearner(learner_sklearn(RandomForestClassifier(n_estimators=50)))
)
learner3 = GraphLearner(pipeline3, id="auto_rf")

# Benchmark
print("\nEjecutando benchmark de pipelines...")
bench_result = benchmark(
    tasks=[task],
    learners=[learner1, learner2, learner3],
    resampling=ResamplingCV(folds=3),
    measures=MeasureClassifAccuracy()
)

print("\nResultados:")
print(bench_result.aggregate())

# Ejemplo 5: Visualización
print("\n" + "="*60)
print("EJEMPLO 5: Visualización de Scores")
print("="*60)

# Comparar scores de diferentes métodos
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top 20 features por método
top_n = 20

# ANOVA
ax = axes[0, 0]
anova_top = anova_result.scores.nlargest(top_n)
anova_top.plot(kind='barh', ax=ax)
ax.set_title('ANOVA Scores')
ax.set_xlabel('F-value')

# Mutual Information
ax = axes[0, 1]
mi_top = mi_result.scores.nlargest(top_n)
mi_top.plot(kind='barh', ax=ax, color='orange')
ax.set_title('Mutual Information Scores')
ax.set_xlabel('MI Score')

# Importance
ax = axes[1, 0]
imp_top = imp_result.scores.nlargest(top_n)
imp_top.plot(kind='barh', ax=ax, color='green')
ax.set_title('Random Forest Importance')
ax.set_xlabel('Importance')

# Ensemble
ax = axes[1, 1]
ensemble_top = ensemble_result.scores.nlargest(top_n)
ensemble_top.plot(kind='barh', ax=ax, color='red')
ax.set_title('Ensemble Scores')
ax.set_xlabel('Combined Score')

plt.tight_layout()
plt.savefig('filter_scores_comparison.png', dpi=150)
print("\nGráficos guardados en 'filter_scores_comparison.png'")

# Ejemplo 6: Uso Rápido
print("\n" + "="*60)
print("EJEMPLO 6: Funciones de Conveniencia")
print("="*60)

# Selección rápida
print("\n6.1 Selección rápida con filter_features():")
top_anova = filter_features(task, method='anova', k=10)
print(f"Top 10 por ANOVA: {top_anova}")

top_percentile = filter_features(task, method='mutual_info', percentile=90)
print(f"\nTop 10% por MI: {len(top_percentile)} características")

# Comparación de métodos
print("\n6.2 Overlap entre métodos:")
top_k = 15
methods = ['anova', 'mutual_info', 'correlation', 'importance']
selected_by_method = {}

for method in methods:
    selected_by_method[method] = set(filter_features(task, method=method, k=top_k))

# Calcular intersecciones
print(f"\nCaracterísticas seleccionadas por TODOS los métodos (k={top_k}):")
common = set.intersection(*selected_by_method.values())
print(f"  {len(common)} características: {sorted(common)}")

print(f"\nCaracterísticas únicas por método:")
for method in methods:
    unique = selected_by_method[method] - set.union(*[s for m, s in selected_by_method.items() if m != method])
    print(f"  {method}: {len(unique)} únicas")

# Resumen
print("\n" + "="*60)
print("RESUMEN DEL SISTEMA DE FILTROS")
print("="*60)
print("""
Características del sistema de filtros MLPY:

1. FILTROS UNIVARIADOS:
   - ANOVA, F-regression, Chi-squared
   - Mutual Information, Correlation
   - Variance threshold

2. FILTROS MULTIVARIADOS:
   - Feature Importance (RF, permutation)
   - RFE (Recursive Feature Elimination)
   - mRMR (Max Relevance Min Redundancy)
   - Relief

3. FILTROS ENSEMBLE:
   - Combinación de múltiples métodos
   - Stability selection
   - Auto-selección basada en datos

4. INTEGRACIÓN:
   - PipeOpFilter para pipelines
   - API simple con filter_features()
   - Compatible con todo el ecosistema MLPY

5. VENTAJAS:
   - Múltiples estrategias de selección
   - Fácil experimentación
   - Visualización de importancias
   - Escalable a grandes datasets
""")