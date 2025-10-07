"""
Demo de filtros basados en teoría de la información.

Muestra el uso de Information Gain, Information Gain Ratio,
Symmetrical Uncertainty y otros métodos.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif
from mlpy.filters import (
    FilterInformationGain,
    FilterInformationGainRatio,
    FilterSymmetricalUncertainty,
    FilterJMIM,
    create_filter
)

print("="*60)
print("FILTROS DE TEORÍA DE LA INFORMACIÓN EN MLPY")
print("="*60)

# Crear dataset con características de diferente calidad informativa
print("\nCreando dataset de prueba...")

# Dataset con características específicas
np.random.seed(42)
n_samples = 1000

# Características informativas (alta relación con target)
X_informative = np.random.randn(n_samples, 5)

# Target basado en las características informativas
y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)
y[X_informative[:, 2] > 1] = 2  # Tercera clase

# Características redundantes (correlacionadas con informativas)
X_redundant = np.column_stack([
    X_informative[:, 0] + 0.5 * np.random.randn(n_samples),
    X_informative[:, 1] * 2 + np.random.randn(n_samples),
    X_informative[:, 2] - 0.3 * np.random.randn(n_samples)
])

# Características con muchos valores únicos (alta entropía)
X_high_entropy = np.column_stack([
    np.random.choice(20, n_samples),  # 20 valores posibles
    np.random.choice(50, n_samples),  # 50 valores posibles
    np.random.uniform(0, 100, n_samples)  # Continua
])

# Características ruidosas (poca información)
X_noise = np.random.randn(n_samples, 5)

# Combinar todas las características
X = np.column_stack([X_informative, X_redundant, X_high_entropy, X_noise])

# Crear nombres descriptivos
feature_names = (
    [f"info_{i}" for i in range(5)] +
    [f"redun_{i}" for i in range(3)] +
    ["high_entropy_20", "high_entropy_50", "high_entropy_cont"] +
    [f"noise_{i}" for i in range(5)]
)

# Crear DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Crear task
task = TaskClassif(data=df, target='target')
print(f"Task: {task.nrow} muestras, {len(task.feature_names)} características, {task.n_classes} clases")

# Ejemplo 1: Information Gain
print("\n" + "="*60)
print("1. INFORMATION GAIN")
print("="*60)

ig_filter = FilterInformationGain(n_bins=10, strategy='quantile')
ig_result = ig_filter.calculate(task)

print("\nTop características por Information Gain:")
print("(Mide la reducción en entropía del target)")
print("-"*40)
for i, (feat, score) in enumerate(ig_result.scores.nlargest(10).items()):
    print(f"{i+1:2d}. {feat:20s}: {score:.4f} bits")

# Ejemplo 2: Information Gain Ratio
print("\n" + "="*60)
print("2. INFORMATION GAIN RATIO")
print("="*60)

igr_filter = FilterInformationGainRatio(n_bins=10, strategy='quantile')
igr_result = igr_filter.calculate(task)

print("\nTop características por Information Gain Ratio:")
print("(Normaliza IG por la entropía de la característica)")
print("-"*40)
for i, (feat, score) in enumerate(igr_result.scores.nlargest(10).items()):
    print(f"{i+1:2d}. {feat:20s}: {score:.4f}")

# Comparar IG vs IGR
print("\n" + "-"*40)
print("Comparación IG vs IGR para características con alta entropía:")
for feat in ["high_entropy_20", "high_entropy_50", "high_entropy_cont"]:
    ig_score = ig_result.scores[feat]
    igr_score = igr_result.scores[feat]
    ig_rank = list(ig_result.features).index(feat) + 1
    igr_rank = list(igr_result.features).index(feat) + 1
    print(f"{feat:20s}: IG={ig_score:.4f} (rank {ig_rank:2d}) | IGR={igr_score:.4f} (rank {igr_rank:2d})")

# Ejemplo 3: Symmetrical Uncertainty
print("\n" + "="*60)
print("3. SYMMETRICAL UNCERTAINTY")
print("="*60)

su_filter = FilterSymmetricalUncertainty(n_bins=10)
su_result = su_filter.calculate(task)

print("\nTop características por Symmetrical Uncertainty:")
print("(Información mutua normalizada, valores en [0,1])")
print("-"*40)
for i, (feat, score) in enumerate(su_result.scores.nlargest(10).items()):
    print(f"{i+1:2d}. {feat:20s}: {score:.4f}")

# Ejemplo 4: JMIM (Joint Mutual Information Maximization)
print("\n" + "="*60)
print("4. JMIM - Joint Mutual Information Maximization")
print("="*60)

jmim_filter = FilterJMIM(n_features=10, n_bins=10)
jmim_result = jmim_filter.calculate(task)

print("\nCaracterísticas seleccionadas por JMIM:")
print("(Considera redundancia entre características)")
print("-"*40)
selected_features = jmim_result.features[:10]
for i, feat in enumerate(selected_features):
    score = jmim_result.scores[feat]
    print(f"{i+1:2d}. {feat:20s}: {score:.4f}")

# Análisis: Ver qué tipo de características selecciona cada método
print("\n" + "="*60)
print("5. ANÁLISIS: ¿Qué características prefiere cada método?")
print("="*60)

# Contar tipos de características en top 10
methods = {
    'IG': ig_result,
    'IGR': igr_result,
    'SU': su_result,
    'JMIM': jmim_result
}

feature_types = {
    'informative': ['info_' in f for f in feature_names],
    'redundant': ['redun_' in f for f in feature_names],
    'high_entropy': ['high_entropy' in f for f in feature_names],
    'noise': ['noise_' in f for f in feature_names]
}

print("\nDistribución de tipos en top 10 características:")
print("-"*60)
print("Método | Informativas | Redundantes | Alta Entropía | Ruido")
print("-"*60)

for method_name, result in methods.items():
    top_10 = result.features[:10]
    counts = {ftype: 0 for ftype in feature_types}
    
    for feat in top_10:
        idx = feature_names.index(feat)
        for ftype, mask in feature_types.items():
            if mask[idx]:
                counts[ftype] += 1
                
    print(f"{method_name:6s} | {counts['informative']:12d} | {counts['redundant']:11d} | "
          f"{counts['high_entropy']:13d} | {counts['noise']:5d}")

# Visualización
print("\n" + "="*60)
print("6. VISUALIZACIÓN")
print("="*60)

# Crear gráfico comparativo
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Preparar datos para visualización
all_features = feature_names
n_show = min(20, len(all_features))

for ax, (method_name, result) in zip(axes.flat, methods.items()):
    # Obtener scores para todas las características
    scores = [result.scores.get(f, 0) for f in all_features[:n_show]]
    
    # Colores por tipo de característica
    colors = []
    for feat in all_features[:n_show]:
        if 'info_' in feat:
            colors.append('green')
        elif 'redun_' in feat:
            colors.append('orange')
        elif 'high_entropy' in feat:
            colors.append('red')
        else:  # noise
            colors.append('gray')
    
    # Crear gráfico de barras
    y_pos = np.arange(n_show)
    ax.barh(y_pos, scores, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_features[:n_show], fontsize=9)
    ax.set_xlabel('Score')
    ax.set_title(f'{method_name} Scores')
    ax.grid(axis='x', alpha=0.3)
    
    # Añadir línea vertical en el percentil 75
    threshold = np.percentile(scores, 75)
    ax.axvline(threshold, color='red', linestyle='--', alpha=0.5, label=f'75th percentile')

# Leyenda
handles = [
    plt.Rectangle((0,0),1,1, color='green', alpha=0.7, label='Informativas'),
    plt.Rectangle((0,0),1,1, color='orange', alpha=0.7, label='Redundantes'),
    plt.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Alta Entropía'),
    plt.Rectangle((0,0),1,1, color='gray', alpha=0.7, label='Ruido')
]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)

plt.tight_layout()
plt.savefig('information_theory_filters_comparison.png', dpi=150, bbox_inches='tight')
print("\nGráfico guardado como 'information_theory_filters_comparison.png'")

# Ejemplo de uso en pipeline
print("\n" + "="*60)
print("7. USO EN PIPELINES")
print("="*60)

from mlpy.pipelines import PipeOpFilter, linear_pipeline, GraphLearner, PipeOpLearner
from mlpy.learners import learner_sklearn
from sklearn.naive_bayes import GaussianNB
from mlpy import resample
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy

# Pipeline con Information Gain Ratio
pipeline_igr = linear_pipeline(
    PipeOpFilter(method='info_gain_ratio', k=10),
    PipeOpLearner(learner_sklearn(GaussianNB()))
)

# Pipeline con JMIM
pipeline_jmim = linear_pipeline(
    PipeOpFilter(
        method='jmim',
        filter_params={'n_features': 10}
    ),
    PipeOpLearner(learner_sklearn(GaussianNB()))
)

# Evaluar
print("\nEvaluando pipelines con diferentes filtros...")

for pipeline, name in [(pipeline_igr, "IGR + NB"), (pipeline_jmim, "JMIM + NB")]:
    learner = GraphLearner(pipeline, id=name)
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy()
    )
    
    scores = result.score()
    print(f"{name:15s}: Accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

# Resumen
print("\n" + "="*60)
print("RESUMEN: FILTROS DE TEORÍA DE LA INFORMACIÓN")
print("="*60)
print("""
1. INFORMATION GAIN (IG):
   - Mide reducción en entropía del target
   - Sesgo hacia características con muchos valores
   - Bueno para árboles de decisión simples

2. INFORMATION GAIN RATIO (IGR):
   - Normaliza IG por entropía de la característica
   - Corrige sesgo de IG
   - Usado en C4.5

3. SYMMETRICAL UNCERTAINTY (SU):
   - Información mutua normalizada
   - Valores en [0, 1]
   - Simétrico: SU(X,Y) = SU(Y,X)

4. JMIM:
   - Considera redundancia entre características
   - Selección conjunta óptima
   - Más costoso computacionalmente

CUÁNDO USAR CADA UNO:
- IG: Árboles simples, interpretabilidad
- IGR: Cuando hay características con muchos valores
- SU: Comparación entre características
- JMIM: Cuando la redundancia es importante
""")

plt.show()