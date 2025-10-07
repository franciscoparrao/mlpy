"""
Diagnóstico de problemas específicos con H2O.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import h2o
from h2o.estimators import (
    H2ORandomForestEstimator,
    H2OGradientBoostingEstimator,
    H2ODeepLearningEstimator,
    H2OGeneralizedLinearEstimator
)
import traceback

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners.h2o_wrapper import learner_h2o
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1, MeasureRegrR2

# Inicializar H2O
print("Inicializando H2O...")
h2o.init(verbose=False, nthreads=-1)
h2o.no_progress()

print("="*60)
print("DIAGNÓSTICO DE PROBLEMAS H2O")
print("="*60)

# Problema 1: H2O modelos no funcionan en multiclase
print("\n1. INVESTIGANDO PROBLEMA MULTICLASE")
print("-"*40)

# Crear dataset multiclase
X, y = make_classification(n_samples=300, n_features=10, n_classes=4, 
                          n_informative=8, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
df['target'] = y
task_multi = TaskClassif(df, 'target')

# Probar cada modelo H2O en multiclase
h2o_models = [
    ("H2O_RF", H2ORandomForestEstimator(ntrees=10, seed=42)),
    ("H2O_GBM", H2OGradientBoostingEstimator(ntrees=10, seed=42)),
    ("H2O_DL", H2ODeepLearningEstimator(hidden=[20,20], epochs=5, seed=42)),
    ("H2O_GLM_multi", H2OGeneralizedLinearEstimator(family="multinomial", seed=42, lambda_=0))
]

for name, model in h2o_models:
    print(f"\nProbando {name}...")
    try:
        learner = learner_h2o(model, id=name)
        learner.train(task_multi)
        pred = learner.predict(task_multi)
        
        # Verificar predicciones
        print(f"  - Tipo de tarea detectado: {learner._task_type}")
        print(f"  - Shape de predicciones: {pred.response.shape}")
        print(f"  - Clases únicas en respuesta: {np.unique(pred.response)}")
        print(f"  - Clases únicas en verdad: {np.unique(pred.truth)}")
        
        # Calcular accuracy
        acc = np.mean(pred.response == pred.truth)
        print(f"  - Accuracy: {acc:.3f}")
        
        # Probar F1
        f1_measure = MeasureClassifF1(average='macro')
        f1_score = f1_measure.score(pred)
        print(f"  - F1 Score (macro): {f1_score:.3f}")
        
        print(f"  ✓ {name} funcionó correctamente")
        
    except Exception as e:
        print(f"  ✗ ERROR en {name}: {str(e)}")
        traceback.print_exc()

# Problema 2: H2O GLM en regresión
print("\n\n2. INVESTIGANDO H2O GLM EN REGRESIÓN")
print("-"*40)

X, y = make_regression(n_samples=300, n_features=10, noise=10, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
df['y'] = y
task_regr = TaskRegr(df, 'y')

try:
    glm_regr = H2OGeneralizedLinearEstimator(
        family="gaussian",
        seed=42,
        lambda_=0.0,  # Probar con 0.0 explícito
        alpha=0.5     # Añadir alpha parameter
    )
    learner_glm = learner_h2o(glm_regr, id="H2O_GLM_regr")
    
    print("Entrenando GLM para regresión...")
    learner_glm.train(task_regr)
    
    print("Haciendo predicciones...")
    pred = learner_glm.predict(task_regr)
    
    # Verificar predicciones
    print(f"  - Tipo de tarea: {learner_glm._task_type}")
    print(f"  - Shape de predicciones: {pred.response.shape}")
    print(f"  - Primeras 5 predicciones: {pred.response[:5]}")
    print(f"  - Primeros 5 valores reales: {pred.truth[:5]}")
    
    # RMSE manual
    rmse = np.sqrt(np.mean((pred.response - pred.truth)**2))
    print(f"  - RMSE: {rmse:.3f}")
    
    print("  ✓ GLM regresión funcionó")
    
except Exception as e:
    print(f"  ✗ ERROR en GLM regresión: {str(e)}")
    traceback.print_exc()

# Problema 3: Métricas R² que devuelven nan
print("\n\n3. INVESTIGANDO PROBLEMA CON R²")
print("-"*40)

# Probar R² con diferentes modelos
models_to_test = [
    ("H2O_RF", H2ORandomForestEstimator(ntrees=10, seed=42)),
    ("H2O_GBM", H2OGradientBoostingEstimator(ntrees=10, seed=42))
]

r2_measure = MeasureRegrR2()

for name, model in models_to_test:
    print(f"\nProbando R² con {name}...")
    try:
        learner = learner_h2o(model, id=name)
        learner.train(task_regr)
        pred = learner.predict(task_regr)
        
        # Calcular R² manualmente
        ss_res = np.sum((pred.truth - pred.response) ** 2)
        ss_tot = np.sum((pred.truth - np.mean(pred.truth)) ** 2)
        r2_manual = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print(f"  - R² manual: {r2_manual:.3f}")
        
        # Usar medida MLPY
        r2_mlpy = r2_measure.score(pred)
        print(f"  - R² MLPY: {r2_mlpy:.3f}")
        
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")

# Problema 4: Verificar configuración de H2O Deep Learning
print("\n\n4. VERIFICANDO H2O DEEP LEARNING")
print("-"*40)

# Dataset binario simple
X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df['target'] = y
task_binary = TaskClassif(df, 'target')

try:
    # Configuración más conservadora para DL
    dl = H2ODeepLearningEstimator(
        hidden=[10, 10],          # Red más pequeña
        epochs=10,                # Más épocas
        seed=42,
        reproducible=True,
        activation="Tanh",        # Activación más estable
        loss="CrossEntropy",      # Loss explícito
        distribution="bernoulli"  # Para binario
    )
    
    learner_dl = learner_h2o(dl, id="H2O_DL_test")
    learner_dl.train(task_binary)
    pred = learner_dl.predict(task_binary)
    
    acc = np.mean(pred.response == pred.truth)
    print(f"  - Accuracy: {acc:.3f}")
    print("  ✓ Deep Learning funcionó con configuración ajustada")
    
except Exception as e:
    print(f"  ✗ ERROR en Deep Learning: {str(e)}")
    traceback.print_exc()

# Resumen de hallazgos
print("\n\n" + "="*60)
print("RESUMEN DE HALLAZGOS")
print("="*60)

# Limpiar
h2o.cluster().shutdown(prompt=False)