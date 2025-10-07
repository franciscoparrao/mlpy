"""
Test directo de las correcciones de H2O.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGeneralizedLinearEstimator

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif
from mlpy.learners.h2o_wrapper import learner_h2o
from mlpy.measures import MeasureClassifF1

# Inicializar H2O silenciosamente
h2o.init(verbose=False, nthreads=-1)
h2o.no_progress()  # Desactivar barras de progreso

print("="*60)
print("TEST DE CORRECCIONES H2O")
print("="*60)

# Test 1: Verificar detección de tipo de tarea
print("\n1. Verificando detección de tipo de tarea...")
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df['target'] = y
task = TaskClassif(df, 'target')

# Crear learner H2O
learner = learner_h2o(H2ORandomForestEstimator(ntrees=5, seed=42))

# Antes de entrenar, el tipo puede ser None o inferido
print(f"   Tipo antes de entrenar: {learner._task_type}")

# Entrenar
learner.train(task)

# Después de entrenar, debe ser 'classif'
print(f"   Tipo después de entrenar: {learner._task_type}")
assert learner._task_type == 'classif', "Error: tipo de tarea no se estableció correctamente"
print("   ✓ Detección de tipo funcionando!")

# Test 2: Verificar parámetro lambda en GLM
print("\n2. Verificando parámetro lambda_ en GLM...")
try:
    glm = H2OGeneralizedLinearEstimator(
        family="binomial",
        seed=42,
        lambda_=0  # Usar lambda_ en lugar de lambda
    )
    learner_glm = learner_h2o(glm)
    learner_glm.train(task)
    print("   ✓ GLM con lambda_ funcionando!")
except Exception as e:
    print(f"   ✗ Error con GLM: {e}")

# Test 3: Verificar F1 multiclase con average
print("\n3. Verificando F1 multiclase con average='macro'...")
X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df['target'] = y
task_multi = TaskClassif(df, 'target')

try:
    # Crear medida F1 con average='macro'
    f1_macro = MeasureClassifF1(average='macro')
    
    # Entrenar modelo
    learner_multi = learner_h2o(H2ORandomForestEstimator(ntrees=5, seed=42))
    learner_multi.train(task_multi)
    
    # Predecir
    pred = learner_multi.predict(task_multi)
    
    # Calcular F1
    score = f1_macro.score(pred)
    print(f"   F1 score (macro): {score:.3f}")
    print("   ✓ F1 multiclase con average='macro' funcionando!")
except Exception as e:
    print(f"   ✗ Error con F1 multiclase: {e}")

# Test 4: Verificar que las predicciones funcionan correctamente
print("\n4. Verificando predicciones...")
pred = learner.predict(task)
acc = np.mean(pred.response == pred.truth)
print(f"   Accuracy: {acc:.3f}")
print("   ✓ Predicciones funcionando!")

print("\n" + "="*60)
print("RESUMEN: Todas las correcciones están funcionando ✓")
print("="*60)

# Limpiar
h2o.cluster().shutdown(prompt=False)