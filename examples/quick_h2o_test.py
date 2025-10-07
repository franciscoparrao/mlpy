"""
Test rápido para identificar el problema con H2O.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif
from mlpy.measures import MeasureClassifAccuracy
from mlpy import resample
from mlpy.resamplings import ResamplingCV

print("="*60)
print("TEST RÁPIDO H2O - Sin inicializar H2O directamente")
print("="*60)

# Crear dataset multiclase simple
X, y = make_classification(n_samples=150, n_features=5, n_classes=3, 
                          n_informative=4, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df['target'] = y
task = TaskClassif(df, 'target')

print(f"\nDataset: {task.nrow} muestras, {len(task.feature_names)} features, {task.n_classes} clases")

# Test 1: Verificar si el problema es con el wrapper o con H2O
print("\n1. Probando sklearn primero (control)...")
from mlpy.learners import learner_sklearn
from sklearn.ensemble import RandomForestClassifier

rf_sklearn = learner_sklearn(RandomForestClassifier(n_estimators=10, random_state=42))
result = resample(
    task=task,
    learner=rf_sklearn,
    resampling=ResamplingCV(folds=2),
    measures=MeasureClassifAccuracy()
)
print(f"   sklearn RF Accuracy: {result.score(measures='classif.acc').mean():.3f}")

# Test 2: Ahora probar H2O
print("\n2. Probando H2O...")
try:
    import h2o
    from h2o.estimators import H2ORandomForestEstimator
    from mlpy.learners.h2o_wrapper import learner_h2o
    
    # Inicializar H2O de forma mínima
    h2o.init(verbose=False, nthreads=1, max_mem_size="1G")
    h2o.no_progress()
    
    # Crear learner H2O
    h2o_rf = learner_h2o(H2ORandomForestEstimator(ntrees=10, seed=42))
    
    # Entrenar directamente
    print("   Entrenando H2O RF...")
    h2o_rf.train(task)
    
    # Predecir
    pred = h2o_rf.predict(task)
    acc = np.mean(pred.response == pred.truth)
    print(f"   H2O RF Accuracy (directo): {acc:.3f}")
    
    # Ahora con resample
    print("   Probando con resample...")
    result_h2o = resample(
        task=task,
        learner=h2o_rf.clone(),  # Clonar para empezar limpio
        resampling=ResamplingCV(folds=2),
        measures=MeasureClassifAccuracy()
    )
    print(f"   H2O RF Accuracy (CV): {result_h2o.score(measures='classif.acc').mean():.3f}")
    
    # Cerrar H2O
    h2o.cluster().shutdown(prompt=False)
    
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)