"""
Debug simple y rápido de problemas H2O.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif
from mlpy.learners.h2o_wrapper import learner_h2o
from mlpy.measures import MeasureClassifAccuracy

print("="*60)
print("DEBUG SIMPLE H2O")
print("="*60)

# Problema principal: valores nan en el benchmark
# Hipótesis: El problema está en cómo se agregan los resultados del benchmark

# Crear dataset multiclase pequeño
X, y = make_classification(n_samples=100, n_features=5, n_classes=3, 
                          n_informative=3, n_redundant=1, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df['target'] = y
task = TaskClassif(df, 'target')

print(f"\nDataset: {task.nrow} muestras, {task.n_classes} clases")

# Inicializar H2O de forma simple
import h2o
from h2o.estimators import H2ORandomForestEstimator

print("\nInicializando H2O...")
h2o.init(verbose=False, nthreads=1)
h2o.no_progress()

# Test directo sin benchmark
print("\nTest 1: Entrenamiento y predicción directa")
print("-"*40)

try:
    # Crear y entrenar modelo
    rf = H2ORandomForestEstimator(ntrees=5, seed=42)
    learner = learner_h2o(rf, id="H2O_RF_test")
    
    print("Entrenando...")
    learner.train(task)
    
    print("Prediciendo...")
    pred = learner.predict(task)
    
    # Verificar resultados
    print(f"Tipo de tarea: {learner._task_type}")
    print(f"Shape respuesta: {pred.response.shape}")
    print(f"Primeras 5 predicciones: {pred.response[:5]}")
    print(f"Primeras 5 verdades: {pred.truth[:5]}")
    
    # Calcular accuracy
    acc = np.mean(pred.response == pred.truth)
    print(f"Accuracy: {acc:.3f}")
    
    # Probar medida
    measure = MeasureClassifAccuracy()
    score = measure.score(pred)
    print(f"Score con medida: {score:.3f}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test con resample simple
print("\n\nTest 2: Con resample (2 folds)")
print("-"*40)

try:
    from mlpy import resample
    from mlpy.resamplings import ResamplingCV
    
    # Nuevo learner
    learner2 = learner_h2o(H2ORandomForestEstimator(ntrees=5, seed=42), id="H2O_RF_cv")
    
    result = resample(
        task=task,
        learner=learner2,
        resampling=ResamplingCV(folds=2),
        measures=MeasureClassifAccuracy()
    )
    
    # Ver resultados crudos
    print("Resultados del resample:")
    print(result.scores)
    
    # Aggregate
    agg = result.aggregate()
    print("\nResultados agregados:")
    print(agg)
    
except Exception as e:
    print(f"ERROR en resample: {e}")
    import traceback
    traceback.print_exc()

# Test específico del problema en benchmark
print("\n\nTest 3: Simulando benchmark")
print("-"*40)

try:
    from mlpy import benchmark
    
    # Solo 1 learner, 1 tarea, 2 folds
    learners = [learner_h2o(H2ORandomForestEstimator(ntrees=5, seed=42), id="H2O_test")]
    
    bench_result = benchmark(
        tasks=[task],
        learners=learners,
        resampling=ResamplingCV(folds=2),
        measures=[MeasureClassifAccuracy()]
    )
    
    print("Resultado benchmark:")
    print(bench_result.scores)
    
    print("\nAgregado:")
    agg_bench = bench_result.aggregate("classif.acc")
    print(agg_bench)
    
except Exception as e:
    print(f"ERROR en benchmark: {e}")
    import traceback
    traceback.print_exc()

# Limpiar
print("\nCerrando H2O...")
h2o.cluster().shutdown(prompt=False)

print("\n" + "="*60)
print("Debug completado")