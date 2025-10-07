"""
Demo funcional de MLPY - Versión corregida
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print("="*60)
print("DEMO FUNCIONAL DE MLPY")
print("="*60)

# 1. Crear datos
print("\n1. Creando datos...")
X, y = make_classification(n_samples=200, n_features=10, n_informative=8, 
                          n_redundant=2, random_state=42)
df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
df['target'] = y

# 2. Crear Task
print("\n2. Creando Task...")
from mlpy.tasks import TaskClassif
task = TaskClassif(data=df, target='target', id='demo')
print(f"   Task: {task.nrow} filas, {task.ncol-1} features")

# 3. Crear Learners
print("\n3. Creando Learners...")
from mlpy.learners import learner_sklearn

learners = {
    'rf': learner_sklearn(RandomForestClassifier(n_estimators=50, random_state=42), id='rf'),
    'lr': learner_sklearn(LogisticRegression(max_iter=1000), id='lr'),
    'dt': learner_sklearn(DecisionTreeClassifier(max_depth=5), id='dt')
}
print(f"   Learners: {list(learners.keys())}")

# 4. Definir Measures
print("\n4. Definiendo Measures...")
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifAUC
measures = [MeasureClassifAccuracy(), MeasureClassifAUC()]
print(f"   Measures: {[m.id for m in measures]}")

# 5. Resampling simple
print("\n5. Ejecutando resample...")
from mlpy import resample
from mlpy.resamplings import ResamplingCV

result = resample(
    task=task,
    learner=learners['rf'],
    resampling=ResamplingCV(folds=5),
    measures=measures  # Nota: es 'measures' no 'measure'
)

scores = result.aggregate()
print(f"   Random Forest CV:")
print(f"   - Accuracy: {scores['classif.acc'][0]:.3f} ± {scores['classif.acc'][1]:.3f}")
print(f"   - AUC: {scores['classif.auc'][0]:.3f} ± {scores['classif.auc'][1]:.3f}")

# 6. Benchmark
print("\n6. Ejecutando benchmark...")
from mlpy import benchmark

bench_result = benchmark(
    tasks=[task],
    learners=list(learners.values()),
    resampling=ResamplingCV(folds=3),
    measures=measures
)

print("\n   Resultados:")
print(bench_result.score_table())

# 7. Pipeline
print("\n7. Creando pipeline...")
from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline

pipeline = linear_pipeline(
    PipeOpScale(id='scale'),
    PipeOpLearner(learners['rf'], id='learner')
)

pipe_result = resample(
    task=task,
    learner=pipeline,
    resampling=ResamplingCV(folds=3),
    measures=[measures[0]]
)

print(f"   Pipeline accuracy: {pipe_result.aggregate()['classif.acc'][0]:.3f}")

# 8. Persistencia
print("\n8. Probando persistencia...")
try:
    from mlpy.persistence import save_model, load_model
    import tempfile
    import os
    
    # Entrenar modelo
    learners['rf'].train(task)
    
    # Guardar
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    save_model(learners['rf'], temp_path)
    print(f"   Modelo guardado en: {temp_path}")
    
    # Cargar
    loaded = load_model(temp_path)
    print(f"   Modelo cargado: {loaded.id}")
    
    # Limpiar
    os.unlink(temp_path)
    print("   Persistencia OK")
    
except Exception as e:
    print(f"   Error en persistencia: {e}")

print("\n" + "="*60)
print("MLPY está funcionando correctamente!")
print("="*60)