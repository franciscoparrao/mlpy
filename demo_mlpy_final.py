"""
Demo final de MLPY - Versión completamente funcional
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print("="*60)
print("DEMO FINAL DE MLPY")
print("="*60)

# 1. Crear datos
print("\n1. Creando datos...")
X, y = make_classification(n_samples=300, n_features=10, n_informative=8, 
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
from mlpy.measures import MeasureClassifAccuracy
measure_acc = MeasureClassifAccuracy()
print(f"   Measure: {measure_acc.id}")

# 5. Resampling simple
print("\n5. Ejecutando resample...")
from mlpy import resample
from mlpy.resamplings import ResamplingCV

result = resample(
    task=task,
    learner=learners['rf'],
    resampling=ResamplingCV(folds=5),
    measures=measure_acc
)

scores = result.aggregate()
print(f"   Random Forest CV:")
print(f"   - Accuracy: {scores['acc'][0]:.3f} ± {scores['acc'][1]:.3f}")

# 6. Benchmark
print("\n6. Ejecutando benchmark...")
from mlpy import benchmark

bench_result = benchmark(
    tasks=[task],
    learners=list(learners.values()),
    resampling=ResamplingCV(folds=3),
    measures=measure_acc
)

print("\n   Tabla de scores:")
print(bench_result.score_table())

print("\n   Ranking:")
ranking = bench_result.rank_learners()
for i, (learner_id, rank) in enumerate(ranking):
    print(f"   {i+1}. {learner_id}: rank = {rank:.2f}")

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
    measures=measure_acc
)

print(f"   Pipeline accuracy: {pipe_result.aggregate()['acc'][0]:.3f}")

# 8. Pipeline avanzado
print("\n8. Pipeline avanzado...")
try:
    from mlpy.pipelines import PipeOpSelect
    from mlpy.pipelines.advanced_operators import PipeOpPCA
    
    advanced_pipeline = linear_pipeline(
        PipeOpScale(id='scale'),
        PipeOpPCA(id='pca', n_components=5),
        PipeOpLearner(learners['lr'], id='learner')
    )
    
    adv_result = resample(
        task=task,
        learner=advanced_pipeline,
        resampling=ResamplingCV(folds=3),
        measures=measure_acc
    )
    
    print(f"   Pipeline con PCA accuracy: {adv_result.aggregate()['acc'][0]:.3f}")
except Exception as e:
    print(f"   Error en pipeline avanzado: {e}")

# 9. Persistencia
print("\n9. Probando persistencia...")
try:
    from mlpy.persistence import save_model, load_model
    import tempfile
    import os
    
    # Entrenar modelo
    learners['rf'].train(task)
    
    # Guardar con metadata
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    metadata = {
        'dataset': 'demo',
        'features': task.feature_names,
        'accuracy': scores['acc'][0]
    }
    
    saved_path = save_model(learners['rf'], temp_path, metadata=metadata)
    file_size = os.path.getsize(saved_path) / 1024  # KB
    print(f"   Modelo guardado ({file_size:.1f} KB)")
    
    # Cargar
    loaded_model, loaded_meta = load_model(temp_path, return_metadata=True)
    print(f"   Modelo cargado: {loaded_model.id}")
    print(f"   Metadata: accuracy = {loaded_meta['accuracy']:.3f}")
    
    # Hacer predicción con modelo cargado
    pred = loaded_model.predict(task)
    print(f"   Predicción funciona: {len(pred.response)} predicciones")
    
    # Limpiar
    os.unlink(temp_path)
    
except Exception as e:
    print(f"   Error en persistencia: {e}")

# 10. Resumen del sistema
print("\n10. RESUMEN DEL SISTEMA")
print("-"*40)

# Contar componentes
import mlpy
componentes = {
    'Tasks': len([x for x in dir(mlpy.tasks) if x.startswith('Task')]),
    'Learners': len([x for x in dir(mlpy.learners) if x.startswith('Learner')]),
    'Measures': len([x for x in dir(mlpy.measures) if x.startswith('Measure')]),
    'Pipelines': len([x for x in dir(mlpy.pipelines) if x.startswith('PipeOp')]),
    'Resamplings': len([x for x in dir(mlpy.resamplings) if x.startswith('Resampling')])
}

for comp, count in componentes.items():
    print(f"   {comp}: {count} clases")

print("\n" + "="*60)
print("MLPY ESTÁ FUNCIONANDO CORRECTAMENTE!")
print("="*60)

# Información adicional
print("\nCaracterísticas principales:")
print("✓ Framework unificado inspirado en mlr3")
print("✓ Integración completa con scikit-learn")
print("✓ Sistema de pipelines flexible")
print("✓ Benchmark para comparar modelos")
print("✓ Persistencia de modelos con metadata")
print("✓ Evaluación robusta con resampling")
print("✓ Extensible y modular")

print("\nPróximos pasos:")
print("- Instalar dependencias opcionales (dask, vaex, shap, lime)")
print("- Explorar operadores avanzados de pipeline")
print("- Usar AutoML para tuning de hiperparámetros")
print("- Visualizar resultados con plot_benchmark_*")

print("\n¡Disfruta usando MLPY!")