"""
Demostración simple de MLPY
===========================
"""

import numpy as np
import pandas as pd

print("=== MLPY Demo ===\n")

# 1. Crear datos sintéticos
print("1. Generando datos sintéticos...")
np.random.seed(42)
n_samples = 1000

# Generar características
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 2, n_samples)
X3 = np.random.choice(['A', 'B', 'C'], n_samples)

# Generar target basado en reglas simples
y_numeric = 2 * X1 - 0.5 * X2 + np.random.normal(0, 0.5, n_samples)
y = ['Positivo' if val > 0 else 'Negativo' for val in y_numeric]

# Crear DataFrame
data = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'categoria': X3,
    'target': y
})

print(f"- Muestras: {n_samples}")
print(f"- Características: 3 (2 numéricas, 1 categórica)")
print(f"- Distribución del target:")
print(data['target'].value_counts())

# 2. Importar MLPY
print("\n2. Cargando MLPY...")
try:
    from mlpy.tasks import TaskClassif
    from mlpy.learners import LearnerClassifFeatureless
    from mlpy.measures import MeasureClassifAccuracy
    from mlpy.resamplings import ResamplingCV
    from mlpy import resample
    
    print("+ MLPY cargado correctamente")
    
    # 3. Crear tarea
    print("\n3. Creando tarea de clasificación...")
    task = TaskClassif(data=data, target='target', id='demo_task')
    print(f"+ Tarea creada: {task.id}")
    print(f"  - Tipo: {task.task_type}")
    print(f"  - Clases: {task.class_names}")
    print(f"  - Número de características: {len(task.feature_names)}")
    print(f"  - Características: {task.feature_names}")
    
    # 4. Crear learner
    print("\n4. Creando modelo baseline...")
    learner = LearnerClassifFeatureless(id='baseline', method='mode')
    print(f"+ Learner creado: {learner.id}")
    
    # 5. Entrenar modelo
    print("\n5. Entrenando modelo...")
    learner.train(task)
    print("+ Modelo entrenado")
    
    # 6. Hacer predicciones
    print("\n6. Realizando predicciones...")
    predictions = learner.predict(task)
    print(f"+ Predicciones generadas")
    print(f"  - Tipo: {type(predictions).__name__}")
    print(f"  - Primeras 5 predicciones: {predictions.response[:5].tolist()}")
    
    # 7. Evaluar con cross-validation
    print("\n7. Evaluando con cross-validation...")
    
    result = resample(
        task=task,
        learner=learner,
        resampling=ResamplingCV(folds=5),
        measures=MeasureClassifAccuracy()
    )
    
    scores = result.aggregate()
    print(f"+ Evaluación completada")
    print(f"  - Resultados: {scores}")
    
    # Obtener las métricas correctamente
    if isinstance(scores, dict):
        for measure_id, values in scores.items():
            print(f"  - {measure_id}: mean={values['mean']:.3f}, sd={values['sd']:.3f}")
    else:
        # Si es un DataFrame
        print(f"  - Accuracy promedio: {scores['mean'].iloc[0]:.3f}")
        print(f"  - Desviación estándar: {scores['std'].iloc[0]:.3f}")
    
    # 8. Resumen
    print("\n" + "="*50)
    print("RESUMEN DE RESULTADOS")
    print("="*50)
    print(f"Dataset: {n_samples} muestras, 3 características")
    print(f"Modelo: {learner.id} (predice siempre la clase mayoritaria)")
    
    # Obtener el accuracy correcto
    if isinstance(scores, dict):
        acc = list(scores.values())[0]['mean']
    else:
        acc = scores['mean'].iloc[0]
    
    print(f"Accuracy: {acc:.1%}")
    print("\nInterpretación:")
    print(f"- El modelo baseline obtiene {acc:.1%} de accuracy")
    print(f"- Esto corresponde a la proporción de la clase mayoritaria")
    print(f"- Es el punto de partida para comparar modelos más complejos")
    
    print("\n+ Demo completada exitosamente!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()