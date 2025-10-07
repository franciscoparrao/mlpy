"""
Demostración de MLPY con modelos de scikit-learn
================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime

print("=== MLPY Demo con Modelos Reales ===\n")
print(f"Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 1. Generar dataset más realista
print("1. GENERANDO DATASET")
print("-" * 40)

np.random.seed(42)
n_samples = 2000

# Simular datos de clientes para predicción de churn
data = pd.DataFrame({
    # Características numéricas
    'edad': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
    'ingresos': np.random.lognormal(10.5, 0.6, n_samples),
    'meses_cliente': np.random.exponential(24, n_samples).clip(1, 120).astype(int),
    'facturas_mensuales': np.random.gamma(2, 50, n_samples),
    'llamadas_servicio': np.random.poisson(2, n_samples),
    'pagos_atrasados': np.random.poisson(0.5, n_samples),
    
    # Características categóricas
    'tipo_contrato': np.random.choice(['Mensual', 'Anual', 'Bianual'], 
                                     n_samples, p=[0.5, 0.3, 0.2]),
    'metodo_pago': np.random.choice(['Electrónico', 'Cheque', 'Transferencia'], 
                                   n_samples),
    
    # Características binarias
    'factura_digital': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'tiene_pareja': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
})

# Crear variable objetivo con relaciones complejas
churn_score = (
    0.3 * (data['tipo_contrato'] == 'Mensual') +
    0.2 * (data['llamadas_servicio'] > 3) +
    0.15 * (data['pagos_atrasados'] > 1) +
    0.15 * (data['facturas_mensuales'] > data['facturas_mensuales'].quantile(0.75)) +
    0.1 * (data['meses_cliente'] < 12) +
    0.05 * (data['factura_digital'] == 0) +
    np.random.normal(0, 0.2, n_samples)
)

data['churn'] = (churn_score > np.percentile(churn_score, 70)).astype(str)
data['churn'] = data['churn'].map({'True': 'Si', 'False': 'No'})

print(f"Dataset creado: {data.shape}")
print(f"\nDistribución del objetivo:")
print(data['churn'].value_counts())
print(f"Tasa de churn: {(data['churn'] == 'Si').mean():.1%}")

# 2. Importar MLPY y crear tarea
print("\n2. CONFIGURANDO MLPY")
print("-" * 40)

try:
    from mlpy.tasks import TaskClassif
    from mlpy.measures import (
        MeasureClassifAccuracy, MeasureClassifF1, 
        MeasureClassifPrecision, MeasureClassifRecall
    )
    from mlpy.resamplings import ResamplingCV, ResamplingHoldout
    from mlpy import resample, benchmark
    
    # Importar learners
    from mlpy.learners import (
        LearnerClassifFeatureless,
        LearnerLogisticRegression,
        LearnerRandomForest,
        LearnerGradientBoosting,
        LearnerSVM,
        LearnerKNN
    )
    
    # Importar pipeline
    from mlpy.pipelines import (
        PipeOpImpute, PipeOpScale, PipeOpEncode,
        PipeOpLearner, linear_pipeline
    )
    
    print("+ MLPY y modelos cargados correctamente")
    
except ImportError as e:
    print(f"Error al importar: {e}")
    print("Verificando si scikit-learn está instalado...")
    
    try:
        import sklearn
        print(f"scikit-learn versión: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn no está instalado. Instalando...")
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-learn"])
        print("Por favor, ejecuta el script de nuevo.")
        exit(1)

# 3. Crear tarea
print("\n3. CREANDO TAREA DE CLASIFICACIÓN")
print("-" * 40)

task = TaskClassif(
    data=data,
    target='churn',
    id='prediccion_churn'
)

print(f"Tarea: {task.id}")
print(f"Tipo: {task.task_type}")
print(f"Clases: {task.class_names}")
print(f"Características: {len(task.feature_names)}")

# 4. Crear pipelines para cada modelo
print("\n4. CREANDO MODELOS Y PIPELINES")
print("-" * 40)

# Definir modelos a comparar
modelos = {
    'baseline': LearnerClassifFeatureless(
        id='baseline',
        method='mode'
    ),
    'logistica': LearnerLogisticRegression(
        id='logistica',
        C=1.0,
        max_iter=1000
    ),
    'random_forest': LearnerRandomForest(
        id='random_forest',
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    'gradient_boosting': LearnerGradientBoosting(
        id='gradient_boosting',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    'svm': LearnerSVM(
        id='svm',
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    ),
    'knn': LearnerKNN(
        id='knn',
        n_neighbors=10,
        weights='distance'
    )
}

# Crear pipelines con preprocesamiento
pipelines = {}
for nombre, modelo in modelos.items():
    pipeline = linear_pipeline(
        PipeOpImpute(id=f'impute_num_{nombre}', method='median', affect_columns='numeric'),
        PipeOpImpute(id=f'impute_cat_{nombre}', method='most_frequent', affect_columns='factor'),
        PipeOpEncode(id=f'encode_{nombre}', method='onehot'),
        PipeOpScale(id=f'scale_{nombre}', method='standard'),
        PipeOpLearner(modelo, id=f'learner_{nombre}')
    )
    pipelines[nombre] = pipeline
    print(f"+ Pipeline creado: {nombre}")

# 5. Evaluar modelo baseline
print("\n5. EVALUACIÓN INICIAL (BASELINE)")
print("-" * 40)

print("Evaluando modelo baseline con CV...")
baseline_result = resample(
    task=task,
    learner=pipelines['baseline'],
    resampling=ResamplingCV(folds=5, stratify=True),
    measures=[
        MeasureClassifAccuracy(),
        MeasureClassifF1(),
        MeasureClassifPrecision(),
        MeasureClassifRecall()
    ]
)

baseline_scores = baseline_result.aggregate()
print("\nResultados Baseline:")
print(baseline_scores)

# 6. Comparar todos los modelos
print("\n6. COMPARACIÓN DE MODELOS")
print("-" * 40)

print("Ejecutando benchmark (esto puede tomar unos minutos)...")
benchmark_result = benchmark(
    tasks=[task],
    learners=list(pipelines.values()),
    resampling=ResamplingCV(folds=3, stratify=True),
    measures=[
        MeasureClassifAccuracy(),
        MeasureClassifF1()
    ]
)

print("\nResultados del Benchmark:")
print(benchmark_result)

# 7. Evaluación detallada del mejor modelo
print("\n7. EVALUACIÓN DETALLADA DEL MEJOR MODELO")
print("-" * 40)

# Obtener el mejor modelo basado en F1
mejores_por_f1 = benchmark_result.rank_learners('classif.f1')
mejor_modelo_id = mejores_por_f1.iloc[0]['learner']
mejor_modelo_nombre = mejor_modelo_id.replace('pipeline_', '')

print(f"Mejor modelo según F1: {mejor_modelo_nombre}")

# Evaluar en holdout
print(f"\nEvaluando {mejor_modelo_nombre} en conjunto de prueba...")
final_result = resample(
    task=task,
    learner=pipelines[mejor_modelo_nombre],
    resampling=ResamplingHoldout(ratio=0.8, stratify=True),
    measures=[
        MeasureClassifAccuracy(),
        MeasureClassifF1(),
        MeasureClassifPrecision(),
        MeasureClassifRecall()
    ]
)

final_scores = final_result.aggregate()
print("\nResultados en conjunto de prueba:")
for idx, row in final_scores.iterrows():
    medida = row['measure']
    valor = row['mean']
    print(f"  {medida}: {valor:.3f}")

# 8. Análisis de predicciones
print("\n8. ANÁLISIS DE PREDICCIONES")
print("-" * 40)

# Entrenar el mejor modelo en todos los datos
mejor_pipeline = pipelines[mejor_modelo_nombre]
mejor_pipeline.train(task)

# Hacer predicciones en una muestra
indices_muestra = list(range(10))
predicciones = mejor_pipeline.predict(task, row_ids=indices_muestra)

print("Muestra de predicciones:")
print("-" * 60)
print(f"{'Índice':<8} {'Real':<10} {'Predicción':<12} {'Correcto':<10}")
print("-" * 60)

for i, idx in enumerate(indices_muestra):
    real = predicciones.truth[i]
    pred = predicciones.response[i]
    correcto = "Sí" if real == pred else "No"
    print(f"{idx:<8} {real:<10} {pred:<12} {correcto:<10}")

# 9. Resumen final
print("\n" + "="*70)
print("RESUMEN DE RESULTADOS")
print("="*70)

print(f"\nDataset:")
print(f"  - Muestras: {n_samples}")
print(f"  - Características: {len(task.feature_names)}")
print(f"  - Tasa de churn: {(data['churn'] == 'Si').mean():.1%}")

print(f"\nComparación de modelos (F1 Score):")
for idx, row in mejores_por_f1.iterrows():
    modelo = row['learner'].replace('pipeline_', '')
    f1 = row['mean_score']
    print(f"  {idx+1}. {modelo:<20} {f1:.3f}")

print(f"\nMejor modelo: {mejor_modelo_nombre}")
print(f"  - Accuracy: {final_scores[final_scores['measure'] == 'classif.acc']['mean'].values[0]:.3f}")
print(f"  - F1 Score: {final_scores[final_scores['measure'] == 'classif.f1']['mean'].values[0]:.3f}")
print(f"  - Precision: {final_scores[final_scores['measure'] == 'classif.precision']['mean'].values[0]:.3f}")
print(f"  - Recall: {final_scores[final_scores['measure'] == 'classif.recall']['mean'].values[0]:.3f}")

# Calcular mejora sobre baseline
baseline_f1 = baseline_scores[baseline_scores['measure'] == 'classif.f1']['mean'].values[0]
mejor_f1 = final_scores[final_scores['measure'] == 'classif.f1']['mean'].values[0]
mejora = ((mejor_f1 - baseline_f1) / baseline_f1 * 100)

print(f"\nMejora sobre baseline: {mejora:+.1f}%")

print(f"\nCompletado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n+ Demo completada exitosamente!")