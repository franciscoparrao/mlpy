"""
Ejemplo de integración con OpenML en MLPY.

Este ejemplo muestra cómo:
1. Descargar datasets y tareas de OpenML
2. Ejecutar benchmarks en suites de OpenML
3. Comparar múltiples learners
4. Subir resultados a OpenML (opcional)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.integrations import (
    OpenMLClient,
    download_dataset,
    download_task,
    list_datasets,
    list_tasks,
    get_benchmark_suite
)
from mlpy.learners.sklearn import (
    LearnerSklearnClassif,
    LearnerSklearnRegr
)
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifF1
from mlpy.benchmark import Benchmark


def example_download_dataset():
    """Ejemplo de descarga de dataset de OpenML."""
    print("=" * 60)
    print("DESCARGANDO DATASET DE OPENML")
    print("=" * 60)
    
    # Descargar el famoso dataset Iris
    print("\n1. Descargando dataset Iris por nombre...")
    task_iris = download_dataset(
        name="iris",
        as_task=True,
        task_type="classification"
    )
    
    print(f"   ✓ Dataset descargado: {task_iris.id}")
    print(f"   ✓ Shape: {task_iris.data.shape}")
    print(f"   ✓ Target: {task_iris.target_col}")
    print(f"   ✓ Clases: {task_iris.class_labels}")
    
    # Descargar dataset por ID
    print("\n2. Descargando dataset Wine (ID: 187)...")
    task_wine = download_dataset(
        dataset_id=187,
        as_task=True,
        task_type="classification"
    )
    
    print(f"   ✓ Dataset descargado: {task_wine.id}")
    print(f"   ✓ Shape: {task_wine.data.shape}")
    
    # Descargar solo datos sin crear tarea
    print("\n3. Descargando datos crudos...")
    data, metadata = download_dataset(
        name="boston",
        as_task=False
    )
    
    print(f"   ✓ Datos descargados: {data.shape}")
    print(f"   ✓ Metadata disponible:")
    for key, value in metadata.items():
        if key != 'qualities':  # Evitar imprimir diccionario grande
            print(f"      - {key}: {value}")
    
    return task_iris, task_wine


def example_list_datasets():
    """Ejemplo de listar datasets disponibles."""
    print("\n" + "=" * 60)
    print("LISTANDO DATASETS DISPONIBLES")
    print("=" * 60)
    
    # Listar datasets populares
    print("\n1. Top 10 datasets más populares...")
    datasets = list_datasets()
    
    if not datasets.empty:
        # Mostrar top 10 por número de instancias
        top_datasets = datasets.nlargest(10, 'NumberOfInstances')[
            ['did', 'name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']
        ]
        
        print(top_datasets.to_string())
    
    # Buscar datasets con tag específico
    print("\n2. Datasets con tag 'study_14' (datasets populares)...")
    tagged_datasets = list_datasets(tag='study_14')
    
    if not tagged_datasets.empty:
        print(f"   ✓ Encontrados {len(tagged_datasets)} datasets")
        print(f"   ✓ Primeros 5:")
        for _, row in tagged_datasets.head().iterrows():
            print(f"      - {row['name']} (ID: {row['did']})")


def example_download_task():
    """Ejemplo de descarga de tarea de OpenML."""
    print("\n" + "=" * 60)
    print("DESCARGANDO TAREA DE OPENML")
    print("=" * 60)
    
    # Descargar tarea específica
    print("\n1. Descargando tarea 59 (Iris classification)...")
    task = download_task(task_id=59)
    
    print(f"   ✓ Tarea descargada: {task.id}")
    print(f"   ✓ Tipo: Clasificación")
    print(f"   ✓ Dataset shape: {task.data.shape}")
    print(f"   ✓ Target: {task.target_col}")
    
    return task


def example_list_tasks():
    """Ejemplo de listar tareas disponibles."""
    print("\n" + "=" * 60)
    print("LISTANDO TAREAS DISPONIBLES")
    print("=" * 60)
    
    # Listar tareas de clasificación
    print("\n1. Tareas de clasificación...")
    classification_tasks = list_tasks(task_type='classification')
    
    if not classification_tasks.empty:
        print(f"   ✓ Encontradas {len(classification_tasks)} tareas de clasificación")
        print(f"   ✓ Primeras 5:")
        for _, row in classification_tasks.head().iterrows():
            print(f"      - Task {row['tid']}: {row.get('name', 'N/A')}")
    
    # Listar tareas de regresión
    print("\n2. Tareas de regresión...")
    regression_tasks = list_tasks(task_type='regression')
    
    if not regression_tasks.empty:
        print(f"   ✓ Encontradas {len(regression_tasks)} tareas de regresión")


def example_benchmark_suite():
    """Ejemplo de benchmark suite de OpenML."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUITE DE OPENML")
    print("=" * 60)
    
    # OpenML-CC18: Suite de clasificación popular
    print("\n1. Obteniendo información de OpenML-CC18...")
    
    try:
        suite_info = get_benchmark_suite(suite_id=99)
        
        print(f"   ✓ Suite: {suite_info['name']}")
        print(f"   ✓ Descripción: {suite_info['description']}")
        print(f"   ✓ Número de tareas: {suite_info['n_tasks']}")
        print(f"   ✓ Primeras 5 tareas: {suite_info['task_ids'][:5]}")
        
        return suite_info
    except Exception as e:
        print(f"   ✗ Error obteniendo suite: {e}")
        return None


def example_run_benchmark():
    """Ejemplo de ejecutar benchmark con OpenML."""
    print("\n" + "=" * 60)
    print("EJECUTANDO BENCHMARK")
    print("=" * 60)
    
    # Crear learners para comparar
    learners = [
        LearnerSklearnClassif(model='RandomForestClassifier', n_estimators=10),
        LearnerSklearnClassif(model='GradientBoostingClassifier', n_estimators=10),
        LearnerSklearnClassif(model='LogisticRegression', max_iter=1000)
    ]
    
    # Descargar algunas tareas para benchmark
    print("\n1. Descargando tareas para benchmark...")
    tasks = []
    
    # Usar tareas pequeñas para ejemplo rápido
    task_ids = [59, 61, 3]  # Iris, Iris (otro split), kr-vs-kp
    
    for task_id in task_ids:
        try:
            task = download_task(task_id)
            tasks.append(task)
            print(f"   ✓ Tarea {task_id} descargada")
        except Exception as e:
            print(f"   ✗ Error descargando tarea {task_id}: {e}")
    
    if not tasks:
        print("   ✗ No se pudieron descargar tareas")
        return
    
    # Configurar benchmark
    print("\n2. Configurando benchmark...")
    resampling = ResamplingCV(folds=3)
    measures = [
        MeasureClassifAccuracy(),
        MeasureClassifF1()
    ]
    
    benchmark = Benchmark(
        learners=learners,
        tasks=tasks,
        resamplings=[resampling],
        measures=measures
    )
    
    # Ejecutar benchmark
    print("\n3. Ejecutando benchmark...")
    results = benchmark.run(parallel=False)
    
    # Mostrar resultados
    print("\n4. Resultados del benchmark:")
    print("=" * 60)
    
    # Agrupar por learner y calcular media
    summary = results.groupby('learner').agg({
        'classif.acc': 'mean',
        'classif.f1': 'mean',
        'runtime': 'mean'
    }).round(3)
    
    print(summary)
    
    # Mejor learner por accuracy
    best_learner = summary['classif.acc'].idxmax()
    print(f"\n✓ Mejor learner: {learners[best_learner].__class__.__name__}")
    print(f"  Accuracy promedio: {summary.loc[best_learner, 'classif.acc']:.3f}")
    
    return results


def example_openml_client():
    """Ejemplo de uso del cliente OpenML."""
    print("\n" + "=" * 60)
    print("CLIENTE OPENML AVANZADO")
    print("=" * 60)
    
    # Crear cliente
    client = OpenMLClient()
    
    # 1. Descargar dataset con metadata completa
    print("\n1. Descargando dataset con metadata...")
    data, metadata = client.download_dataset(name="credit-g")
    
    print(f"   ✓ Dataset: {metadata['name']}")
    print(f"   ✓ Versión: {metadata['version']}")
    print(f"   ✓ URL: {metadata['url']}")
    print(f"   ✓ Features categóricas: {len(metadata['categorical_features'])}")
    
    # 2. Descargar tarea con información de splits
    print("\n2. Descargando tarea con splits...")
    task, task_metadata = client.download_task(task_id=31)
    
    print(f"   ✓ Tarea: {task_metadata['task_id']}")
    print(f"   ✓ Tipo: {task_metadata['task_type']}")
    print(f"   ✓ Medida de evaluación: {task_metadata['evaluation_measure']}")
    
    # 3. Listar datasets con filtros
    print("\n3. Buscando datasets de clasificación binaria...")
    binary_datasets = client.list_datasets(
        NumberOfClasses=2,
        output_format='dataframe'
    )
    
    if not binary_datasets.empty:
        print(f"   ✓ Encontrados {len(binary_datasets)} datasets binarios")
        
        # Mostrar algunos ejemplos
        examples = binary_datasets.head(3)[['did', 'name', 'NumberOfInstances']]
        for _, row in examples.iterrows():
            print(f"      - {row['name']} (ID: {row['did']}, Instancias: {row['NumberOfInstances']})")


def main():
    """Función principal."""
    print("\n" + "=" * 60)
    print("EJEMPLO DE INTEGRACIÓN CON OPENML")
    print("=" * 60)
    
    try:
        # 1. Descargar datasets
        task_iris, task_wine = example_download_dataset()
        
        # 2. Listar datasets disponibles
        example_list_datasets()
        
        # 3. Descargar tarea
        task = example_download_task()
        
        # 4. Listar tareas
        example_list_tasks()
        
        # 5. Obtener suite de benchmark
        suite_info = example_benchmark_suite()
        
        # 6. Ejecutar benchmark
        results = example_run_benchmark()
        
        # 7. Uso avanzado del cliente
        example_openml_client()
        
        print("\n" + "=" * 60)
        print("✓ EJEMPLO COMPLETADO CON ÉXITO")
        print("=" * 60)
        
        print("\nLa integración con OpenML permite:")
        print("  • Acceder a miles de datasets curados")
        print("  • Descargar tareas predefinidas")
        print("  • Ejecutar benchmarks estándar")
        print("  • Comparar resultados con la comunidad")
        print("  • Reproducir experimentos publicados")
        
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nPara usar la integración con OpenML, instala:")
        print("  pip install openml")
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()