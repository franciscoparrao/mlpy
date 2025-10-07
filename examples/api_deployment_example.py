"""
Ejemplo de deployment de modelos MLPY con FastAPI.

Este script muestra cómo:
1. Entrenar y registrar un modelo
2. Iniciar el servidor API
3. Hacer predicciones usando el cliente
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import subprocess
import sys
from pathlib import Path

# Agregar el directorio padre al path para importar mlpy
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif
from mlpy.learners import LearnerClassifSklearn
from mlpy.registry import FileSystemRegistry
from mlpy.deploy.client import MLPYClient


def prepare_and_register_models():
    """Prepara y registra modelos de ejemplo en el registry."""
    
    print("=" * 60)
    print("PREPARANDO Y REGISTRANDO MODELOS")
    print("=" * 60)
    
    # Crear registry
    registry = FileSystemRegistry("./example_models")
    
    # 1. Modelo Iris - Logistic Regression
    print("\n1. Entrenando modelo Iris...")
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    
    task_iris = TaskClassif(df_iris, target='target')
    learner_iris = LearnerClassifSklearn(LogisticRegression(max_iter=200))
    learner_iris.train(task_iris)
    learner_iris.task_type = "classification"
    
    # Registrar modelo
    model_v1 = registry.register_model(
        model=learner_iris,
        name="iris_classifier",
        description="Clasificador de especies de Iris",
        author="MLPY Demo",
        tags={"dataset": "iris", "algorithm": "logistic_regression"},
        metrics={"accuracy": 0.95, "f1_score": 0.94}
    )
    print(f"   ✓ Registrado: iris_classifier v{model_v1.metadata.version}")
    
    # 2. Modelo Wine - Random Forest
    print("\n2. Entrenando modelo Wine...")
    wine = load_wine()
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_wine['target'] = wine.target
    
    task_wine = TaskClassif(df_wine, target='target')
    learner_wine = LearnerClassifSklearn(RandomForestClassifier(n_estimators=50, random_state=42))
    learner_wine.train(task_wine)
    learner_wine.task_type = "classification"
    
    model_v2 = registry.register_model(
        model=learner_wine,
        name="wine_classifier",
        description="Clasificador de tipos de vino",
        author="MLPY Demo",
        tags={"dataset": "wine", "algorithm": "random_forest"},
        metrics={"accuracy": 0.98, "f1_score": 0.97}
    )
    print(f"   ✓ Registrado: wine_classifier v{model_v2.metadata.version}")
    
    # Promover modelos a producción
    registry.update_model_stage("iris_classifier", model_v1.metadata.version, 
                               registry.ModelStage.PRODUCTION)
    registry.update_model_stage("wine_classifier", model_v2.metadata.version,
                               registry.ModelStage.PRODUCTION)
    
    print("\n✓ Modelos registrados y promovidos a producción")
    print(f"  - iris_classifier v{model_v1.metadata.version}")
    print(f"  - wine_classifier v{model_v2.metadata.version}")
    
    return registry


def test_client_predictions():
    """Prueba el cliente haciendo predicciones."""
    
    print("\n" + "=" * 60)
    print("PROBANDO CLIENTE DE PREDICCIONES")
    print("=" * 60)
    
    # Crear cliente
    client = MLPYClient(base_url="http://localhost:8000")
    
    # Verificar conexión
    print("\n1. Verificando conexión...")
    health = client.health_check()
    print(f"   ✓ Servidor estado: {health['status']}")
    print(f"   ✓ Modelos cargados: {health['models_loaded']}")
    
    # Listar modelos
    print("\n2. Listando modelos disponibles...")
    models = client.list_models()
    for model in models:
        info = client.get_model_info(model)
        print(f"   - {model}: {info['task_type']} (stage: {info['stage']})")
    
    # Predicción con Iris
    print("\n3. Predicción con iris_classifier...")
    iris_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]
    
    result = client.predict(
        data=iris_data,
        model_name="iris_classifier",
        return_probabilities=True
    )
    
    print(f"   Entrada: {iris_data}")
    print(f"   Predicciones: {result['predictions']}")
    if result.get('probabilities'):
        print(f"   Probabilidades:")
        for i, probs in enumerate(result['probabilities']):
            print(f"      Muestra {i+1}: {[f'{p:.3f}' for p in probs]}")
    print(f"   Tiempo: {result['prediction_time']:.3f} segundos")
    
    # Predicción con Wine
    print("\n4. Predicción con wine_classifier...")
    wine_data = {
        "alcohol": [13.2, 12.5],
        "malic_acid": [1.78, 2.1],
        "ash": [2.14, 2.5],
        "alcalinity_of_ash": [11.2, 12.0],
        "magnesium": [100, 95],
        "total_phenols": [2.65, 2.5],
        "flavanoids": [2.76, 2.4],
        "nonflavanoid_phenols": [0.26, 0.3],
        "proanthocyanins": [1.28, 1.5],
        "color_intensity": [4.38, 5.0],
        "hue": [1.05, 0.95],
        "od280/od315_of_diluted_wines": [3.4, 3.2],
        "proline": [1050, 980]
    }
    
    result = client.predict(
        data=wine_data,
        model_name="wine_classifier"
    )
    
    print(f"   Predicciones: {result['predictions']}")
    print(f"   Modelo usado: {result['model_name']} v{result['model_version']}")
    print(f"   Tiempo: {result['prediction_time']:.3f} segundos")
    
    # Métricas del modelo
    print("\n5. Obteniendo métricas de uso...")
    for model in models:
        metrics = client.get_model_metrics(model)
        print(f"   {model}:")
        print(f"      - Total predicciones: {metrics['total_predictions']}")
        print(f"      - Tiempo promedio: {metrics['avg_prediction_time']:.3f}s")
        print(f"      - Tasa de error: {metrics['error_rate']:.1%}")


def test_batch_predictions():
    """Prueba predicciones en batch."""
    
    print("\n" + "=" * 60)
    print("PROBANDO PREDICCIONES EN BATCH")
    print("=" * 60)
    
    client = MLPYClient(base_url="http://localhost:8000")
    
    # Crear datos de batch
    batch_data = []
    for _ in range(10):
        batch_data.append([
            np.random.uniform(4.3, 7.9),  # sepal length
            np.random.uniform(2.0, 4.4),  # sepal width
            np.random.uniform(1.0, 6.9),  # petal length
            np.random.uniform(0.1, 2.5)   # petal width
        ])
    
    print(f"\nEnviando batch de {len(batch_data)} muestras...")
    
    result = client.predict_batch(
        batch_id="test_batch_001",
        data=batch_data,
        model_name="iris_classifier"
    )
    
    print(f"   Batch ID: {result['batch_id']}")
    print(f"   Estado: {result['status']}")
    print(f"   Predicciones: {result['predictions'][:5]}...")  # Primeras 5
    print(f"   Progreso: {result['progress']}%")


def main():
    """Función principal."""
    
    print("\n" + "=" * 60)
    print("EJEMPLO DE DEPLOYMENT DE MODELOS MLPY")
    print("=" * 60)
    
    # Paso 1: Preparar modelos
    registry = prepare_and_register_models()
    
    # Paso 2: Instrucciones para iniciar el servidor
    print("\n" + "=" * 60)
    print("INSTRUCCIONES PARA INICIAR EL SERVIDOR")
    print("=" * 60)
    print("\nPara iniciar el servidor, ejecuta en otra terminal:")
    print("\n  python -m mlpy.deploy.cli serve --registry-path ./example_models")
    print("\nO con autenticación:")
    print("\n  python -m mlpy.deploy.cli serve --registry-path ./example_models \\")
    print("    --enable-auth --api-key mysecretkey")
    print("\nO directamente con uvicorn:")
    print("\n  uvicorn mlpy.deploy.api:create_app --factory --reload")
    
    # Esperar confirmación
    input("\nPresiona Enter cuando el servidor esté iniciado...")
    
    # Paso 3: Probar cliente
    try:
        test_client_predictions()
        test_batch_predictions()
        
        print("\n" + "=" * 60)
        print("✓ EJEMPLO COMPLETADO CON ÉXITO")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nAsegúrate de que el servidor esté ejecutándose.")
        print("Usa: python -m mlpy.deploy.cli serve --registry-path ./example_models")


if __name__ == "__main__":
    main()